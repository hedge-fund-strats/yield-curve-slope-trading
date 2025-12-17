# ----------------------------------------------------------------------------------------- #

# Implementation of rolling PCA strategy backtesting

'''
 1. Class: SpreadTradesRollingPCA: get series of weights for the spreads short vs long
    Member functions:
    1.1 spread_weights_rollPCA(): compute loadigns for the spread
    1.2 zscore_spread_trades(): using above weights get z-score on signals and trade horizon, 
    get backtested pnl and vol
 '''

# ----------------------------------------------------------------------------------------- # 


import pandas as pd
import numpy as np
import module_rollingPCA_EMA as mod
import matplotlib.pyplot as plt

class SpreadTradesRollingPCA:
    def __init__(self, df, sample_size_corr, sample_size_vol, instruments):
        self.df = df
        self.sample_size_corr = sample_size_corr
        self.sample_size_vol = sample_size_vol
        self.instruments = instruments

    def spread_weigths_rollPCA(self):
        '''Compute the rolling PCA and get w_1 and w_2'''
        _, rolling_pca = mod.rolling_pca(self.df, self.sample_size_corr, self.sample_size_vol, self.sample_size_vol) #recalibration of PCA based on vol window.
        weights_list = []
        for date, data in rolling_pca.items():
            loadings = data.get('loadings')[:,self.df.columns.get_indexer(self.instruments)]
            try: 
                w_1_normalized = 1
                w_2_normalized = - loadings[0][0]/loadings[0][1] 
                weights_list.append({
                    'Calibration_Date': date,
                    'w_1': w_1_normalized,
                    'w_2': w_2_normalized
                })
            except Exception as e:
                print(f"Warning: {e} in {date}. Skipping.")
                weights_list.append({
                    'Calibration_Date': date,
                    'w_1': np.nan,
                    'w_2': np.nan
                })

        df_weights = pd.DataFrame(weights_list).set_index('Calibration_Date')
        print("------------------------------------------")
        print("--- Rolling Factor Weights Retrieved ---")
        print("--- Most recent calibration ---")
        print(df_weights.tail())
        self.df_weights = df_weights
        return df_weights
    
    def zscore_spread_trades(self, window, entry_z, exit_z):
        df_rates = self.df.sort_values('Trade Day')
        df_weights = self.df_weights.sort_index()
        update_dates = df_weights.index.tolist()
        all_results = []
        current_pos = 0.0
        # Iterate through weight periods
        for i in range(len(update_dates)):
            start_date = update_dates[i]
            # End date is the day before the next update, or the end of the rates file
            end_date = update_dates[i+1] if i+1 < len(update_dates) else df_rates.index[-1]
            
            # Get current weights
            current_weights = df_weights.loc[start_date].values

            #Portfolio series at current rates:
            simple_spread_series = df_rates[self.instruments].dot([1,-1])
            full_portfolio_series = df_rates[self.instruments].dot(current_weights)
            full_price_changes = full_portfolio_series.diff() 
            
            # Determine the data needed: (start_date - window) to have a full Z-score on day 1
            roll_mean = full_portfolio_series.rolling(window=window).mean()
            roll_std = full_portfolio_series.rolling(window=window).std()
            roll_z = (full_portfolio_series - roll_mean) / roll_std
            
            trading_dates = df_rates.index[(df_rates.index >= start_date) & (df_rates.index <= end_date)]
        
            period_positions = []
            period_pnls = []
            
            for date in trading_dates:
                z = roll_z.loc[date]
                price_change = full_price_changes.loc[date]
                
                # --- PnL Calculation ---
                daily_pnl = current_pos * price_change
                period_pnls.append(daily_pnl)

                if current_pos == 0:
                    if z >= entry_z:
                        current_pos = -1  # Mean reversion: Short if expensive (short short rate)
                    elif z <= -entry_z:
                        current_pos = 1   # Mean reversion: Long if cheap (long short rate)
                else:
                    # Exit condition
                    if abs(z) < exit_z:
                        current_pos = 0
                
                period_positions.append(current_pos)
                
            period_df = pd.DataFrame({
                'Z_Score': roll_z.loc[trading_dates],
                'Position': period_positions,
                'Portfolio_val':full_portfolio_series.loc[trading_dates],
                'Simple_spread': simple_spread_series.loc[trading_dates],
                'Daily_PnL': period_pnls,
                'Mean': roll_mean,
                'Sigma': roll_std,
                'w_S': current_weights[0],
                'w_L': current_weights[1],
                'Calibration_Date': start_date
            }, index=trading_dates)
            all_results.append(period_df)
        
        # Combine all periods into one continuous signal dataframe
        
        df_signal = pd.concat(all_results)
        df_signal = pd.DataFrame(df_signal)
        df_signal.index.name = 'Trade Day'
        df_signal['Trade_date'] = df_signal.index
        df_signal.loc[-1,'Position'] = 0 # Close position if any at the end
        df_signal['Cumulative_PnL'] = df_signal['Daily_PnL'].cumsum()
        df_signal['Trade_ID'] = (df_signal['Position'] != df_signal['Position'].shift()).cumsum()

        self.df_signal = df_signal    
        print("-"*35)
        print("Signals, Positions and P&L done") 
        print("-"*35)
        return df_signal
    

    def summarize_strategy_performance(self):
        # 1. Identify Discrete Trades
        df = self.df_signal.copy()
        
        # Filter out periods where position is 0 (no trade)
        df_active = df[df['Position'] != 0].copy()
        
        # 2. Define Trade Type (Steepener vs Flattener)
        # Long Position (1) on a spread where w_short is positive = Steepener
        def identify_type(row):
            # A standard steepener usually has a positive weight on the short tenor
            is_pos_short_weight = row['w_S'] > 0
            if row['Position'] == 1:
                return 'Steepener' if is_pos_short_weight else 'Flattener'
            else:
                return 'Flattener' if is_pos_short_weight else 'Steepener'

        df_active['Trade_Type'] = df_active.apply(identify_type, axis=1)
        df_active['Year'] = pd.to_datetime(df_active.index).year
        df_active['Trade_date'] = df_active.index
        
        # 3. Aggregate data by Trade_ID
        trade_metrics = df_active.groupby(['Trade_ID', 'Trade_Type', 'Year', 'Position']).agg(
            Total_PnL=('Daily_PnL', 'sum'),
            Daily_Vol=('Daily_PnL', 'std'),
            Duration=('Trade_date', 'count')
        ).reset_index()

        # 4. Helper Function for Stats
        def get_stats(group):
            wins = group[group['Total_PnL'] > 0]
            losses = group[group['Total_PnL'] <= 0]
            return pd.Series({
                'Count': len(group),
                'Wins': len(wins),
                'Losses': len(losses),
                'Avg_Holding_Period': round(group['Duration'].mean()),
                'Avg_Return_per_Trade': group['Total_PnL'].mean(),
                'Max. Return':group['Total_PnL'].max(),
                'Min. Return':group['Total_PnL'].min(),
                'Avg_Vol_per_Trade': group['Daily_Vol'].mean(),
                'Avg_Sharpe_Ratio': group['Total_PnL'].mean()/group['Daily_Vol'].mean(),
                'Win_Rate': len(wins) / len(group) if len(group) > 0 else 0
            })

        # Compute Summaries
        #Summary all:
        summary_all = get_stats(trade_metrics)
        # Summary by Trade Type (Steepener/Flattener)
        summary_type = trade_metrics.groupby('Trade_Type').apply(get_stats)
        
        # Summary by Calendar Year
        summary_year = trade_metrics.groupby('Year').apply(get_stats)
        
        # Summary by Side (Long/Short)
        trade_metrics['Side'] = trade_metrics['Position'].map({1: 'Long', -1: 'Short'})
        summary_side = trade_metrics.groupby('Side').apply(get_stats)

        return {
            'All': summary_all,
            'By_Type': summary_type,
            'By_Year': summary_year,
            'By_Side': summary_side,
            'Raw_Trades': trade_metrics
        }


def plot_backtest_summary(df_backtest):
    # Ensure dates are in datetime format
    df_backtest['Trade_date'] = pd.to_datetime(df_backtest['Trade_date'])
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                gridspec_kw={'height_ratios': [1.5, 1]})

    # --- 1. Z-Score and Signal Markers ---
    ax1.plot(df_backtest['Trade_date'], df_backtest['Z_Score'], color='gray', alpha=0.5, label='Z-Score')
    
    # Detect position changes for markers
    # prev_pos is the position from the previous row
    df_backtest['prev_pos'] = df_backtest['Position'].shift(1).fillna(0)
    
    # Long Entry (0 -> 1)
    l_entry = df_backtest[(df_backtest['Position'] == 1) & (df_backtest['prev_pos'] == 0)]
    ax1.scatter(l_entry['Trade_date'], l_entry['Z_Score'], color='green', marker='^', s=100, label='Long Entry')
    
    # Short Entry (0 -> -1)
    s_entry = df_backtest[(df_backtest['Position'] == -1) & (df_backtest['prev_pos'] == 0)]
    ax1.scatter(s_entry['Trade_date'], s_entry['Z_Score'], color='red', marker='v', s=100, label='Short Entry')
    
    # Exits (Non-zero -> 0)
    exits = df_backtest[(df_backtest['Position'] == 0) & (df_backtest['prev_pos'] != 0)]
    ax1.scatter(exits['Trade_date'], exits['Z_Score'], color='black', marker='o', facecolors='none', s=80, label='Exit')

    ax1.set_title('Z-Score Mean Reversion and Trade Signals', fontsize=14)
    ax1.set_ylabel('Z-Score')
    ax1.axhline(0, color='black', lw=1, alpha=0.3)
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.2)

    # --- 2. Cumulative PnL with Shading ---
    ax2.plot(df_backtest['Trade_date'], df_backtest['Cumulative_PnL'], color='black', lw=1.5)
    
    # Apply conditional shading
    ax2.fill_between(df_backtest['Trade_date'], df_backtest['Cumulative_PnL'], 0, 
                    where=(df_backtest['Cumulative_PnL'] >= 0), color='green', alpha=0.2, interpolate=True)
    ax2.fill_between(df_backtest['Trade_date'], df_backtest['Cumulative_PnL'], 0, 
                    where=(df_backtest['Cumulative_PnL'] < 0), color='red', alpha=0.2, interpolate=True)

    ax2.set_title('Cumulative PnL - Mean Reversion Strategy', fontsize=14)
    ax2.set_ylabel('PnL (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()