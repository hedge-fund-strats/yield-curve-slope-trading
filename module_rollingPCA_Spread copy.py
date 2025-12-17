# ----------------------------------------------------------------------------------------- #

# Implementation of rolling PCA strategy backtesting

'''
 1. Class: SpreadTradesRollingPCA: get series of weights for the spreads short vs long
    Member functions:
    1.1 spread_weights_rollPCA(): compute loadigns for the spread
    1.2 zscore_signals(): using above weights get z-score (train) and signals (test)
    1.3 trades_pnl_vol(): based on signals and trade horizon, get backtested pnl and vol
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


    def zscore_signals(self, training_size, test_size):
        '''Compute Z-score for defined train/test samples based on self.weights'''
        df_rates = self.df.sort_values(by='Trade Day')
        z_score_records = []
        for cal_date, weights_row in self.df_weights.iterrows():
            current_weights = weights_row.values
            # --- Define Training Sample Boundaries ---
            # The training sample ends one period BEFORE the calibration date
            try:
                cal_idx = df_rates.index.get_loc(cal_date)
            except KeyError:
                continue 
            # Training sample start index: [cal_idx - training_size] 
            start_idx = cal_idx - training_size 
            # Training sample end index: [cal_idx - 1] (i.e., the last observation used for calibration)
            end_idx = cal_idx - 1 
            if start_idx < 0 or end_idx < 0:
                print(f"Skipping {cal_date}: Insufficient lookback data ({training_size} periods needed).")
                continue
            # Select the training rates data
            df_train = df_rates.iloc[start_idx : end_idx + 1][self.instruments]

            # --- Compute Training Sample Statistics ---
            # Portfolio_Value = w_1 * Short_R + w_2 * Long_R
            portfolio_value_train = np.dot(df_train.values, current_weights)

            # Store the mean and standard deviation
            mu_train = portfolio_value_train.mean()
            sigma_train = portfolio_value_train.std()
            
            if sigma_train == 0:
                # Avoid division by zero if the data is constant
                print(f"Warning: Zero standard deviation for {cal_date}. Skipping.")
                continue
            
            # --- Define Test Sample Boundaries and Compute Z-Scores ---
            # Test sample starts AT the calibration date and goes forward
            test_start_idx = cal_idx
            test_end_idx = cal_idx + test_size - 1 # Last observation index
            
            df_test = df_rates.iloc[test_start_idx : test_end_idx + 1][self.instruments]
            
            portfolio_value_test = np.dot(df_test.values, current_weights)
            
            z_scores_test = (portfolio_value_test - mu_train) / sigma_train # based on train mu/sigma.
            
            test_dates = df_rates.index[test_start_idx : test_end_idx + 1]
            
            for trade_date, z_score in zip(test_dates, z_scores_test):
                z_score_records.append({
                    'Trade_date': trade_date,
                    'Z_Score': z_score,
                    'Calibration_Date': cal_date,
                    'mu_train': mu_train,
                    'sigma_train': sigma_train
                })
                
        df_signal = pd.DataFrame(z_score_records).sort_values(by='Trade_date').drop_duplicates(subset=['Trade_date'], keep='first')
        self.df_signal = df_signal
        print("------------------------------------------")
        print("------ Signals Succesfuly Computed -----")
        print("------------------------------------------")
        return df_signal
    
    
    
    def trades_pnl_vol(self, z_score_threshold, exit_z_score_threshold, trade_period):
        '''Create trade entries/exit based on Z-score'''
        """
        Generates a trade log by filtering Z-score signals, applying mean-reversion logic to 
        determine trade direction, and computing the realized PnL.
        Parameters:

        - z_score_threshold (float): Minimum absolute Z-score required to initiate a trade.
        - exit_periods (int): Number of periods the trade is held for PnL calculation.
        - instruments (list): List of tenor columns used in the trade, matching the weight columns.

        Returns:
        - pd.DataFrame: A detailed trade log with entry, exit, initial value, and PnL.
        """

        # Prepare Data and Filter Signals
        
        # Ensure rates are indexed for fast lookup
        df_rates_indexed = self.df.sort_values(by='Trade Day')
        
        # Filter for high-conviction signals
        df_trades = self.df_signal[self.df_signal['Z_Score'].abs() >= z_score_threshold].copy()
        
        # Merge weights (Signal date corresponds to the weight's calibration date)
        df_trades = df_trades.merge(
            self.df_weights.reset_index().rename(columns={'index': 'Calibration_Date'}),
            left_on='Calibration_Date',
            right_on='Calibration_Date',
            how='left'
        )
        weight_cols = ['w_1', 'w_2']
        df_trades.dropna(subset=weight_cols, inplace=True)
        # Compute Entry/Exit Dates and Trade Values
        
        trade_log_records = []
        active_dates = set()

        if df_trades.empty:
            print("No trades generated based on the Z-score threshold.")
            return pd.DataFrame()

        for index, trade in df_trades.iterrows():
            signal_date = trade['Trade_date']
            if signal_date in active_dates:
                continue
            try:
                entry_idx = df_rates_indexed.index.get_loc(signal_date) + 1
                if entry_idx >= len(df_rates_indexed): continue
                entry_date = df_rates_indexed.index[entry_idx]
                R_entry = df_rates_indexed.loc[entry_date, self.instruments].values
                
                mu_trade =  df_trades[df_trades["Trade_date"] == signal_date]["mu_train"].values
                sigma_trade = df_trades[df_trades["Trade_date"] == signal_date]["sigma_train"].values
            
            except Exception as e:
                print(f"Passing Itearation, error {e}")
                continue
            
            # Determine Trade Direction Multiplier
            z_score = trade['Z_Score']
            # Mean-Reversion Logic:
            # Z > 0 (Expensive) -> Expect price drop (Short Short Rate) -> Multiplier = -1
            # Z < 0 (Cheap) -> Expect price rise (Long Short Rate) -> Multiplier = +1
            direction_multiplier = -np.sign(z_score) 
            
            # Calculate Traded Weights
            calibrated_weights = trade[weight_cols].values
            traded_weights = calibrated_weights * direction_multiplier
            
            # Get Entry/Exit Rates
        
            # Value of the traded portfolio (signed weights) at entry rates
            future_rates = df_rates_indexed.iloc[entry_idx:entry_idx+trade_period][self.instruments]
            portfolio_evaluation = np.dot(future_rates.values, traded_weights)
            portfolio_evaluation_diff = pd.Series(portfolio_evaluation, index = future_rates.index).diff()


            z_scores_future = (portfolio_evaluation - mu_trade)/ sigma_trade
            z_scores_future = pd.Series(
                    z_scores_future,
                    index=future_rates.index
                )
            exit_condition = (z_scores_future[:-1].abs() < exit_z_score_threshold) #evaluate criteria up to trade period

            first_exit_idx = exit_condition.values.argmax()-1
            if first_exit_idx == 0 and not exit_condition.iloc[0]:
                exit_date = z_scores_future.index[trade_period-1] # Unwind position at max horizon
                R_exit = df_rates_indexed.loc[exit_date, self.instruments].values
                vol_trade = portfolio_evaluation_diff[:trade_period].std() # Daily volatility
                
            else:
                exit_date = z_scores_future.index[first_exit_idx] # Unwind one-day after returning to threshold
                R_exit = df_rates_indexed.loc[exit_date, self.instruments].values
                vol_trade = portfolio_evaluation_diff[:first_exit_idx].std()  # Dialy volatility

            # Compute PnL
            initial_trade_value = np.dot(R_entry, traded_weights)
            if R_exit is not None:
                # PnL = (Exit Rates - Entry Rates) * Traded Weights
                pnl_rates = R_exit - R_entry
                pnl = np.dot(pnl_rates, traded_weights)
                
            else:
                pnl = np.nan
                vol_trade = np.nan
            
            # Determine Trade Type for reporting
            # The calibrated weights (w_1, w_2) determine the underlying trade type (Steepener vs Flattener)
            # We trade the inverse if Z > 0.
            is_steepener = (np.sign(calibrated_weights[0]) != np.sign(calibrated_weights[1]))
            
            if is_steepener:
                # Calibrated weights are a Steepener (Long short-end, short long-end)
                trade_type = 'Steepener' if direction_multiplier > 0 else 'Flattener'
            else:
                # Calibrated weights are a Flattener (Short short-end, long long-end)
                trade_type = 'Flattener' if direction_multiplier > 0 else 'Steepener' # Should not happen if PC2 is slope

            # Store the log
            trade_log_records.append({
                'Trade_date_Entry': entry_date,
                'Trade_date_Exit': exit_date,
                'Signal_Z_Score': z_score,
                'Direction_Multiplier': direction_multiplier,
                'Trade_Type': trade_type,
                'w_shortR_Traded': traded_weights[0],
                'w_longR_Traded': traded_weights[1],
                'Initial_Trade_Value': initial_trade_value,
                'PnL': pnl,
                'Vol_daily': vol_trade
            })

        # Finalize Output
        df_trade_log = pd.DataFrame(trade_log_records).sort_values(by='Trade_date_Entry')
        print("------------------------------------------")
        print("---------- Trade PnL and Vol -----------")
        print("Total Trades = {a:.1f}".format(a = len(df_trade_log)))
        print("Average PnL (bps) = {a:.2f}".format(a = df_trade_log['PnL'].mean()*100))
        print("Average Vol (bps) = {a:.2f}".format(a = df_trade_log['Vol_daily'].mean()*100))
        print("------------------------------------------")
        print("------ Steepeners PnL and Vol -----------")
        print("Total Trades = {a:.1f}".format(a = len(df_trade_log[df_trade_log['Trade_Type'] == 'Steepener'])))
        print("Average PnL (bps) = {a:.2f}".format(a = df_trade_log[df_trade_log['Trade_Type'] == 'Steepener']['PnL'].mean()*100))
        print("Average Vol (bps) = {a:.2f}".format(a = df_trade_log[df_trade_log['Trade_Type'] == 'Steepener']['Vol_daily'].mean()*100))
        print("------------------------------------------")
        print("------ Steepeners PnL and Vol -----------")
        print("Total Trades = {a:.1f}".format(a = len(df_trade_log[df_trade_log['Trade_Type'] == 'Flattener'])))
        print("Average PnL (bps) = {a:.2f}".format(a = df_trade_log[df_trade_log['Trade_Type'] == 'Flattener']['PnL'].mean()*100))
        print("Average Vol (bps) = {a:.2f}".format(a = df_trade_log[df_trade_log['Trade_Type'] == 'Flattener']['Vol_daily'].mean()*100))
        print("------------------------------------------")
        self.df_trade_log = df_trade_log
        return df_trade_log




def plot_backtest_summary(df_backtest):
    # Ensure dates are in datetime format
    df_backtest['Trade_date'] = pd.to_datetime(df_backtest['Trade_date'])
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, 
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