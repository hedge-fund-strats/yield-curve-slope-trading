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
import module_joaquin as mod


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



