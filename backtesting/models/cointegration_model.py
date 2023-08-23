# imports
import yfinance as yf
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint

class CointegrationModel():

    def __init__(self, primary: str, secondary: str, start_date: str, end_date: str, interval: str = "1d"):
        """
        :param primary: The primary symbol to be used in the model.
        :param secondary: The secondary symbol to be used in the model.
        :param start_date: The start date of the data to be used in the model.
        :param end_date: The end date of the data to be used in the model.
        :param interval: The time interval of the data to be used in the model.
        Valid intervals are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 
        3mo. (Default is 1d)
        """
        self.primary_symbol = primary
        self.secondary_symbol = secondary
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.hedge_ratio = 1
        # Attempt to get data from yfinance for the primary and secondary symbols.
        print("Initializing data from yfinance...")
        self.__init_data()
        print("Cleaning the data...")
        self.__clean_data()
        print("Checking for a hedge ratio...")
        self.__init_hedge_ratio()

    def __init_data(self):
        """
        Uses yfinance to obtain data for the primary and secondary symbols at the specified
        time interval and for the specified time frame.
        """
        try:
            self.primary_data = yf.download(self.primary_symbol, start=self.start_date, end=self.end_date, interval=self.interval)
            self.secondary_data = yf.download(self.secondary_symbol, start=self.start_date, end=self.end_date, interval=self.interval)
        except Exception as e:
            raise Exception(f"Unable to obtain data from yfinance for the symbols {self.primary_symbol} and {self.secondary_symbol}. Error trace: {e}")

    def __clean_data(self):
        """
        Cleaning the data so the cointegration test can be performed.
        """
        # Drop any rows with missing values.
        self.primary_data.dropna(inplace=True)
        self.secondary_data.dropna(inplace=True)
        # Attempt to merge the dataframes
        self.__merge_dataframes()
            
    def __merge_dataframes(self):
        """
        Merge the dataframes.
        """
        try:
            self.merged_df = pd.merge(self.primary_data, self.secondary_data, on="Date", suffixes=(f"_{self.primary_symbol}", f"_{self.secondary_symbol}"))
        except Exception as e:
            try:
                self.merged_df = pd.merge(self.primary_data, self.secondary_data, on="Datetime", suffixes=(f"_{self.primary_symbol}", f"_{self.secondary_symbol}"))
            except Exception as e:
                raise Exception(f"Unable to merge the dataframes. Error trace: {e}")
    
    def __init_hedge_ratio(self):
        """
        First, check if it is necessary for a hedge ratio.
        If the last price entries of the primary and secondary symbols are more than 1% apart, then a hedge ratio is necessary.
        """
        try:
            last_primary_price = self.merged_df[f'Adj Close_{self.primary_symbol}']
            last_secondary_price = self.merged_df[f'Adj Close_{self.secondary_symbol}']
        except Exception as e:
            raise Exception(f"Unable to obtain the last price entries of the primary and secondary symbols. Error trace: {e}")
        try:
            if abs(last_primary_price / last_secondary_price) > 0.01:
                print("Detected that a hedge ratio is necessary for this pair, calculating...")
                self.__calculate_hedge_ratio()
        except Exception as e:
            print(f"Unable to calculate the hedge ratio. Continuing without one. Error trace: {e}")
        
    def __calculate_hedge_ratio(self):

        """
        Using statsmodels OLS to find the hedge ratio.
        Where the slope of the regression line (line that minimizes the sum of the squared residuals) from OLS is the hedge ratio.
        """
        try:
            model = OLS(self.merged_df[f'Adj Close_{self.primary_symbol}'], self.merged_df[f'Adj Close_{self.secondary_symbol}'])
            results = model.fit()
            hedge_ratio = results.params[0]
            print(f'hedge ratio: {hedge_ratio}')
            self.hedge_ratio = hedge_ratio
        except Exception as e:
            raise Exception(f"Unexpected error when calculating a hedge ratio. This may interfere with other class functionality. Error trace: {e}")

    def run_coint_test(self):
        """
        coint_t: Cointegration test statistic (t-statistic) for the null hypothesis that the two series are not cointegrated.
        pvalue: MacKinnon's approximate p-value
        crit_value: Critical values for the test statistic at the 1 %, 5 %, and 10 % levels.
        """
        try:
            coint_t, pvalue, crit_value = coint(
                self.merged_df[f'Adj Close_{self.primary_symbol}'], self.merged_df[f'Adj Close_{self.secondary_symbol}']
                )
        except Exception as e:
            raise Exception(f"Unexpected error when running the cointegration test. Error trace: {e}")
        # Determine if the p-value is low enough to reject the null hypothesis that the two series are not cointegrated.
        if pvalue < 0.05:
            print(f"The p-value of {pvalue} is sufficient to reject the null hypothesis that the two series are not cointegrated.")
        else:
            print(f"The p-value of {pvalue} is not sufficient to reject the null hypothesis that the two series are not cointegrated.")
        print("Additional information:")
        print(f'cointegration test statistic: {coint_t}')
        print(f'p-value: {pvalue}')
        print(f'critical values: {crit_value}')