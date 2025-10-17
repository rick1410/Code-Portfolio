import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import jarque_bera

class DataAnalysis:
    def __init__(self, full_file_path, irregular_file_path):
        
        self.full_file_path = full_file_path
        self.irregular_file_path = irregular_file_path
        self.one_sec_price_data = None
        self.irregular_one_sec_price_data = None
        self.one_sec_logprice_data = None
        self.irregular_one_sec_logprice_data = None
        self.log_returns = None
        self.close_prices = None
        self.realised_vol = None

    def import_data(self):
        self.one_sec_price_data = pd.read_csv(self.full_file_path, parse_dates=['DT'], index_col='DT', memory_map=True)
        self.irregular_one_sec_price_data = pd.read_csv(self.irregular_file_path, parse_dates=['DT'], index_col='DT', memory_map=True)

    def compute_log_prices(self, data):
        return np.log(data["PRICE"]).rename("log_price")

    def close_to_close_returns(self, log_price_data):
        close_prices = log_price_data.resample('1D').last().dropna()
        log_returns = 100 * close_prices.diff().dropna()
        self.close_prices, self.log_returns = close_prices, log_returns
        return log_returns, close_prices

    def realised_volatility(self, irregular_data):
        interval_prices = irregular_data['PRICE'].resample('5T').median()
        log_returns = 100 * np.log(interval_prices).diff().dropna()
        realised_variance = log_returns.groupby(log_returns.index.date).apply(lambda x: np.sum(x**2))
        self.realised_vol = realised_variance.tail(-1)
        return self.realised_vol

    @staticmethod
    def plot_log_returns(log_returns):
        plt.figure(figsize=(36, 24))
        log_returns.plot(title='Log Returns for All Stocks', alpha=0.8)
        plt.xlabel('Date')
        plt.ylabel('Log Return')
        plt.legend(loc = 'upper left')
        plt.show()

    @staticmethod
    def compute_statistics(log_returns):

        desc_stats = log_returns.describe().T  
        # Compute ADF test p-values
        adf_tests = log_returns.apply(lambda x: sm.tsa.adfuller(x.dropna())[1])
        desc_stats["adf_pvalue"] = adf_tests

        # Compute Jarque-Bera test p-values, skewness, and kurtosis
        jb_tests = log_returns.apply(lambda x: jarque_bera(x.dropna())[1])
        desc_stats["jb_pvalue"] = jb_tests
        desc_stats["skewness"] = log_returns.skew()
        desc_stats["kurtosis"] = log_returns.kurt()

        # Select only a few statistics
        stats_df = desc_stats[["count", "mean", "adf_pvalue", "jb_pvalue", "skewness", "kurtosis"]]
        
        print("\n=== Descriptive Statistics ===\n", stats_df.round(3))
        return stats_df
    

def process_stock(stock, data_folder):
    print(f"Processing {stock}...")
    stock_dir = os.path.join(data_folder, stock)
    full_file_path = os.path.join(stock_dir, f"filled_data_{stock}.csv")
    irregular_file_path = os.path.join(stock_dir, f"cleaned_data_{stock}.csv")
    analysis = DataAnalysis(full_file_path, irregular_file_path)
    analysis.import_data()
    analysis.one_sec_logprice_data = analysis.compute_log_prices(analysis.one_sec_price_data)
    analysis.irregular_one_sec_logprice_data = analysis.compute_log_prices(analysis.irregular_one_sec_price_data)
    analysis.log_returns, analysis.close_prices = analysis.close_to_close_returns(analysis.one_sec_logprice_data)
    analysis.realised_vol = analysis.realised_volatility(analysis.irregular_one_sec_price_data)
    
    return {"stock": stock,"log_returns": analysis.log_returns,"realised_vol": analysis.realised_vol,"one_sec_logprice_data": analysis.one_sec_logprice_data,"irregular_one_sec_logprice_data": analysis.irregular_one_sec_logprice_data}
