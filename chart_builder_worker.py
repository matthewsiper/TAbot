import sys,os
BASE_DIR = os.path.join(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
print(BASE_DIR)

import datetime as dt

from config import Config
from constants import PRED_CHART_DIR

import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates

style.use('ggplot')
CONFIG = Config("chart_builder_params")


class ChartBuilderWorker(object):
    def __init__(self):
        self.chart_home_dir = CONFIG.params["chart_home_dir"]

    def run(self, pattern=None, ticker='', mode="train"):
        try:
            hist_df = self.get_price_data(ticker)
            if len(hist_df) > 0:
                self.build_chart(pattern, hist_df)
        except:
            msg = f'Error in run for ticker {ticker}'
            raise Exception(msg)

    def get_price_data(self, ticker):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="120d")
            return hist
        except:
            msg = "Error in get_price_data"
            raise Exception(msg)

    def build_chart(self, pattern, hist, file_num=0, ticker=None, alternate_path=None):
        try:
            #TODO: remove x-axis and y-axis
            hist_copy = hist.copy()
            hist_copy.reset_index(inplace=True)
            hist_copy.rename(columns={"time": "Date"}, inplace=True)
            hist_copy['Date'] = hist_copy['Date'].map(mdates.date2num)
            ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
            ax1.xaxis_date()
            candlestick_ohlc(ax1, hist_copy.values, width=2, colorup='g')
            if not alternate_path:
                if pattern == "live":
                    plt.savefig(f"{PRED_CHART_DIR}/{ticker}_{file_num}.jpg")
                else:
                    plt.savefig(f"{pattern}/{pattern}_{file_num}.jpg")
            else:
                plt.savefig(f"{alternate_path}/{ticker}_{file_num}.jpg")

        except:
            msg = "Error in build_chart"
            raise Exception(msg)


