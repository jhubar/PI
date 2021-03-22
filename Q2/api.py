from yahoo_fin.stock_info import get_data
import yahoo_fin.stock_info as si
import matplotlib.pyplot as plt
import seaborn
from pandas_datareader import data
import pandas as pd
import numpy as np
import plotly.express as px


from datetime import datetime, date, timedelta
import yfinance as yf
import mplfinance as mpf


def daterange(start_date, end_date):
    dt = []
    for n in range(int((end_date - start_date).days)):
         dt.append(start_date + timedelta(n))

    return dt


start_date = datetime(2019, 1, 1)
end_date = datetime(2021, 3, 15)

data = pd.DataFrame()
companies = ("WMT", "GB", "BA", "LMT", "AIR", "RTX")

set_date = True
for company_ in companies:

    data['Date'] = (yf.download(company_, start=start_date,
                                                end=end_date, show_nontrading=True)).index.values
    set_date = False

    data["{}".format(company_)] = ((yf.download(company_, start=start_date,
                                                end=end_date, show_nontrading=True)["Open"] + \
                                  yf.download(company_, start=start_date,
                                              end=end_date, show_nontrading=True)["Close"]) / 2).to_numpy()


print(data)

fig = px.line(data, x='Date', y=data.columns, title='Time Series of our Companies')
fig.update_xaxes(rangeslider_visible=True)
fig.show()
# print(data["Date"])








# # Define the instruments to download. We would like to see Apple, Microsoft and the S&P500 index.
# tickers = ['amzn']

# # We would like all available data from 01/01/2000 until 12/31/2016.
# start_date = '2019-01-01'
# end_date = '2021-01-01'

# # User pandas_reader.data.DataReader to load the desired data. As simple as that.
# panel_data = data.DataReader('INPX', 'yahoo', start_date, end_date)

# # print(panel_data)

# # Getting just the adjusted closing prices. This will return a Pandas DataFrame
# # The index in this DataFrame is the major index of the panel_data.
# close = panel_data['Close']

# print(close)

# # Getting all weekdays between 01/01/2019and 01/01/2021
# all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')

# # How do we align the existing prices in adj_close with our new set of dates?
# # All we need to do is reindex close using all_weekdays as the new index
# close = close.reindex(all_weekdays)

# # Reindexing will insert missing values (NaN) for the dates that were not present
# # in the original set. To cope with this, we can fill the missing by replacing them
# # with the latest available price for each instrument.
# close = close.fillna(method='ffill')

# # print(close.head(10))

# # print(all_weekdays)


# plt.plot(close.index, close, label='AMZN')
# plt.xlabel('Date')
# plt.ylabel('Closing price ($)')
# plt.legend()

# plt.show()


# amazon_weekly= get_data("amzn", start_date="01/01/2019", end_date="01/01/2021", index_as_date = True, interval="1wk")

# print(amazon_weekly)

# # ticker_list = ["amzn", "aapl"]
# # historical_datas = {}
# # for ticker in ticker_list:
# #     historical_datas[ticker] = get_data(ticker)

# # print(historical_datas['aapl'])

# amazon_weekly['Close'].plot(figsize=(16, 9))




# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import requests
# import json
# from datetime import datetime



