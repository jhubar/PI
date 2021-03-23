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

#
#   Disclaimer: within a csv we can only import data from a same marketplace
#   otherwise there are problem with dates
#

#   What I added: -JSON file to translate index to company name
#                 -Just run the python script and change the company tuple
#                  it produces a csv with values and variation (in decimal!)
#                  (variation ===> today = yesterday + (variation . yesterday)


companies = ('DAL', 'AAL', 'UAL', 'ZNH', 'CEA')

start_date = datetime(2019, 1, 1)
end_date = datetime(2021, 1, 1)

data = pd.DataFrame()

set_date = True
for company_ in companies:

    if set_date:
        data['Date'] = (yf.download(company_, start=start_date,
                                    end=end_date, show_nontrading=True)).index.values
        set_date = False

    data["{}".format(company_)] = ((yf.download(company_, start=start_date,
                                                end=end_date, show_nontrading=True)["Open"] + \
                                    yf.download(company_, start=start_date,
                                                end=end_date, show_nontrading=True)["Close"]) / 2).to_numpy()

    data["{}-var".format(company_)] = np.zeros(len(data["{}".format(company_)]))
    for i in range(1, len(data["{}".format(company_)])):
        diff = data["{}".format(company_)][i] - data["{}".format(company_)][i-1]
        data["{}-var".format(company_)][i] = diff / data["{}".format(company_)][i-1]


data.to_csv("csv_files/{}.csv".format(datetime.now()))

fig = px.line(data, x='Date', y=data.columns, title='Time Series of our Companies')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

