#!/usr/bin/env python
# coding: utf-8

# # Rocket Bunny Dashboard Prototype

# FILENAME: rb_visuals.ipynb
# 
# DATE CREATED: 10-APR-21

# ## PHASE 1: Environment Setup

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import getpass as gp

import pandas_ta as ta

from matplotlib import pyplot
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


# Function declaration

# In[2]:


def is_ma_pos(val):
    
    if val > 0:
        return True
    else:
        return False


# In[3]:


def numpy_ewma_vectorized(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


# In[4]:


def rma(x, n, y0):
        a = (n-1) / n
        ak = a**np.arange(len(x)-1, -1, -1)
        return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]


# Class definition

# ## PHASE 2: Data ETL

# Read in the raw data

# In[5]:




raw_data = pd.read_csv("data/bunny-usd-max.csv")
#data = data.query("type == 'conventional' and region == 'Albany'")
#raw_data["date"] = pd.to_datetime(raw_data["date"], format="%Y-%m-%d")

#data.sort_values("Date", inplace=True)

raw_data.head(5)


# In[6]:


app = dash.Dash(__name__)


# In[7]:



data = raw_data.rename({'snapped_at': 'dtg'}, axis=1)
data.dtypes


# In[8]:


col_list = list(data.columns)
col_list


# In[9]:


close_np = data['price'].to_list()
close_np


# In[10]:


data['date'] = pd.to_datetime(data['dtg']).dt.date
data['year'] = pd.DatetimeIndex(data['date']).year
data['month'] = pd.DatetimeIndex(data['date']).month
data['day'] = pd.DatetimeIndex(data['date']).day
data['hour'] = pd.DatetimeIndex(data['date']).hour.astype(str)


# In[11]:


data


# Calculate 'gain' and 'loss' columns

# In[12]:


data['close_diff'] = data['price'].diff()
data['bool_close_diff'] = data['close_diff'].apply(is_ma_pos)


# In[13]:


data['gain'] = data.close_diff.mask(data.close_diff < 0, 0.0)
data['loss'] = -data.close_diff.mask(data.close_diff > 0, -0.0)

n=14 # rsi window number
data['avg_gain'] = rma(data.gain[n+1:].to_numpy(), n, np.nansum(data.gain.to_numpy()[:n+1])/n)
data['avg_loss'] = rma(data.loss[n+1:].to_numpy(), n, np.nansum(data.loss.to_numpy()[:n+1])/n)


# Calculate rolling mean

# In[14]:


data['SMA7'] = data['price'].rolling(window=7).mean()
data['SMA30'] = data['price'].rolling(window=30).mean()


# Calculate rolling standard deviation

# In[15]:


data['STD7'] = data['price'].rolling(window=7).std()
data['STD30'] = data['price'].rolling(window=30).std()

#data['STD99'] = self.aggregate_df['Close'].rolling(window=99).std()


# Calculate rolling variance

# In[16]:


data['VAR7'] = data['price'].rolling(window=7).var()
data['VAR30'] = data['price'].rolling(window=30).var()

#data['VAR99'] = self.aggregate_df['Close'].rolling(window=99).var()


# Calculate MACD

# In[17]:


data['SMA-MACD'] = data['SMA7'] - data['SMA30']
data['SMA-MACD-ratio'] = data['SMA7'] / data['SMA30']
data['bool_diff_SMA7_SMA30'] = data['SMA-MACD'].apply(is_ma_pos)


# Calculate EMA

# In[18]:


window = 12


# In[19]:


data["EMA12"] = ta.ema(data["price"], length=12)
data["EMA20"] = ta.ema(data["price"], length=20)
data


# In[20]:


data.head(5)


# In[21]:


col_list = list(data.columns)
col_list


# ### ARIMA

# In[22]:


price_series = data['price'].values
price_series

# split into train and test sets
X = price_series
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
# ## PHASE 3: Data Visualization

# Simple Moving Average (SMA) Visualization

# In[23]:


sma_fig = go.Figure()
sma_fig.add_trace(go.Scatter(x=data['date'], y=data['price'],
                    mode='lines',
                    name='Price Timeseries'))
sma_fig.add_trace(go.Scatter(x=data['date'], y=data['SMA7'],
                    mode='lines',
                    name='Timeseries SMA7'))
sma_fig.add_trace(go.Scatter(x=data['date'], y=data['SMA30'],
                    mode='lines', name='Timeseries SMA30'))

sma_fig.update_layout(
    title="Rocket Bunny Close Price and Simple Moving Average",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Legend Title",
    font=dict(
        family="Helvetica",
        size=13,
        color="Black"
    )
)
sma_fig.show()


# Closs differential

# In[24]:


close_fig = go.Figure()
close_fig.add_trace(go.Scatter(x=data['date'], y=data['close_diff'],
                    mode='lines',
                    name='Price Timeseries'))

close_fig.update_layout(
    title="Rocket Bunny Close Differential Timeseries",
    xaxis_title="Date",
    yaxis_title="Daily Closing Price differential",
    legend_title="Legend Title",
    font=dict(
        family="Helvetica",
        size=13,
        color="Black"
    )
)
close_fig.show()


# Gain / Loss visualization 

# In[25]:


gain_loss_fig = go.Figure()
gain_loss_fig.add_trace(go.Scatter(x=data['date'], y=data['gain'],
                    mode='lines',
                    name='Daily Gain Timeseries'))
gain_loss_fig.add_trace(go.Scatter(x=data['date'], y=data['loss'],
                    mode='lines',
                    name='Daily Loss Timeseries'))

gain_loss_fig.update_layout(
    title="Rocket Bunny Daily Change in Gain & Loss",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Legend Title",
    font=dict(
        family="Helvetica",
        size=13,
        color="Black"
    )
)
gain_loss_fig.show()


# Exponential Moving Average Graph

# In[26]:


ema_fig = go.Figure()
ema_fig.add_trace(go.Scatter(x=data['date'], y=data['price'],
                    mode='lines',
                    name='Price Timeseries'))
ema_fig.add_trace(go.Scatter(x=data['date'], y=data['EMA12'],
                    mode='lines',
                    name='Timeseries EMA 12'))
ema_fig.add_trace(go.Scatter(x=data['date'], y=data['EMA20'],
                    mode='lines', name='Timeseries EMA 20'))

ema_fig.update_layout(
    title="Rocket Bunny Close Price and Exponential Moving Average",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Legend Title",
    font=dict(
        family="Helvetica",
        size=13,
        color="Black"
    )
)
ema_fig.show()


# # DASH DASHBOARD

# In[27]:


app.layout = html.Div(
    children=[
        html.H1(children="Rocket Bunny Analytics",),
        html.P(
            children="Analyze the behavior of Rocket Bunny"
            " spanning the past thirty (30) days",
        ),
        dcc.Graph(
            figure=sma_fig
        ),
        dcc.Graph(
            figure=close_fig
        ),
        dcc.Graph(
            figure=gain_loss_fig
        ),
        dcc.Graph(
            figure=ema_fig
        )
    ]
)


# In[1]:


sma_fig.write_html("visuals/sma.html")
sma_fig.write_html("visuals/close_fig.html")
sma_fig.write_html("visuals/gain_loss_fig.html")
sma_fig.write_html("visuals/ema_fig.html")

if __name__ == "__main__":
    app.run_server(debug=True)
# In[ ]:


get_ipython().system('jupyter nbconvert --to script rb_visuals.ipynb')


# # END OF PROGRAM

# "HIS POWER LEVEL IS OVER 9000?!?!?!"
# 
# - Vegeta

# In[ ]:




