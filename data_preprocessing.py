import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import pandas as pd

import yfinance as yf
import pendulum

# fetch history series
stock = 'GOOG' # TSLA, AAPL, GOOG, META, MSFT, AMZN, DIS, IBM etc
# price_history = yf.Ticker(stock).history(period='10y', # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#                                    interval='1d', # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
#                                    actions=False)
price_history = yf.Ticker(stock).history(start="2012-11-30", end="2020-12-31", interval='1d', actions=False)
# price_history.head()
train_portion = 0.75 # 0.8 for 10 years, 0.75 for 8 years

# Moving Average Convergence Divergence (MACD)
k = price_history['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
d = price_history['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
macd = k - d
price_history['macd'] = price_history.index.map(macd)

# Relative Strength Index (RSI)
window_length = 14
diff = price_history['Close'].diff(1)
gain = diff.clip(lower=0).round(2)
loss = diff.clip(upper=0).abs().round(2)
price_history['avg_gain'] = gain.rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
price_history['avg_loss'] = loss.rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
for i, row in enumerate(price_history['avg_gain'].iloc[window_length+1:]):
    price_history['avg_gain'].iloc[i + window_length + 1] = (price_history['avg_gain'].iloc[i + window_length] * (window_length - 1) + gain.iloc[i + window_length + 1]) / window_length
# Average Losses
for i, row in enumerate(price_history['avg_loss'].iloc[window_length+1:]):
    price_history['avg_loss'].iloc[i + window_length + 1] = (price_history['avg_loss'].iloc[i + window_length] * (window_length - 1) + loss.iloc[i + window_length + 1]) / window_length
rs = price_history['avg_gain'] / price_history['avg_loss']
price_history['rsi'] = 100 - (100 / (1.0 + rs))
price_history = price_history.drop('avg_gain', axis=1)
price_history = price_history.drop('avg_loss', axis=1)

# Commodity Channel Index (CCI)
tp = (price_history['High'] + price_history['Low'] + price_history['Close']) / 3
price_history['sma'] = tp.rolling(20).mean()
price_history['adv'] = tp.rolling(20).apply(lambda x: pd.Series(x).mad())
price_history['cci'] = (tp - price_history['sma']) / (0.015 * price_history['adv'])
price_history = price_history.drop('sma', axis=1)
price_history = price_history.drop('adv', axis=1)

# Average Directional Index (ADX)
plus_dm = price_history['High'].diff()
minus_dm = price_history['Low'].diff()
plus_dm[plus_dm < 0] = 0
minus_dm[minus_dm > 0] = 0

tr1 = pd.DataFrame(price_history['High'] - price_history['Low'])
tr2 = pd.DataFrame(abs(price_history['High'] - price_history['Close'].shift(1)))
tr3 = pd.DataFrame(abs(price_history['Low'] - price_history['Close'].shift(1)))
frames = [tr1, tr2, tr3]
tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
atr = tr.rolling(14).mean()

plus_di = 100 * (plus_dm.ewm(alpha = 1/14).mean() / atr)
minus_di = abs(100 * (minus_dm.ewm(alpha = 1/14).mean() / atr))
dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
adx = ((dx.shift(1) * (14 - 1)) + dx) / 14
adx_smooth = adx.ewm(alpha = 1/14).mean()
price_history['adx'] = adx_smooth

price_history = price_history.dropna()

# # verbose
# price_history.head(30)

# time_series = list(price_history['Close'])
time_series = price_history
print('length', len(time_series))
dt_list = [pendulum.parse(str(dt)).float_timestamp for dt in list(price_history.index)]

# split into train, val, test
train_ind = int(len(time_series) * train_portion)
# val_ind = train_ind + int(len(time_series) * 0.25)

train_series = time_series[:train_ind]
# val_series = time_series[train_ind:val_ind]
test_series = time_series[train_ind:]


# # verbose
# time_int = [
#     (0, train_ind), 
# #     (train_ind, val_ind), 
#     (train_ind, len(time_series))
# ]
# titles = [
#     'Train Series', 
# #     'Validation Series', 
#     'Test Series'
# ]

# # plot
# i=0
# for series in [train_series, test_series]:
#     plt.style.use('dark_background')
#     dt_list_c = dt_list[time_int[i][0]:time_int[i][1]]
#     plt.plot(dt_list_c, series['Close'], linewidth=2)
#     plt.title(titles[i])
#     plt.xlabel('Time (1 day interval)')
#     plt.ylabel('Close Price ($)')
#     i += 1
#     plt.show()