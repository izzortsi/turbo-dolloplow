from binance.client import Client
import pandas as pd
import numpy as np
from grabber import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt


# %%
client=Client()
grab = GrabberMACD(client)
grab.get_data()
ohlcv = grab.ohlcv.copy()
macd = grab.compute_indicators()
# %%
len(ohlcv)

# %%
macd_hist = np.array(macd.histogram)
hist = macd_hist[~np.isnan(macd_hist)]
idx_dif = abs(len(hist) - len(ohlcv))
# %%
diffs = np.array(ohlcv.close - ohlcv.open)[idx_dif:]
closes = np.array(ohlcv.close)[idx_dif:]
vols = np.array(ohlcv.volume)[idx_dif:]
labels = np.array([1 if diff > 0 else 0 for diff in diffs])

# %%
lens = len(diffs)
diffs = diffs.reshape((lens, 1))
closes = closes.reshape((lens, 1))
vols = vols.reshape((lens, 1))
hist = hist.reshape((lens, 1))
# %%
Xt = np.hstack((diffs, closes, vols, hist))
# %%

# %%

x_input = Xt[:-1]
y_target = labels[1:]
# %%
X_train, X_test, y_train, y_test = train_test_split(x_input, y_target, random_state=42)

# x_train = x_input[:300]
# y_train = y_target[:300]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000, 2), random_state=1)
# %%
pipe = make_pipeline(StandardScaler(), clf)
# %%


# %%


pipe.fit(X_train, y_train)

# %%
pipe.predict(X_test)

# %%
pipe.score(X_train, y_train)

# %%
x_test = x_input[301:]
y_test = y_target[301:]
# %%
pipe.score(x_test, y_test)
# %%

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(y=x_input[:, 0], mode='lines'))
fig.add_trace(go.Scatter(y=y_target*200, mode='lines'))
