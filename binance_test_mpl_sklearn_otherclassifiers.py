from binance.client import Client
import pandas as pd
import numpy as np
from grabber import *
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

# SVC(gamma=2, C=1),
# GaussianProcessClassifier(1.0 * RBF(1.0)),

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
svc = SVC(gamma=2, C=1)
gpc = GaussianProcessClassifier(1.0 * RBF(1.0))
# %%
pipe1 = make_pipeline(StandardScaler(), svc)

# %%
pipe2 = make_pipeline(StandardScaler(), gpc)

# %%

pipe1.fit(X_train, y_train)
pipe1.predict(X_test)
pipe1.score(X_test, y_test)
pipe1.score(X_train, y_train)


# %%
pipe2.fit(X_train, y_train)
pipe2.predict(X_test)
pipe2.score(X_test, y_test)
pipe2.score(X_train, y_train)
# %%


import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(y=x_input[:, 0], mode='lines'))
fig.add_trace(go.Scatter(y=y_target*200, mode='lines'))
