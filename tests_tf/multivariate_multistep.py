# multivariate multi-step stacked lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
import numpy as np
from grabber import *

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# %%
from binance.client import Client

n_steps_in, n_steps_out = 10, 1

# %%
client=Client()
#klines = client.get_klines(symbol='BNBBTC', interval=Client.KLINE_INTERVAL_30MINUTE)
grab = Grabber(client)
# %%
grab.get_data()

ohlcv = grab.ohlcv
# %%
diffs = ohlcv.close - ohlcv.open
# %%
diffs = np.array(diffs)
vols = np.array(ohlcv.volume)
# define input sequence
in_seq1 = diffs[:-n_steps_in]
in_seq2 = vols[:-n_steps_in]
out_seq = diffs[n_steps_in:]

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
in_seq1

# %%

dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
dataset
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
dataset.shape

# %%

dataset[10:10+n_steps_in+1, 0:2].shape
# %%

x_input = dataset[50:50+n_steps_in, 0:2].reshape(1, n_steps_in, n_features)
x_input
# x_input = array([[70, 75], [80, 85], [90, 95]])
# x_input = x_input.reshape((1, n_steps_in, n_features))


# %%

yhat = model.predict(x_input, verbose=0)
print(yhat)
dataset[50:50+n_steps_in+1, :].shape
dataset[50:50+n_steps_in+1, :]
