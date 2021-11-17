def trim_ohlcv(grab, symbol):
  kldata = grab.klines[symbol]
  df = pd.DataFrame(data=kldata)
  
  dohlcv = df.iloc[:, [0, 1, 2, 3, 4, 5]]
  dohlcv[0] = pd.to_datetime(dohlcv[0], unit="ms")
  dohlcv.columns = ["date", "open", "high", "low", "close", "volume"]
  
  ohlcv = dohlcv.iloc[:, [1, 2, 3, 4, 5]]
  ohlcv.set_index(pd.DatetimeIndex(dohlcv["date"]), inplace=True)
  ohlcv = ohlcv.astype("float64")
  grab.ohlcvs[symbol] = ohlcv
  return ohlcv