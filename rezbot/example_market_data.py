
# %%

#  MarketData
from kucoin.client import Market
import pandas as pd
import pandas_ta as ta
# client = Market(url='https://api.kucoin.com')
client = Market()

# or connect to Sandbox
# client = Market(url='https://openapi-sandbox.kucoin.com')
# client = Market(is_sandbox=True)

# get symbol kline
klines = client.get_kline('BTC-USDT','1min')

# %%
klines
# %%
dfkl = pd.DataFrame(klines, columns=["timestamp", "open", "close", "high", "low", "transactionAmt", "transactionVol"])
# %%
dfkl
# %%

# %%

# get symbol ticker
server_time = client.get_server_timestamp()

api_key = '<api_key>'
api_secret = '<api_secret>'
api_passphrase = '<api_passphrase>'

# Trade
from kucoin.client import Trade
client = Trade(key='', secret='', passphrase='', is_sandbox=False, url='')

# or connect to Sandbox
# client = Trade(api_key, api_secret, api_passphrase, is_sandbox=True)

# place a limit buy order
order_id = client.create_limit_order('BTC-USDT', 'buy', '1', '8000')

# place a market buy order   Use cautiously
order_id = client.create_market_order('BTC-USDT', 'buy', size='1')

# cancel limit order
client.cancel_order('5bd6e9286d99522a52e458de')

# User
from kucoin.client import User
client = User(api_key, api_secret, api_passphrase)

# or connect to Sandbox
# client = User(api_key, api_secret, api_passphrase, is_sandbox=True)

address = client.get_withdrawal_quota('KCS')
