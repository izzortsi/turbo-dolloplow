
# %%

import asyncio
from kucoin.client import WsToken
from kucoin.ws_client import KucoinWsClient
from threading import Thread

async def openWs(symbol, dataObj):
    async def deal_msg(msg):
        if msg['topic'] == f'/market/ticker:{symbol}':
            data.append(msg['data'])
            # dataObj.rawdata["symbol"].append(msg["data"])

    # is public
    client = WsToken()
    #is private
    # client = WsToken(key='', secret='', passphrase='', is_sandbox=False, url='')
    # is sandbox
    # client = WsToken(is_sandbox=True)
    # append_msg = lambda msg: deal_msg(msg, symbol, dataObj)
    ws_client = await KucoinWsClient.create(None, client, dataObj.deal_msg, private=False)
    await ws_client.subscribe(f'/market/ticker:{symbol}')
    # await ws_client.subscribe('/spotMarket/level2Depth5:BTC-USDT,KCS-USDT')
    while running:
        await asyncio.sleep(10, loop=loop)


# %%



loop = asyncio.new_event_loop()
running = True


# %%

def evaluate(future):
    global running
    stop = future.result()
    if stop:
        print("press enter to exit...")
        running = False


def side_thread(loop, openWs, dataObj):
    asyncio.set_event_loop(loop)
    # loop.run_forever()
    loop.run_until_complete(openWs('BTC-USDT', dataObj))


# %%
class DataObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    async def deal_msg(self, msg):
        if msg['topic'] == f'/market/ticker:{self.symbol}':
            print(msg['data'])
            self.rawdata.append(msg['data'])

dataObj = DataObject(rawdata={f"{symbol}": [] for symbol in ['BTC-USDT']})
# %%

# async def display(text):
#     await asyncio.sleep(5)
#     print("echo:", text)
#     return text == "exit"

thread = Thread(target=side_thread, args=(loop, openWs, dataObj), daemon=True)
thread.start()

# %%
sym = 'BTC-USDT'
len(dataObj.rawdata[sym])
# while running:
#   text = input("enter text: ")
#   future = asyncio.run_coroutine_threadsafe(display("text"), loop)
#   future.add_done_callback(evaluate)


# print("exiting")    

# %%
#
# from unicorn_binance_rest_api.unicorn_binance_rest_api_manager import (
#     BinanceRestApiManager as Client,
# )
#
# from unicorn_binance_websocket_api.unicorn_binance_websocket_api_manager import (
#     BinanceWebSocketApiManager,
# )
#
# %%

import threading
import time


class ThreadedManager:
    def __init__(self, client, wsclient, symbols, tframe, rate=1):

        self.client = client
        self.wsclient = wsclient
        self.symbols = [s.upper() for s in symbols]
        self.rate = rate  # debbug purposes. will be removed
        self.tframe = tframe

        self.qty = {symbol: None for symbol in self.symbols}
        self.price_formatter = {symbol: None for symbol in self.symbols}
        self.traders = {}

        self.is_monitoring = False
