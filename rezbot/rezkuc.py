
# %%

import asyncio
from kucoin.client import WsToken
from kucoin.ws_client import KucoinWsClient
from threading import Thread
from time import sleep, time
import pandas as pd
import numpy as np
# %%


class ThreadedWS(Thread):
    def __init__(self, parent=None, public=True):
        self.parent = parent
        self.data = []
        self.loop = asyncio.new_event_loop()
        self.running = True
        self.subscriptions = []
        self.callbacks = {}
        self.public = public
        # self.ws_client = None
        
        super().__init__(group=None, daemon=True)

        self.start()

    def run(self):

        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.main())

    def stop(self):
        self.running = False
        self.loop.stop()

    async def main(self):

        async def deal_msg(msg):
            #'/market/ticker:ETH-USDT'
            for subscription in self.subscriptions:
                if msg['topic'] == subscription:    
                    if time() // 30 == 0:
                        print(msg["data"])
                    data = self.callbacks[subscription](msg)
                    self.data.append(data)
                    # print(data)
            
        # is public
        # if self.public:
        client = WsToken()
        #is private
        # client = WsToken(key='', secret='', passphrase='', is_sandbox=False, url='')
        # is sandbox
        # client = WsToken(is_sandbox=True)

        self.ws_client = await KucoinWsClient.create(None, client, deal_msg, private=False)
        # await self.ws_client.subscribe('/market/ticker:BTC-USDT,ETH-USDT')
        # self.subscriptions = ['/market/candles:{symbol}_{type}', '/market/ticker:ETH-USDT']
        # await ws_client.subscribe('/spotMarket/level2Depth5:BTC-USDT,KCS-USDT')
        while True:
            await asyncio.sleep(60, loop=self.loop)
            if not self.running:
                break

    async def subscribe_klines(self, symbol, type, callback = lambda x: x):
        topic = f'/market/candles:{symbol}_{type}'
        await self.ws_client.subscribe(topic)
        self.subscriptions.append(topic)
        self.callbacks[topic] = callback
    
    async def subscribe_snapshot(self, symbol, callback = lambda x: x):
        topic = f'/market/snapshot:{symbol}'
        await self.ws_client.subscribe(topic)
        self.subscriptions.append(topic)
        self.callbacks[topic] = callback

    async def subscribe_ticker(self, symbol, callback = lambda x: x):
        topic = f'/market/ticker:{symbol}'
        await self.ws_client.subscribe(topic)
        self.subscriptions.append(topic)
        self.callbacks[topic] = callback        

    async def process_data(self):
        pass
#%%


# %%
# sleep(10)
# #%%
# tws.stop()

# #%%
# len(tws.data), tws.data

# msg=json.dump({
#         "type":"message",
#     "topic":"/market/candles:BTC-USDT_1hour",
#     "subject":"trade.candles.update",
#     "data":{

#         "symbol":"BTC-USDT",    // symbol
#         "candles":[

#             "1589968800",   // Start time of the candle cycle
#             "9786.9",       // open price
#             "9740.8",       // close price
#             "9806.1",       // high price
#             "9732",         // low price
#             "27.45649579",  // Transaction volume
#             "268280.09830877"   // Transaction amount
#         ],
#         "time":1589970010253893337  // now（us）
#     }
# }

# %%
def process_klines(msg):
    klines = msg["data"]["candles"]
    print(klines)
    klines = np.atleast_2d(np.array(klines))
    df = pd.DataFrame(klines, columns=["timestamp", "open", "close", "high", "low", "transactionVol","transactionAmt"])
    return df


SYM = 'BTC-USDT'
TYPE = '1min'
tws = ThreadedWS()

# %%

await tws.subscribe_klines("ETH-USDT", "1min", callback=process_klines)

# %%

tws.data[-1]
# %%
tws.running = False
# %%

# %%
tws.data[-1]
# %%
class RingBuffer(pd.DataFrame):
    def __init__(self, window_length, granularity):
        self.window_length = window_length
        self.granularity = granularity
        



# %%
