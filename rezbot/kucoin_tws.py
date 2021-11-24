
# %%

import asyncio
from kucoin.client import WsToken
from kucoin.ws_client import KucoinWsClient
from threading import Thread
from time import sleep, time

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
                    data = self.callbacks[subscription](msg["data"])
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
SYM = 'BTC-USDT'
TYPE = '1min'
tws = ThreadedWS()

# %%
# sleep(10)
# #%%
# tws.stop()

# #%%
# len(tws.data), tws.data
#%%
