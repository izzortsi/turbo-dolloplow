
# %%

import asyncio
from kucoin.client import WsToken
from kucoin.ws_client import KucoinWsClient
from threading import Thread
from time import sleep

# %%


class ThreadedWS(Thread):
    def __init__(self, parent=None):
        self.parent = parent
        self.data = []
        self.loop = asyncio.new_event_loop()
        self.running = True
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

            if msg['topic'] == '/market/ticker:BTC-USDT':    
                print(msg["data"])
                self.data.append(msg["data"])
            elif msg['topic'] == '/market/ticker:ETH-USDT':
                print(msg["data"])
        
        # is public
        client = WsToken()
        #is private
        # client = WsToken(key='', secret='', passphrase='', is_sandbox=False, url='')
        # is sandbox
        # client = WsToken(is_sandbox=True)

        ws_client = await KucoinWsClient.create(None, client, deal_msg, private=False)
        await ws_client.subscribe('/market/ticker:BTC-USDT,ETH-USDT')
        # await ws_client.subscribe('/spotMarket/level2Depth5:BTC-USDT,KCS-USDT')
        while True:
            await asyncio.sleep(60, loop=self.loop)
            if not self.running:
                break
#%%
tws = ThreadedWS()

# %%
sleep(10)
#%%
tws.stop()

#%%
tws.data
#%%
