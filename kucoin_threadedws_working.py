
# %%

import asyncio
from kucoin.client import WsToken
from kucoin.ws_client import KucoinWsClient

# %%


async def main():

    async def deal_msg(msg):
        # if msg['topic'] == '/spotMarket/level2Depth5:BTC-USDT':
        
        # global data

        if msg['topic'] == '/market/ticker:BTC-USDT':    
            print(msg["data"])
            data.append(msg["data"])
        elif msg['topic'] == '/market/ticker:ETH-USDT':
            print(msg["data"])
            # print(f'Get KCS level3:{msg["data"]}')

    # is public
    client = WsToken()
    #is private
    # client = WsToken(key='', secret='', passphrase='', is_sandbox=False, url='')
    # is sandbox
    # client = WsToken(is_sandbox=True)
    data = []
    ws_client = await KucoinWsClient.create(None, client, deal_msg, private=False)
    await ws_client.subscribe('/market/ticker:BTC-USDT,ETH-USDT')
    # await ws_client.subscribe('/spotMarket/level2Depth5:BTC-USDT,KCS-USDT')
    while True:
        await asyncio.sleep(60, loop=loop)

# if __name__ == "__main__":
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(main())

# %%
from threading import Thread

loop = asyncio.new_event_loop()
running = True
# data = []
def side_thread(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    # loop.run_forever()

thread = Thread(target=side_thread, args=(loop,), daemon=True)
thread.start()


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
        self.loop.run_forever()
#%%


#%%
