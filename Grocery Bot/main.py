import asyncio
import json
import websockets
from tqdm import tqdm

from good_strategy import GoodStrategy

WS_URL = open(".ws_url").readline()
ROUNDS = 300

strategy = GoodStrategy()
 
async def play():
    async with websockets.connect(WS_URL) as ws:
        #progress = tqdm(total=ROUNDS)
        while True:
            state = json.loads(await ws.recv())
 
            if state["type"] == "game_over":
                #progress.close()
                print(f"Game over! Score: {state['score']}")
                break
 
            actions = strategy.step(state)
            print(actions)
            await ws.send(json.dumps({"actions": actions}))
            #progress.update()

if __name__ == "__main__":
    asyncio.run(play())