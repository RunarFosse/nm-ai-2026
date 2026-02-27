import asyncio
import json
import websockets

from example_strategy import ExampleStrategy

WS_URL = open(".ws_url").readline()

strategy = ExampleStrategy()
 
async def play():
    async with websockets.connect(WS_URL) as ws:
        while True:
            state = json.loads(await ws.recv())
 
            if state["type"] == "game_over":
                print(f"Game over! Score: {state['score']}")
                break
 
            actions = strategy.predict(state)
            await ws.send(json.dumps({"actions": actions}))

if __name__ == "__main__":
    asyncio.run(play())