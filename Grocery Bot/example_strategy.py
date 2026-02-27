from typing import List
from schemas import *
 
class ExampleStrategy():
    def __init__(self):
        pass
 
    def predict(self, state: State) -> Response:
        self.state = state

        actions = []
        for bot in self.state["bots"]:
            action = self.decide(bot)
            actions.append(action)

        return actions

    def decide(self, bot: Bot) -> Action:
        x, y = bot["position"]
        drop_off = self.state["drop_off"]
 
        if bot["inventory"] and [x, y] == drop_off:
            return {"bot": bot["id"], "action": "drop_off"}
    
        if len(bot["inventory"]) >= 3:
            return self.move_toward(bot["id"], [x, y], drop_off)
    
        active = next((o for o in self.state["orders"] if o["status"] == "active"), None)
        if not active:
            return {"bot": bot["id"], "action": "wait"}
    
        needed = list(active["items_required"])
        for d in active["items_delivered"]:
            if d in needed:
                needed.remove(d)
    
        for item in self.state["items"]:
            if item["type"] in needed:
                ix, iy = item["position"]
                if abs(ix - x) + abs(iy - y) == 1:
                    return {"bot": bot["id"], "action": "pick_up", "item_id": item["id"]}
    
        for item in self.state["items"]:
            if item["type"] in needed:
                return self.move_toward(bot["id"], [x, y], item["position"])
    
        if bot["inventory"]:
            return self.move_toward(bot["id"], [x, y], drop_off)
    
        return {"bot": bot["id"], "action": "wait"}
    
    def move_toward(self, bot_id: int, position: List[int], target: List[int]) -> Action:
        (x, y), (tx, ty) = position, target
        if abs(tx - x) > abs(ty - y):
            return {"bot": bot_id, "action": "move_right" if tx > x else "move_left"}
        elif ty != y:
            return {"bot": bot_id, "action": "move_down" if ty > y else "move_up"}
        return {"bot": bot_id, "action": "wait"}
