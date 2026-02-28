from typing import List
from schemas import *

from utils import constructGraph, mapOutItems, computeDistances
from paths import a_start_pathfinding, navigate_to_item, navigate_to_position, manhattan_heuristic
 
class EasyStrategy():
    def __init__(self):
        self.graph = None
        self.distance_to_drop_off = None
        self.item_positions = None
 
    def step(self, state: State) -> Response:
        self.state = state

        # Pre-computation
        if self.item_positions is None:
            self.item_positions = mapOutItems(self.state["items"])
        if self.graph is None:
            self.graph = constructGraph(self.state["grid"], self.item_positions)
        if self.distance_to_drop_off is None:
            self.distance_to_drop_off = computeDistances(self.graph, state["drop_off"])

        actions = []
        for bot in self.state["bots"]:
            action = self.act(bot)
            actions.append({"bot": bot["id"]} | action)
        print(actions)
        return actions

    def act(self, bot: Bot) -> Action:
        # TODO: Temporary order grab
        order = next((order for order in self.state["orders"] if order["status"] == "active"), None)
        if not order:
            return {"action": "wait"}

        # Find missing items
        missing = order["items_required"]
        for item in bot["inventory"] + order["items_delivered"]:
            if item in missing:
                missing.remove(item)

        #print(bot["inventory"])
        #print(order)

        missing.sort(key=lambda item_type: navigate_to_item(self.graph, self.item_positions, bot["position"], item_type)[1])

        if missing and len(bot["inventory"]) < 3:
            # Navigate to the closest, first item, not grabbed
            plan, distance, end = navigate_to_item(self.graph, self.item_positions, bot["position"], missing[0])

            if distance + 1 + self.distance_to_drop_off[end[1]][end[0]] > 300 - self.state["round"]:
                plan, distance = navigate_to_position(self.graph, bot["position"], self.state["drop_off"])
                if distance == 0:
                    return {"action": "drop_off"}

        else:  
            plan, distance = navigate_to_position(self.graph, bot["position"], self.state["drop_off"])
            if distance == 0:
                return {"action": "drop_off"}

        return plan[0]