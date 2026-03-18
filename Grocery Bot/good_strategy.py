from itertools import pairwise, permutations
from heapq import heappop, heappush
from collections import defaultdict, deque
from typing import Deque, Dict, Set, Tuple
from schemas import *


from solution import Solution
 
class GoodStrategy():
    def __init__(self):
        self.initialized = False
        pass

    def initialize(self) -> None:
        self.initialized = True

        self.action_plan = {}

        self.item_ids = defaultdict(list)
        self.item_positions = defaultdict(list)
        self.distance_matrix = defaultdict(dict)
        self.drop_off_position = tuple(self.state["drop_off"])

        self.grid_width, self.grid_height = self.state["grid"]["width"], self.state["grid"]["height"]
        self.blocked = set()
        for wall in self.state["grid"]["walls"]:
            self.blocked.add(tuple(wall))

        for item in self.state["items"]:
            self.item_ids[item["type"]].append(item["id"])
            self.blocked.add(tuple(item["position"]))
        
        for item in self.state["items"]:
            (x, y) = tuple(item["position"])
            for a, b in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if x + a < 0 or x + a >= self.grid_width or y + b < 0 or y + b >= self.grid_height or (x + a, y + b) in self.blocked:
                    continue
                self.item_positions[item["id"]].append((x + a, y + b))

        self.adjls = defaultdict(list)
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if (x, y) in self.blocked:
                    continue
                
                for a, b in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if x + a < 0 or x + a >= self.grid_width or y + b < 0 or y + b >= self.grid_height or (x + a, y + b) in self.blocked:
                        continue
                    self.adjls[(x, y)].append((x + a, y + b))

        notable_positions = set([position for positions in self.item_positions.values() for position in positions] + [self.drop_off_position] + [tuple(bot["position"]) for bot in self.state["bots"]])
        for notable_position in notable_positions:
            queue, seen = deque([(notable_position, 0)]), set()
            while queue:
                position, distance = queue.popleft()
                if position in seen:
                    continue
                seen.add(position)

                if position in notable_positions:
                    self.distance_matrix[notable_position][position] = distance

                for neighbour in self.adjls[position]:
                    queue.append((neighbour, distance + 1))
 
    def step(self, state: State) -> Response:
        self.state = state

        if not self.initialized:
            self.initialize()

        #actions = self.manage(state["bots"])
        #return actions
        

        solutions = {}
        for bot in self.state["bots"]:
            if bot["id"] in self.action_plan:
                continue

            solution = self.act(bot)
            if solution is None:
                continue

            solutions[bot["id"]] = solution

        actions = self.solutions_to_actions(solutions)
        self.action_plan |= actions

        actions = []
        for bot in self.state["bots"]:
            if bot["id"] not in self.action_plan:
                action = {"bot": bot["id"], "action": "wait"}
            else:
                action = self.action_plan[bot["id"]].popleft()
                if not self.action_plan[bot["id"]]:
                    self.action_plan.pop(bot["id"])
            actions.append(action)

        return actions
    
    def manage(self, bots: List[Bot]) -> List[Action]:
        unassigned_active_items, unassigned_preview_items = [], []
        for order in self.state["orders"]:
            if order["status"] == "active":
                unassigned_active_items.extend(order["items_required"])
                for delivered in order["items_delivered"]:
                    unassigned_active_items.remove(delivered)
            else:
                unassigned_preview_items.extend(order["items_required"])
        
        for bot in bots:
            if bot["id"] not in self.action_plan:

                # Grab 3 active items
                pass
        # TODO!!!!!!
        return None


    def act(self, bot: Bot) -> Solution:
        # TODO: Temporary order grab, add manager object
        #print(self.state["orders"])
        items, preview_items = [], []
        for order in self.state["orders"]:
            if order["status"] == "active":
                for item in order["items_required"]:
                    if item in order["items_delivered"]:
                        order["items_delivered"].remove(item)
                    else:
                        items.append(item)
            else:
                for item in order["items_required"]:
                    if item in order["items_delivered"]:
                        order["items_delivered"].remove(item)
                    else:
                        preview_items.append(item)
        
        # TODO: Handle this in manager, but pad with lowest cost preview items
        if len(items) < 3:
            best_additions, best_cost = [], 1e9
            for preview_additions in permutations(preview_items, 3 - len(items)):
                preview_rest = [item for item in preview_items]
                for item in preview_additions:
                    preview_rest.remove(item)
                solution_cost = (
                    Solution(items + list(preview_additions), bot=bot, drop_off_position=self.drop_off_position, item_ids=self.item_ids, item_positions=self.item_positions, distance_matrix=self.distance_matrix).cost
                    - Solution(items, bot=bot, drop_off_position=self.drop_off_position, item_ids=self.item_ids, item_positions=self.item_positions, distance_matrix=self.distance_matrix).cost
                )
                solution_remaining_cost = (
                    Solution(preview_rest + list(preview_additions), bot=bot, drop_off_position=self.drop_off_position, item_ids=self.item_ids, item_positions=self.item_positions, distance_matrix=self.distance_matrix).cost
                    - Solution(preview_rest, bot=bot, drop_off_position=self.drop_off_position, item_ids=self.item_ids, item_positions=self.item_positions, distance_matrix=self.distance_matrix).cost
                    )
                if solution_cost < best_cost and solution_cost < solution_remaining_cost:
                    best_additions = preview_additions
                    best_cost = solution_cost

            items.extend(best_additions)

        # Compute the best possible solution
        #print(f"Current required: {items}", f"Current inventory: {bot['inventory']}")
        solution = Solution(items, bot=bot, drop_off_position=self.drop_off_position, item_ids=self.item_ids, item_positions=self.item_positions, distance_matrix=self.distance_matrix)

        return solution
    
    def solutions_to_actions(self, solutions: Dict[int, Solution]) -> Dict[int, Deque[Action]]:
        # Sort solutions based on cost (highest cost gets higher priority)
        solutions = sorted(solutions.items(), key=lambda solution: solution[1].cost, reverse=True)

        bot_actions = {}
        reservation_table = set()
        for (bot_id, solution) in solutions:
            actions = self.solution_to_actions(solution, bot_id=bot_id, reservation_table=reservation_table)
            bot_actions[bot_id] = actions

        return bot_actions

    def solution_to_actions(self, solution: Solution, bot_id: str, reservation_table: Set[Tuple[int, int, int]]) -> Deque[Action]:
        actions = []
        start_time = self.state["round"]
        dropped_off = False
        for i in range(len(solution.path) - 1):
            current_time = start_time + len(actions)
            current_state, _ = solution.path[i]
            target_state, description = solution.path[i + 1]
            path = self.pathfind(current_state.position, current_time, target_state.position, reservation_table=reservation_table)

            # Populate reservation table, ensuring Space-Time A*
            for j, (x, y, time) in enumerate(path):
                reservation_table.add((x, y, time))
                if time > 0:
                    previous_x, previous_y, _ = path[j - 1]
                    reservation_table.add((x, y, previous_x, previous_y, time))


            # TODO: Add chunking to manager, not here:
            if not dropped_off:
                move_actions = self.path_to_actions(path, bot_id=bot_id)
                actions.extend(move_actions)

                if description == "drop_off":
                    dropped_off = True
                    actions.append({"bot": bot_id, "action": "drop_off"})
                elif description != "":
                    actions.append({"bot": bot_id, "action": "pick_up", "item_id": description})
        #print(f"Current plan: {actions}")
        return deque(actions)
    
    def pathfind(self, start_position: Tuple[int, int], start_time: int, target_position: Tuple[int, int], reservation_table: Set[Tuple[int, int, int]] = {}) -> List[Tuple[int, int, int]]:

        def get_neighbours(x: int, y: int, time: int) -> List[Tuple[int, int, int]]:
            neighbours = []
            new_time = time + 1
            for new_x, new_y in self.adjls[(x, y)] + [(x, y)]:
                # Check is someone is standing at target position
                if (new_x, new_y, new_time) in reservation_table:
                    continue
            
                # Check if we are colliding with someone else moving opposite direction
                if (new_x, new_y, x, y, new_time) in reservation_table:
                    continue
                    
                neighbours.append((new_x, new_y, new_time))
            return neighbours
        
        def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
            return abs(x1 - x2) + abs(y1 - y2)

        queue, seen, came_from = [(0, 0, (*start_position, start_time))], set(), {}
        while queue:
            _, g_cost, (x, y, time) = heappop(queue)
            if (x, y, time) in seen:
                continue
            seen.add((x, y, time))

            if (x, y) == target_position:
                path = []
                current_position_time = (x, y, time)
                while current_position_time in came_from:
                    path.append(current_position_time)
                    current_position_time = came_from[current_position_time]
                path.append(current_position_time)
                path.reverse()
                return path

            for neighbour in get_neighbours(x, y, time):
                new_g_cost = g_cost + 1
                new_h_cost = manhattan_distance(x, y, neighbour[0], neighbour[1])
                new_f_cost = new_h_cost + new_g_cost
                came_from[neighbour] = (x, y, time)
                heappush(queue, (new_f_cost, new_g_cost, neighbour))
        
        raise Exception(f"No path from {start_position} to {target_position}!")

    def path_to_actions(self, path: List[Tuple[int, int, int]], bot_id: int) -> List[Action]:
        difference_to_action = {
            (0, -1): "move_up",
            (0, 1): "move_down",
            (-1, 0): "move_left",
            (1, 0): "move_right",
            (0, 0): "wait",
        }

        actions = []
        for (x1, y1, _), (x2, y2, _) in pairwise(path):
            dx, dy = x2 - x1, y2 - y1
            action = difference_to_action[(dx, dy)]
            actions.append({"bot": bot_id, "action": action})
        
        return actions


if __name__ == "__main__":
    state = {
        'type': 'game_state', 
        'round': 0, 
        'max_rounds': 300, 
        'grid': {
            'width': 12, 
            'height': 10, 
            'walls': [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [0, 1], [11, 1], [0, 2], [2, 2], [6, 2], [10, 2], [11, 2], [0, 3], [2, 3], [6, 3], [10, 3], [11, 3], [0, 4], [2, 4], [6, 4], [10, 4], [11, 4], [0, 5], [11, 5], [0, 6], [2, 6], [6, 6], [10, 6], [11, 6], [0, 7], [11, 7], [0, 8], [11, 8], [0, 9], [1, 9], [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [7, 9], [8, 9], [9, 9], [10, 9], [11, 9]]}, 
            'bots': [
                {'id': 0, 'position': [10, 8], 'inventory': []}
            ], 
            'items': [
                {'id': 'item_0', 'type': 'cheese', 'position': [3, 2]}, 
                {'id': 'item_1', 'type': 'butter', 'position': [5, 2]}, 
                {'id': 'item_2', 'type': 'yogurt', 'position': [3, 3]}, 
                {'id': 'item_3', 'type': 'milk', 'position': [5, 3]}, 
                {'id': 'item_4', 'type': 'cheese', 'position': [3, 4]}, 
                {'id': 'item_5', 'type': 'butter', 'position': [5, 4]}, 
                {'id': 'item_6', 'type': 'yogurt', 'position': [7, 2]}, 
                {'id': 'item_7', 'type': 'milk', 'position': [9, 2]}, 
                {'id': 'item_8', 'type': 'cheese', 'position': [7, 3]}, 
                {'id': 'item_9', 'type': 'butter', 'position': [9, 3]}, 
                {'id': 'item_10', 'type': 'yogurt', 'position': [7, 4]}, 
                {'id': 'item_11', 'type': 'milk', 'position': [9, 4]}, 
                {'id': 'item_12', 'type': 'butter', 'position': [3, 6]}, 
                {'id': 'item_13', 'type': 'yogurt', 'position': [5, 6]}, 
                {'id': 'item_14', 'type': 'milk', 'position': [7, 6]}, 
                {'id': 'item_15', 'type': 'butter', 'position': [9, 6]}
            ], 
            'orders': [
                {'id': 'order_0', 'items_required': ['cheese', 'milk', 'milk', 'yogurt'], 'items_delivered': [], 'complete': False, 'status': 'active'}, 
                {'id': 'order_1', 'items_required': ['yogurt', 'milk', 'yogurt'], 'items_delivered': [], 'complete': False, 'status': 'preview'}
            ], 
            'drop_off': [1, 8], 
            'score': 0, 
            'active_order_index': 0, 
            'total_orders': 50
        }

    strategy = GoodStrategy()
    strategy.step(state=state)

