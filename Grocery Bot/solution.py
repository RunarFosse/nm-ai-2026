import itertools

from collections import Counter
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Dict, Tuple

from heapq import heappop, heappush

from schemas import Bot

@dataclass(frozen=True)
class State():
    """ State space State """
    position: Tuple[int, int]
    inventory_count: int
    remaining_items: FrozenSet[Tuple[str, int]]

class Solution():
    def __init__(self, items: List[str], bot: Bot, drop_off_position: Tuple[int, int], item_ids: Dict[str, List[str]], item_positions: Dict[str, List[Tuple[int, int]]], distance_matrix: Dict[Tuple[int, int], Dict[Tuple[int, int], int]]):
        self.item_ids = item_ids
        self.item_positions = item_positions
        self.distance_matrix = distance_matrix
        self.drop_off_position = drop_off_position

        items = [item for item in items if item not in bot["inventory"]]

        # State is (position, items picked, remaining items)
        start_state = State(position=tuple(bot["position"]), inventory_count=len(bot["inventory"]), remaining_items=frozenset(Counter(items).items()))
        self.path, self.cost = self.solve(start_state)
    
    def __repr__(self) -> str:
        return f"(Path: {self.path}, Cost: {self.cost})"
    
    def solve(self, start_state: State) -> Tuple[List[Tuple[State, str]], int]:
        counter = itertools.count()

        queue, seen, came_from = [(0, next(counter), 0, start_state)], set(), {}
        g_scores = {start_state: 0}
        while queue:
            _, _, g_cost, state = heappop(queue)
            if state in seen:
                continue
            seen.add(state)

            if self.is_goal(state):
                path = self.reconstruct_path(state, came_from)
                return (path, g_cost)
        
            for new_state, transition_cost, description in self.get_transitions(state):
                new_g_cost = g_cost + transition_cost
                if new_state not in g_scores or new_g_cost < g_scores[new_state]:
                    g_scores[new_state] = new_g_cost
                    new_f_cost = self.get_heuristic(new_state) + new_g_cost
                    heappush(queue, (new_f_cost, next(counter), new_g_cost, new_state))
                    came_from[new_state] = (state, description)

    def get_heuristic(self, state: State) -> int:
        distance_to_drop_off = self.distance_matrix[state.position][self.drop_off_position]
        if len(state.remaining_items) == 0:
            return distance_to_drop_off

        if state.inventory_count == 3:
            return distance_to_drop_off + min(self.distance_matrix[self.drop_off_position][item_position] for (item, _) in state.remaining_items for item_id in self.item_ids[item] for item_position in self.item_positions[item_id])

        return min(self.distance_matrix[state.position][item_position] for (item, _) in state.remaining_items for item_id in self.item_ids[item] for item_position in self.item_positions[item_id])

    def get_transitions(self, state: State) -> Iterable[Tuple[State, int, str]]:
        if state.inventory_count < 3:
            for (item, count) in state.remaining_items:
                for item_id in self.item_ids[item]:
                    for item_position in self.item_positions[item_id]:
                        new_cost = self.distance_matrix[state.position][item_position] + 1

                        remaining_items = state.remaining_items.difference({(item, count)})
                        if count != 1:
                            remaining_items = remaining_items.union({(item, count - 1)})

                        new_state = State(position=item_position, inventory_count=state.inventory_count + 1, remaining_items=remaining_items)
                        yield (new_state, new_cost, item_id)

        if state.inventory_count > 0:
            new_cost = self.distance_matrix[state.position][self.drop_off_position] + 1
            new_state = State(position=self.drop_off_position, inventory_count=0, remaining_items=state.remaining_items)
            yield (new_state, new_cost, "drop_off")

    def is_goal(self, state: State) -> bool:
        return state.position == self.drop_off_position and state.inventory_count == 0 and len(state.remaining_items) == 0

    def reconstruct_path(self, state: State, came_from: Dict[State, Tuple[State, str]]) -> List[Tuple[State, str]]:
        path = []
        current_state = state
        while current_state in came_from:
            previous_state, description = came_from[current_state]
            path.append((current_state, description))
            current_state = previous_state
        path.append((current_state, "start"))
        path.reverse()
        return path










"""
    def compute_best_plan(self, items: List[str], bot_position: List[int], drop_off_position: List[int]) -> Tuple[List[Action], int]:
        best_plan, best_cost = [], 1e9
        for route in self.get_routes(items):
            print(route)
            continue
            plan, cost = self.compute_plan(route, bot_position, drop_off_position)
            if cost < best_cost:
                best_plan, best_cost = plan, cost
        
        return (best_plan, best_cost)

    def compute_plan(self, route: List[str], bot_position: List[int], drop_off_position: List[int]) -> Tuple[List[Action], int]:
        n = len(route)

        cost = 0
        current_plan = []
        current_position = bot_position
        for i in range(n):
            lookahead = route[i + 1] if i != n - 1 else None
            plan, additional_cost, new_position = self.plan_to_item(route[i], lookahead, current_position, drop_off_position)
            cost += additional_cost
            current_plan += plan
            current_position = new_position
        
        return (current_plan, cost)
    
    def plan_to_item(self, item_id: str, lookahead: Optional[str], current_position: List[int], drop_off_position: List[int]) -> Tuple[List[Action], int, List[int]]:
        item_position = self.item_positions[item_id]
        possible_targets = self.graph[item_position[1]][item_position[0]]



        pass

    def route_to_plan(self, route: List[str], bot_position: List[int], drop_off_position: List[int]) -> List[Action]:
        pass
    
    def get_routes(self, items: List[str]) -> Iterable[List[str]]:
        seen = set()
        for permutation in permutations(items):
            # Replace items with specific item ids
            routes = product(*map(lambda item: self.item_ids[item], permutation))
            for route in routes:
                # Only add unseen permutations
                if route in seen:
                    continue
                seen.add(route)

                # Add specific neighbour tiles to item ids
                neighbour_tiles_of = lambda item_id: (item_id, self.graph[self.item_positions[item_id][1]][self.item_positions[item_id][0]])
                specific_routes = product(*map(neighbour_tiles_of, route))
                for specific_route in specific_routes:
                    route_with_drop_offs = self.add_drop_offs(list(specific_routes))
                    yield from route_with_drop_offs
    
    def add_drop_offs(self, route: List[str], start_index: int = 0) -> List[str]:
        n, routes = len(route), []
        if start_index == n:
            routes.append([])

        for i in range(start_index, min(start_index + 3, n)):
            routes_further = self.add_drop_offs(route, start_index=i + 1)
            for j in range(len(routes_further)):
                routes_further[j] = route[start_index:i+1] + ["drop_off"] + routes_further[j]
            
            routes.extend(routes_further)

        return routes

if __name__ == "__main__":
    item_ids = {"milk": ["item_0"], "sugar": ["item_1", "item_2"], "water": ["item_3"], "paint": ["item_4"]}
    item_positions = {"item_0": [0, 0], "item_1": [0, 1], "item_2": [2, 1], "item_3": [2, 2], "item_4": [2, 3]}
    graph = [
        [[[1, 0]], [[1, 1]], [[]], [[]]], 
        [[[1, 1]], [[1, 0], [1, 2]], [[1, 1], [1, 3]], [[1, 2]]], 
        [[[1, 0]], [[2, 0], [1, 1]], [[1, 2]], [[1, 3]]]
    ]
    #     0 1 2 3
    #   + -------
    # 0 | 0 1 # #
    # 1 | . . . .
    # 2 | . 2 3 4
    Solution(items=["milk", "sugar", "milk", "sugar"], bot_position=None, drop_off_position=None, item_ids=item_ids, item_positions=item_positions, graph=graph)
"""