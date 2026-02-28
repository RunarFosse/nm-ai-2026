from typing import Dict, List, Tuple
from schemas import *

from heapq import *

def manhattan_heuristic(position: List[int], target: List[int]) -> int:
    return abs(position[0] - target[0]) + abs(position[1] - target[1])

def a_start_pathfinding(graph: List[List[List[int]]], start: List[int], target: List[int]) -> Tuple[List[List[int]], int]:
    queue = [(0, start)]
    distances, came_from = {tuple(start): 0}, {}
    while queue:
        _, position = heappop(queue)

        if position == target:
            # Reconstruct the path
            path, distance = [], distances[tuple(position)]
            while tuple(position) in came_from:
                path.append(position)
                position = came_from[tuple(position)]
            path.reverse()
            return path, distance
        
        for neighbour in graph[position[1]][position[0]]:
            new_distance = distances[tuple(position)] + 1
            if tuple(neighbour) not in distances or new_distance < distances[tuple(neighbour)]:
                came_from[tuple(neighbour)] = position
                distances[tuple(neighbour)] = new_distance

                distance_estimate = new_distance + manhattan_heuristic(position, neighbour)
                heappush(queue, (distance_estimate, neighbour))

    raise Exception(f"No path from {start} to {target}!")

def path_to_actions(path: List[List[int]], start: List[int]) -> List[Dict[str, str]]:
    actions = []
    position = start

    directions = {
        (0, -1): "move_up",
        (0, 1): "move_down",
        (-1, 0): "move_left",
        (1, 0): "move_right",
    }

    for next_position in path:
        dx, dy = next_position[0] - position[0], next_position[1] - position[1]
        actions.append({"action": directions[(dx, dy)]})
        position = next_position

    return actions

def navigate_to_item(graph: List[List[List[int]]], item_positions: Dict[str, Tuple[str, List[int]]], start: List[int], item_type: str) -> Tuple[List[str | Tuple[str, str]], int]:
    items = item_positions[item_type]
    best = ([], 1e9, None)
    for item_id, position in items:
        current = ([], 1e9)
        for isle in graph[position[1]][position[0]]:
            path, distance = a_start_pathfinding(graph, start, isle)
            if distance < current[1]:
                current = (path, distance)

        if current[1] < best[1]:
            best = (current[0], current[1], item_id)

    if best[2] is None:
        return [], 1e9

    plan = path_to_actions(best[0], start) + [{"action": "pick_up", "item_id": best[2]}]
    return plan, best[1]

def navigate_to_position(graph: List[List[List[int]]], start: List[int], target: List[int]) -> Tuple[List[str], int]:
    path, distance = a_start_pathfinding(graph, start, target)
    return path_to_actions(path, start), distance