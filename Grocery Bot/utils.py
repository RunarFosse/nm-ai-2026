from typing import List, Dict, Tuple
from schemas import *
from collections import defaultdict, deque

def constructGraph(grid: Grid, item_positions: Dict[str, List[int]]) -> List[List[List[int]]]:
    height, width = grid["height"], grid["width"]

    # Store walls as a tuple in a set
    walls = set(tuple(wall) for wall in grid["walls"])

    # Also ensure you can't walk onto item shelves
    items = set(tuple(position) for item in item_positions.values() for _, position in item)

    # Construct the graph
    graph = []
    for y in range(height):
        row = []
        for x in range(width):
            # If in a wall, can't move anywhere
            if (x, y) in walls:
                row.append([])
                continue
            
            # Otherwise, add position of all non-wall neighbours
            neighbours = []
            for a, b in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (x + a, y + b) in walls | items:
                    continue
                neighbours.append([x + a, y + b])
            row.append(neighbours)
        graph.append(row)

    return graph

def mapOutItems(items: List[Item]) -> Dict[str, List[Tuple[str, List[int]]]]:
    item_positions = defaultdict(list)
    for item in items:
        item_positions[item["type"]].append((item["id"], item["position"]))
    return item_positions

def computeDistances(graph: List[List[List[int]]], position: List[int]) -> List[List[int]]:
    m, n = len(graph), len(graph[0])
    distances = [[0] * n for _ in range(m)]

    queue, seen = deque([(position, 0)]), set()
    while queue:
        (x, y), distance = queue.popleft()
        if (x, y) in seen:
            continue
        seen.add((x, y))

        distances[y][x] = distance
        for neighbour in graph[y][x]:
            queue.append((neighbour, distance + 1))
    return distances