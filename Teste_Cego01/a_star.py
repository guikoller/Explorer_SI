from math import sqrt
from dijkstar import Graph, find_path
from dijkstar.algorithm import (
    single_source_shortest_paths as calc_shortest_paths,
    extract_shortest_path_from_predecessor_list as extract_shortest_path,
    find_path
)
from vs.constants import VS
from map import Map
from typing import List, Tuple, Dict, Any

class AStar:
    def __init__(self, base: Tuple[int, int] = (0, 0), map: Map = None):
        self.normal_cost = 1.0
        self.diagonal_cost = 1.5
        self._graph = Graph()
        self.base = base
        self._graph.add_node(base)
        self._directions: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, int]] = {}
        self.map = map
        if self.map:
            self.map_to_graph(self.map)

    def map_to_graph(self, map: Map) -> None:
        for coord in map.data:
            self._graph.add_node(coord)
            diff, vic_id, actions_res = map.get(coord)
            x, y = coord
            if actions_res[0] == VS.CLEAR:  # up
                up = (x, y - 1)
                if map.in_map(up):
                    neighbor_diff = map.get_difficulty(up)
                    self.add_edge((x, y), up, self.normal_cost * neighbor_diff, self.normal_cost * diff)
            if actions_res[1] == VS.CLEAR:  # up right
                up_right = (x + 1, y - 1)
                if map.in_map(up_right):
                    neighbor_diff = map.get_difficulty(up_right)
                    self.add_edge((x, y), up_right, self.diagonal_cost * neighbor_diff, self.diagonal_cost * diff)
            if actions_res[2] == VS.CLEAR:  # right
                right = (x + 1, y)
                if map.in_map(right):
                    neighbor_diff = map.get_difficulty(right)
                    self.add_edge((x, y), right, self.normal_cost * neighbor_diff, self.normal_cost * diff)
            if actions_res[3] == VS.CLEAR:  # down right
                down_right = (x + 1, y + 1)
                if map.in_map(down_right):
                    neighbor_diff = map.get_difficulty(down_right)
                    self.add_edge((x, y), down_right, self.diagonal_cost * neighbor_diff, self.diagonal_cost * diff)
            if actions_res[4] == VS.CLEAR:  # down
                down = (x, y + 1)
                if map.in_map(down):
                    neighbor_diff = map.get_difficulty(down)
                    self.add_edge((x, y), down, self.normal_cost * neighbor_diff, self.normal_cost * diff)
            if actions_res[5] == VS.CLEAR:  # down left
                down_left = (x - 1, y + 1)
                if map.in_map(down_left):
                    neighbor_diff = map.get_difficulty(down_left)
                    self.add_edge((x, y), down_left, self.diagonal_cost * neighbor_diff, self.diagonal_cost * diff)
            if actions_res[6] == VS.CLEAR:  # left
                left = (x - 1, y)
                if map.in_map(left):
                    neighbor_diff = map.get_difficulty(left)
                    self.add_edge((x, y), left, self.normal_cost * neighbor_diff, self.normal_cost * diff)
            if actions_res[7] == VS.CLEAR:  # up left
                up_left = (x - 1, y - 1)
                if map.in_map(up_left):
                    neighbor_diff = map.get_difficulty(up_left)
                    self.add_edge((x, y), up_left, self.diagonal_cost * neighbor_diff, self.diagonal_cost * diff)

    def add_edge(self, node1: Tuple[int, int], node2: Tuple[int, int], cost1_2: float, cost2_1: float) -> None:
        self._graph.add_edge(node1, node2, cost1_2)
        self._graph.add_edge(node2, node1, cost2_1)
        self._directions[(node1, node2)] = self._get_direction(node1, node2)
        self._directions[(node2, node1)] = self._get_direction(node2, node1)

    def check_edge(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> bool:
        try:
            if self._graph.get_edge(node1, node2):
                return True
            return False
        except:
            return False

    def _get_direction(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> Tuple[int, int]:
        x1, y1 = node1
        x2, y2 = node2
        dx = x2 - x1
        dy = y2 - y1
        return (dx, dy)

    def _estimate_heuristics(self, node1: Tuple[int, int], node2: Tuple[int, int], edge: float, prev_edge: float) -> float:
        x1, y1 = node1
        x2, y2 = node2
        heuristics = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if not prev_edge and edge < heuristics:
            return edge
        if prev_edge and edge + prev_edge < heuristics:
            return edge + prev_edge
        return heuristics

    def calc_shortest_path(self, node1: Tuple[int, int], node2: Tuple[int, int], tlim: float = float('inf')) -> Tuple[List[Tuple[int, int]], float]:
        path = find_path(self._graph, node1, node2, heuristic_func=self._estimate_heuristics)
        if path[3] > tlim:
            return [], -1
        return path[0], path[3]

    def get_shortest_cost(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
        return self.calc_shortest_path(node1, node2)[1]

    def get_shortest_path(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> List[Tuple[int, int]]:
        return self.calc_shortest_path(node1, node2)[0]

    def calc_plan(self, node1: Tuple[int, int], node2: Tuple[int, int], tlim: float = float('inf')) -> Tuple[List[Tuple[int, int]], float]:
        path, cost = self.calc_shortest_path(node1, node2, tlim)
        if not path:
            return [], -1
        plan = []
        start = path.pop(0)
        for node in path:
            plan.append(self._directions[(start, node)])
            start = node
        return plan, cost

    def calc_backtrack(self, node: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        backtrack = []
        nodes, cost = self.calc_shortest_path(node, self.base)
        last = nodes.pop()
        while nodes:
            tmp = nodes.pop()
            backtrack.insert(0, self._directions[(tmp, last)])
            last = tmp
        return backtrack, cost
