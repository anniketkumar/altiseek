"""Finite lattice graph with exact movement, snapping and energy costs."""

import heapq
from dataclasses import dataclass

import numpy as np

from altiseek.env.energy import EnergyModel
from altiseek.geometry.levels import LevelTable, altitude_for_footprint_m

ACTION_NAMES = ("N", "E", "S", "W", "ASCEND", "DESCEND")


@dataclass(frozen=True)
class Node:
    level: int
    ix: int
    iy: int


class Navigation:
    def __init__(self, table: LevelTable, hfov_deg: float, energy: EnergyModel) -> None:
        self.table, self.energy = table, energy
        self.nodes = tuple(
            Node(li, ix, iy)
            for li, level in enumerate(table.levels)
            for ix, iy in table.lattice(level)
        )
        self.index = {node: i for i, node in enumerate(self.nodes)}
        self.positions = np.array(
            [table.centre_m(table.levels[n.level], (n.ix, n.iy)) for n in self.nodes]
        )
        self.altitudes = np.array(
            [altitude_for_footprint_m(table.footprint_m(level), hfov_deg) for level in table.levels]
        )
        self.next = np.full((len(self.nodes), 6), -1, dtype=np.int32)
        self.time = np.zeros_like(self.next, dtype=float)
        self.cost = np.zeros_like(self.next, dtype=float)
        self.distance_m = np.zeros_like(self.next, dtype=float)
        for i, node in enumerate(self.nodes):
            stride = table.levels[node.level].stride_cells
            for action, (dx, dy) in enumerate(
                ((0, stride), (stride, 0), (0, -stride), (-stride, 0))
            ):
                target = self.index.get(Node(node.level, node.ix + dx, node.iy + dy), -1)
                self.next[i, action] = target
            for action, li in ((4, node.level + 1), (5, node.level - 1)):
                if 0 <= li < len(table.levels):
                    self.next[i, action] = self.nearest(li, self.positions[i])
            for action, target in enumerate(self.next[i]):
                if target < 0:
                    continue
                delta = self.positions[target] - self.positions[i]
                dz = self.altitudes[self.nodes[target].level] - self.altitudes[node.level]
                time, cost = energy.move(float(delta[0]), float(delta[1]), float(dz))
                self.time[i, action], self.cost[i, action] = (
                    time + energy.dwell_s,
                    cost + energy.dwell_s,
                )
                self.distance_m[i, action] = np.linalg.norm(delta)
        self.distances, self.first_actions = self._shortest_paths()

    def nearest(self, level: int, position_m: np.ndarray) -> int:
        candidates = [i for i, n in enumerate(self.nodes) if n.level == level]
        # Stable lattice order breaks equal-distance ties toward the south/west.
        return min(candidates, key=lambda i: float(np.sum((self.positions[i] - position_m) ** 2)))

    def bounds(self, index: int) -> tuple[int, int, int, int]:
        n = self.nodes[index]
        size = self.table.levels[n.level].footprint_cells
        return n.ix, n.iy, n.ix + size, n.iy + size

    def _shortest_paths(self) -> tuple[np.ndarray, np.ndarray]:
        count = len(self.nodes)
        distances = np.full((count, count), np.inf)
        first = np.full((count, count), -1, dtype=np.int32)
        for source in range(count):
            distances[source, source] = 0
            queue = [(0.0, source)]
            while queue:
                cost, node = heapq.heappop(queue)
                if cost > distances[source, node]:
                    continue
                for action, target in enumerate(self.next[node]):
                    if target < 0:
                        continue
                    candidate = cost + self.cost[node, action]
                    if candidate < distances[source, target] - 1e-10:
                        distances[source, target] = candidate
                        first[source, target] = action if node == source else first[source, node]
                        heapq.heappush(queue, (candidate, int(target)))
        return distances, first
