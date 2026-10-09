import math
import heapq

class Node:
    """Node representation for A* Pathfinding."""
    def __init__(self, position, g_cost=float('inf'), h_cost=0, parent=None):
        self.position = position  # (row, col)
        self.g_cost = g_cost      # Cost from start to current node
        self.h_cost = h_cost      # Estimated cost from current node to goal
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

class AStarPlanner:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape

    def heuristic(self, a, b):
        """Euclidean distance heuristic."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def get_neighbors(self, node_pos):
        """Returns valid 8-directional neighbors (orthogonal & diagonal)."""
        r, c = node_pos
        directions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),  # Orthogonal
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414) # Diagonal
        ]
        
        neighbors = []
        for dr, dc, cost in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr, nc] == 0:  # Must be traversable free space
                    neighbors.append(((nr, nc), cost))
        return neighbors

    def plan(self, start, goal):
        """
        Computes shortest path from start (r, c) to goal (r, c) using A*.
        Returns (path, explored_nodes).
        """
        if self.grid[start[0], start[1]] == 1 or self.grid[goal[0], goal[1]] == 1:
            print("[ERROR] Start or Goal point is situated inside an obstacle!")
            return None, []

        open_set = []
        start_node = Node(start, g_cost=0, h_cost=self.heuristic(start, goal))
        heapq.heappush(open_set, start_node)

        g_scores = {start: 0}
        node_map = {start: start_node}
        closed_set = set()
        explored_nodes = []

        while open_set:
            current = heapq.heappop(open_set)

            if current.position in closed_set:
                continue

            closed_set.add(current.position)
            explored_nodes.append(current.position)

            # Reached destination goal node
            if current.position == goal:
                path = []
                curr = current
                while curr:
                    path.append(curr.position)
                    curr = curr.parent
                return path[::-1], explored_nodes  # Return reversed path (Start -> Goal)

            for neighbor_pos, move_cost in self.get_neighbors(current.position):
                if neighbor_pos in closed_set:
                    continue

                tentative_g = current.g_cost + move_cost

                if tentative_g < g_scores.get(neighbor_pos, float('inf')):
                    g_scores[neighbor_pos] = tentative_g
                    h_cost = self.heuristic(neighbor_pos, goal)
                    neighbor_node = Node(neighbor_pos, g_cost=tentative_g, h_cost=h_cost, parent=current)
                    node_map[neighbor_pos] = neighbor_node
                    heapq.heappush(open_set, neighbor_node)

        print("[WARNING] No valid path found!")
        return None, explored_nodes