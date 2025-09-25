import heapq
import time
import random

class GridCity:
    def __init__(self, width, height, terrain_costs, static_obstacles, dynamic_obstacles):
        self.width = width
        self.height = height
        self.terrain_costs = terrain_costs
        self.static_obstacles = set(static_obstacles)
        self.dynamic_obstacles = dynamic_obstacles

    def get_cost(self, pos):
        return self.terrain_costs.get(pos, 1)

    def is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height and pos not in self.static_obstacles

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_valid((nx, ny)):
                neighbors.append((nx, ny))
        return neighbors

class Agent:
    def __init__(self, start, goal, grid_city):
        self.start = start
        self.goal = goal
        self.grid = grid_city
        self.path = []
        self.position = start
        self.fuel = 1000  # Example constraint

    # A* Search with Admissible Heuristic (Manhattan Distance)
    def a_star_search(self, start_pos, goal_pos):
        open_set = [(0, start_pos)]  # (f_cost, position)
        came_from = {}
        g_cost = {start_pos: 0}
        nodes_expanded = 0

        while open_set:
            f_cost, current_pos = heapq.heappop(open_set)
            nodes_expanded += 1

            if current_pos == goal_pos:
                return self._reconstruct_path(came_from, current_pos), g_cost[current_pos], nodes_expanded

            for neighbor in self.grid.get_neighbors(current_pos):
                if neighbor in self.grid.dynamic_obstacles:
                    continue
                
                new_g_cost = g_cost[current_pos] + self.grid.get_cost(neighbor)
                
                if new_g_cost < g_cost.get(neighbor, float('inf')):
                    came_from[neighbor] = current_pos
                    g_cost[neighbor] = new_g_cost
                    h_cost = abs(neighbor[0] - goal_pos[0]) + abs(neighbor[1] - goal_pos[1])
                    f_cost = new_g_cost + h_cost
                    heapq.heappush(open_set, (f_cost, neighbor))

        return None, float('inf'), nodes_expanded

    # Uniform-Cost Search (uninformed)
    def uniform_cost_search(self, start_pos, goal_pos):
        open_set = [(0, start_pos)]  # (cost, position)
        came_from = {}
        g_cost = {start_pos: 0}
        nodes_expanded = 0

        while open_set:
            cost, current_pos = heapq.heappop(open_set)
            nodes_expanded += 1

            if current_pos == goal_pos:
                return self._reconstruct_path(came_from, current_pos), g_cost[current_pos], nodes_expanded

            for neighbor in self.grid.get_neighbors(current_pos):
                if neighbor in self.grid.dynamic_obstacles:
                    continue

                new_g_cost = g_cost[current_pos] + self.grid.get_cost(neighbor)

                if new_g_cost < g_cost.get(neighbor, float('inf')):
                    came_from[neighbor] = current_pos
                    g_cost[neighbor] = new_g_cost
                    heapq.heappush(open_set, (new_g_cost, neighbor))
        
        return None, float('inf'), nodes_expanded

    # Helper function to reconstruct the path
    def _reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(self.start)
        path.reverse()
        return path

    # Local Search Replanning (using a simple Hill-Climbing strategy)
    def hill_climbing_search(self, start_pos, goal_pos):
        current_pos = start_pos
        path = [current_pos]
        cost = 0
        nodes_expanded = 1

        for _ in range(500):  # Limit iterations to avoid infinite loops
            if current_pos == goal_pos:
                return path, cost, nodes_expanded

            neighbors = self.grid.get_neighbors(current_pos)
            if not neighbors:
                return None, float('inf'), nodes_expanded # Dead end

            # Find the neighbor closest to the goal (greedy approach)
            best_neighbor = min(neighbors, key=lambda p: abs(p[0] - goal_pos[0]) + abs(p[1] - goal_pos[1]))
            
            if best_neighbor in self.grid.dynamic_obstacles:
                # If path is blocked, try a random restart
                current_pos = random.choice(neighbors)
            else:
                current_pos = best_neighbor
            
            path.append(current_pos)
            cost += self.grid.get_cost(current_pos)
            nodes_expanded += 1
        
        return None, float('inf'), nodes_expanded # Did not find the goal

    # Function to simulate agent movement and dynamic obstacles
    def simulate(self, algorithm_name):
        self.position = self.start
        self.path = []
        total_cost = 0
        nodes_expanded = 0
        
        # Initial search
        search_start_time = time.time()
        if algorithm_name == 'A*':
            path, cost, expanded = self.a_star_search(self.position, self.goal)
        elif algorithm_name == 'UCS':
            path, cost, expanded = self.uniform_cost_search(self.position, self.goal)
        elif algorithm_name == 'Hill-Climbing':
            path, cost, expanded = self.hill_climbing_search(self.position, self.goal)
        else:
            raise ValueError("Invalid algorithm name")
        
        total_search_time = time.time() - search_start_time
        nodes_expanded += expanded

        if not path:
            return None, total_cost, total_search_time, nodes_expanded
        
        # Simulate movement with replanning
        planned_path = path[1:]
        while planned_path:
            next_step = planned_path.pop(0)

            # Check for dynamic obstacles and replan if needed
            if next_step in self.grid.dynamic_obstacles:
                print(f"Obstacle detected at {next_step}. Replanning...")
                # The agent stays put and re-plans
                search_start_time = time.time()
                if algorithm_name == 'A*':
                    new_path, new_cost, new_expanded = self.a_star_search(self.position, self.goal)
                elif algorithm_name == 'UCS':
                    new_path, new_cost, new_expanded = self.uniform_cost_search(self.position, self.goal)
                else:
                    new_path, new_cost, new_expanded = self.hill_climbing_search(self.position, self.goal)

                total_search_time += time.time() - search_start_time
                nodes_expanded += new_expanded

                if not new_path:
                    print("Failed to find a new path.")
                    return None, total_cost, total_search_time, nodes_expanded
                
                planned_path = new_path[1:]
                next_step = planned_path.pop(0)

            # Move and update state
            self.position = next_step
            self.path.append(self.position)
            total_cost += self.grid.get_cost(self.position)
            
            # Simulate dynamic obstacle movement
            self.grid.dynamic_obstacles = {(x+1, y) for x, y in self.grid.dynamic_obstacles}
            
        return self.path, total_cost, total_search_time, nodes_expanded

# Main Execution Block
if __name__ == "__main__":
    # --- Map Instance 1: Simple Map with Varying Costs ---
    map1_terrain = {(x, y): 2 for x in range(3, 7) for y in range(3, 7)}
    map1_obstacles = {(2, 5), (5, 2), (5, 8)}
    map1_dynamic_obstacles = {(4, 4), (5, 5)}
    city1 = GridCity(10, 10, map1_terrain, map1_obstacles, map1_dynamic_obstacles)
    start1, goal1 = (0, 0), (9, 9)

    print("--- Running on Map 1 (10x10 grid) ---")
    results = []

    for algo in ['A*', 'UCS', 'Hill-Climbing']:
        agent = Agent(start1, goal1, city1)
        path, cost, search_time, expanded = agent.simulate(algo)
        results.append((algo, cost, search_time, expanded, "Found" if path else "Failed"))
    
    print("\n--- Map 1 Results ---")
    print(f"{'Algorithm':<15}{'Path Cost':<12}{'Search Time (s)':<18}{'Nodes Expanded':<18}{'Status':<10}")
    print("-" * 75)
    for res in results:
        print(f"{res[0]:<15}{res[1]:<12.2f}{res[2]:<18.4f}{res[3]:<18}{res[4]:<10}")
    
    # --- Map Instance 2: Larger Map with More Obstacles ---
    map2_terrain = {(x, y): 3 for x in range(5, 15) for y in range(5, 15)}
    map2_obstacles = {(x, y) for x in range(10) for y in [8, 9, 10]} | \
                     {(x, y) for x in [12, 13, 14] for y in range(10, 20)}
    map2_dynamic_obstacles = {(1, 1), (1, 2)}
    city2 = GridCity(20, 20, map2_terrain, map2_obstacles, map2_dynamic_obstacles)
    start2, goal2 = (0, 0), (19, 19)

    print("\n--- Running on Map 2 (20x20 grid) ---")
    results = []
    
    for algo in ['A*', 'UCS', 'Hill-Climbing']:
        agent = Agent(start2, goal2, city2)
        path, cost, search_time, expanded = agent.simulate(algo)
        results.append((algo, cost, search_time, expanded, "Found" if path else "Failed"))
    
    print("\n--- Map 2 Results ---")
    print(f"{'Algorithm':<15}{'Path Cost':<12}{'Search Time (s)':<18}{'Nodes Expanded':<18}{'Status':<10}")
    print("-" * 75)
    for res in results:
        print(f"{res[0]:<15}{res[1]:<12.2f}{res[2]:<18.4f}{res[3]:<18}{res[4]:<10}")