import numpy as np
from queue import PriorityQueue
import heapq

class Algorithm:
    def __init__(self, deliveryMap) -> None:
        self.ui_map = deliveryMap

    def bfs_level1(self) -> list:  
        maze = self.ui_map.map
        rows, cols = maze.shape     # size of the maze

        # get the index of start and goal
        start_location = np.where(maze == 'S')
        goal_location = np.where(maze == 'G')
        start = (start_location[0][0], start_location[1][0])
        goal = (goal_location[0][0], goal_location[1][0])

        if start == goal:
            return [start]

        # keep track of visited cell and the path
        visited = np.zeros((rows, cols), dtype=bool)
        come_from = np.full((rows, cols, 2), -1, dtype=int)
        path = []

        frontier = []
        frontier.append(start)
        visited[start] = True
        totalCost = 0

        # Directions for moving in the maze (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
        while frontier:
            current = frontier.pop(0)
            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                # neighbor must be in the maze and is not an obstacle
                if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                    not visited[neighbor] and maze[neighbor] in {'0', 'G'}):

                    # Color neighbors
                    self.ui_map.color_cell(neighbor, start, goal)

                    # arrive the goal
                    if neighbor == goal:
                        path.append(neighbor)
                        while current != start:
                            totalCost += 1
                            path.append(current)
                            current = tuple(come_from[current])

                        path.append(start)
                        totalCost += 1
                        path.reverse()

                        # Draw path
                        self.ui_map.root.after(100, self.ui_map.draw_path(path))

                        # Print result
                        self.ui_map.print_result_lv1(totalCost)

                        # Generate map again for other algorithms
                        return path

                    # add neighbor to frontier, visited cell and the path
                    frontier.append(neighbor)
                    visited[neighbor] = True
                    come_from[neighbor] = current
            
        self.ui_map.print_no_path()
        return path
    
    def dfs_level1(self) -> list:
        maze = self.ui_map.map
        start_location = np.where(maze == 'S')
        goal_location = np.where(maze == 'G')

        start = (start_location[0][0], start_location[1][0])
        goal = (goal_location[0][0], goal_location[1][0])

        frontier = [start]
        visited = set(start)
        parent = {start: None}
        path = []
        totalCost = 0

        while frontier:
            current = frontier.pop()

            # Color neighbors
            self.ui_map.color_cell(current, start, goal)

            if current == goal:  
                while current is not None:
                    path.append(current)
                    current = parent[current]
                    totalCost += 1
                path.reverse()

                # Draw path
                self.ui_map.root.after(100, self.ui_map.draw_path(path))

                # Print result
                self.ui_map.print_result_lv1(totalCost - 1)

                # Generate map again for other algorithms
                return path

            visited.add(current)     
            x, y = current

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for i in range(len(directions) - 1, -1, -1):
                neighbor = (x + directions[i][0], y + directions[i][1])
                if neighbor in visited:
                    continue
         
                if (0 <= neighbor[0] < len(self.ui_map.map) and 0 <= neighbor[1] < len(self.ui_map.map[0])
                    and self.ui_map.map[neighbor] in {'0', 'G'}):
                    
                    frontier.append(neighbor)
                    parent[neighbor] = current        

        self.ui_map.print_no_path()
        return path
    
    def gbfs_level1(self) -> list:
        maze = self.ui_map.map
        start_location = np.where(maze == 'S')
        goal_location = np.where(maze == 'G')

        start = (start_location[0][0], start_location[1][0])
        goal = (goal_location[0][0], goal_location[1][0])
        
        visited = set()
        frontier = PriorityQueue()
        totalCost = 0
        frontier.put((0,start, totalCost, [start]))
        
        while not frontier.empty():
            _, current, totalCost, path=frontier.get()

            self.ui_map.color_cell(current, start, goal)
                       
            if current == goal:
                self.ui_map.root.after(100, self.ui_map.draw_path(path))

                # Print result
                self.ui_map.print_result_lv1(totalCost)

                return path
            
            visited.add(current)
            
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
            
                if neighbor not in visited and 0 <= neighbor[0] < len(self.ui_map.map) and 0 <= neighbor[1] < len(self.ui_map.map[0]) and self.ui_map.map[neighbor[0]][neighbor[1]] != '-1':
                    new_path = path + [neighbor]
                    
                    # Manhattan distance from neighbor to goal
                    heuristic = abs(int(neighbor[0]) - int(goal[0])) + abs(int(neighbor[1]) - int(goal[1]))
                    
                    frontier.put((heuristic, neighbor, totalCost + 1, new_path))
                    visited.add(neighbor)

        self.ui_map.print_no_path()            
        return []

        
    def ucs_level1(self)->list:
        maze = self.ui_map.map
        start_location = np.where(maze == 'S')
        goal_location = np.where(maze == 'G')

        start = (start_location[0][0], start_location[1][0])
        goal = (goal_location[0][0], goal_location[1][0])
        
        if start == goal:
            return [start]
        
        rows, cols = maze.shape
        visited = np.zeros((rows, cols), dtype=bool)
        parent = np.full((rows, cols, 2), -1, dtype=int)
        path = []
        totalCost = 0

        # Directions for moving in the maze (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        frontier = PriorityQueue()
        frontier.put((0,start))
        visited[start] = True
        while frontier.empty() == 0:
            current_cost, current = frontier.get()
            
            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                    not visited[neighbor] and maze[neighbor] in {'0', 'G'}):

                    # Color neighbors
                    self.ui_map.color_cell(neighbor, start, goal)

                    if neighbor == goal:
                        path.append(neighbor)
                        while current != start:
                            path.append(current)
                            current = tuple(parent[current])
                            totalCost += 1

                        path.append(start)
                        totalCost += 1
                        path.reverse()

                        # Draw path
                        self.ui_map.root.after(100, self.ui_map.draw_path(path))

                        # Print result
                        self.ui_map.print_result_lv1(totalCost)

                        return path
            
                    frontier.put((current_cost + int(maze[neighbor]), neighbor))
                    visited[neighbor] = True
                    parent[neighbor] = current

        self.ui_map.print_no_path()            
        return path
    
    def heuristic(self, a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star_level1(self) -> list:
        maze = self.ui_map.map
        rows, cols = maze.shape     # size of the maze

        # get index of start and goal
        start_location = np.where(maze == 'S')
        goal_location = np.where(maze == 'G')
        start = (start_location[0][0], start_location[1][0])
        goal = (goal_location[0][0], goal_location[1][0])

        totalCost = 0
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {}
        cost_to_get = {start: 0}

        # Directions for moving in the maze (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        path = []

        while frontier:
            current = heapq.heappop(frontier)[1]    # get the cell that has the smallest heuristic value
            # check all neighbor (up, down, left, right)
            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                # the neighbor must be in the maze and is not an obstacle
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor] in {'0', 'G'}:
                    # color the neighbor 
                    self.ui_map.color_cell(neighbor, start, goal)

                    if neighbor == goal:
                        path.append(neighbor)
                        while current in came_from:
                            path.append(current)
                            current = came_from[current]
                            totalCost += 1
                            
                        path.append(start)
                        totalCost += 1
                        path.reverse()

                        # draw path
                        self.ui_map.root.after(100, self.ui_map.draw_path(path))

                        # Print result
                        self.ui_map.print_result_lv1(totalCost)

                        return path
                
                    curr_cost = cost_to_get[current] + 1
                    # if neighbor is not in the frontier or the current cost smaller than the existed cost
                    if neighbor not in cost_to_get or curr_cost < cost_to_get[neighbor]:
                        came_from[neighbor] = current
                        cost_to_get[neighbor] = curr_cost
                        f_cost = curr_cost + self.heuristic(neighbor, goal)
                        heapq.heappush(frontier, (f_cost, neighbor))

        self.ui_map.print_no_path()
        return path
    
    # ================================ LEVEL 2 ================================

    def ucs_level2(self, t : int) -> list:
        maze = self.ui_map.map
        start_location = np.where(maze == 'S')
        goal_location = np.where(maze == 'G')

        start = (start_location[0][0], start_location[1][0])
        goal = (goal_location[0][0], goal_location[1][0])
        
        if start == goal:
            return [start]
        
        rows, cols = maze.shape
        visited = np.zeros((rows, cols), dtype=bool)
        parent = np.full((rows, cols, 2), -1, dtype=int)
        totalTime = 0
        path = []

        # Directions for moving in the maze (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        frontier = PriorityQueue()
        frontier.put((0, 0, start))
        visited[start] = True
        
        while frontier.empty() == 0:
            current_cost, current_step, current = frontier.get()
            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                
                if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                    not visited[neighbor] and not maze[neighbor] in {'-1', 'S'}):

                    # Color neighbors
                    self.ui_map.color_cell(neighbor, start, goal)

                    if neighbor == goal:
                        totalTime = current_cost + 1
                        totalCost = current_step + 1
                        if totalTime > t:
                            self.ui_map.print_no_path()
                            return []
                        
                        path.append(neighbor)
                        while current != start:
                            path.append(current)
                            current = tuple(parent[current])
                        path.append(start)
                        path.reverse()

                        # Draw path
                        self.ui_map.root.after(100, self.ui_map.draw_path(path))
                        self.ui_map.print_result_lv2(totalTime, totalCost)
                        return path

                    frontier.put((current_cost + int(maze[neighbor]) + 1, current_step + 1 , neighbor))
                    visited[neighbor] = True
                    parent[neighbor] = current
        
        self.ui_map.print_no_path()
        return []
    
    # ================================ LEVEL 3 ================================
    def a_star_level3(self, start, goal, max_time, max_fuel, cost_to_get=None, time_to_get=None, fuel_to_get=None) -> list:
        maze = self.ui_map.map
        rows, cols = maze.shape     # the size of the maze

        if cost_to_get is None: cost_to_get = {start: 0}
        if time_to_get is None: time_to_get = {start: 0}
        if fuel_to_get is None: fuel_to_get = {start: 0}

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {}

        # Directions for moving in the maze (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        path = []

        while frontier:
            current = heapq.heappop(frontier)[1]    # get the cell that has the smallest heuristic value
            # check all neighbor (up, down, left, right)
            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                # the neighbor must be in the maze and is not an obstacle or start index
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor] not in  {'-1', 'S'}:
                    self.ui_map.color_cell(neighbor, start, goal)
                    if neighbor == goal:
                        if time_to_get[current] <= max_time - 1 and fuel_to_get[current] <= max_fuel - 1:
                            # if the goal is the fuel_station
                            if maze[neighbor][0] == 'F':
                                # update the cost from start to here
                                new_g_score = cost_to_get[current] + 1
                                cost_to_get.clear()
                                cost_to_get.update({neighbor: new_g_score})

                                # update the time from start to here
                                time_to_refuel = int(maze[neighbor].split('F')[1])
                                new_time = time_to_get[current] + time_to_refuel + 1
                                time_to_get.clear()
                                time_to_get.update({neighbor: new_time})

                                # refuel 
                                fuel_to_get.clear()
                                fuel_to_get.update({neighbor: 0})

                                # update cost and time
                                cost = new_g_score
                                time = new_time
                            else:
                                cost = cost_to_get[current] + 1
                                time = time_to_get[current] + 1

                            # track back to get the path
                            path.append(neighbor)
                            while current in came_from:
                                path.append(current)
                                current = came_from[current]
                            
                            if maze[start][0] != 'F':
                                path.append(start)
                                
                            path.reverse()
                            if maze[neighbor][0] == 'F':
                                return path, new_g_score, new_time
                            return path, cost, time
                        return -1, 0, 0
                    
                    # update the cost, time and fuel while
                    curr_cost = cost_to_get[current] + 1
                    curr_fuel = fuel_to_get[current] + 1
                    if maze[neighbor] == '0':   # path
                        curr_time = time_to_get[current] + 1
                    elif maze[neighbor][0] == 'F':  # fuel_station
                        time_to_refuel = int(maze[neighbor].split('F')[1])
                        curr_time = time_to_get[current] + time_to_refuel + 1
                        curr_fuel = 0
                    else:   # toll booth
                        curr_time = time_to_get[current] + int(maze[neighbor]) + 1
                    
                    # if neighbor is still not in the frontier or the time to get there less than the existed one (if the time is equal, compare the cost)
                    if neighbor not in cost_to_get or curr_time < time_to_get[neighbor] or (curr_time == time_to_get[neighbor] and curr_cost < cost_to_get[neighbor]):
                        came_from[neighbor] = current
                        cost_to_get[neighbor] = curr_cost
                        time_to_get[neighbor] = curr_time
                        fuel_to_get[neighbor] = curr_fuel
                        f_cost = curr_time + self.heuristic(neighbor, goal)
                        heapq.heappush(frontier, (f_cost, neighbor))
        return None, 0, 0

    def search_level3(self, max_time, max_fuel, start=None, goal=None, fuel_station_indices=None) -> list:
        maze = self.ui_map.map
        totalTime = 0
        totalCost = 0

        # get the index of start and goal
        if start is None and goal is None:
            start_location = np.where(maze == 'S')
            goal_location = np.where(maze == 'G')
            start = (start_location[0][0], start_location[1][0])
            goal = (goal_location[0][0], goal_location[1][0])

        path, totalCost, totalTime = self.a_star_level3(start, goal, max_time, max_fuel)
        # if the path exist and with time and fuel are enough to get to the goal
        if not path or path != -1:
            self.ui_map.root.after(100, self.ui_map.draw_path(path))
            self.ui_map.print_result_lv3(totalTime, totalCost)
            return path, totalCost, totalTime

        # else find the indices of all fuel_station in the maze
        if fuel_station_indices is None:
            mask = np.char.startswith(maze.tolist(), 'F')
            rows = np.where(mask)[0]
            cols = np.where(mask)[1]
            # sort the fuel_station's indices in the descending order of the heuristic value from start to that fuel_station
            fuel_station_indices = sorted(list(zip(rows, cols)), key=lambda x: self.heuristic(x, goal))
        else:
            fuel_station_indices.pop(0)

        # Generate again map
        self.ui_map.create_map()

        # for each fuel_station
        for fuel_station in fuel_station_indices:

            # find the path from the closest fuel station to 'goal'
            first_path, totalCost1, totalTime1 = self.a_star_level3(fuel_station, goal, max_time, max_fuel)
            if first_path and first_path != -1:
                # find the path from 'start' to that fuel station
                second_path, totalCost2, totalTime2 = self.search_level3(max_time, max_fuel, start, fuel_station, fuel_station_indices)
                # if 2 paths exist, add together and return the final path
                if second_path and second_path != -1:
                    result_path = second_path + first_path
                    totalTime = totalTime2 + totalTime1
                    totalCost = totalCost2 + totalCost1
                    
                    # check the time is valid or not
                    if totalTime > max_time:
                        self.ui_map.create_map()
                        continue

                    self.ui_map.root.after(100, self.ui_map.draw_path(result_path))
                    self.ui_map.print_result_lv3(totalTime, totalCost)
                    
                    return result_path, totalCost, totalTime
            return None, 0, 0
        
        self.ui_map.print_no_path()
        return None, 0, 0
    
    # ========== LEVEL 4 ==========
    def find_start_goal_pairs(self, maze):
        starts = []
        goals = []

        # Loop through the maze to find 'S' and 'G' with digits
        for i in range(len(maze)):
            for j in range(len(maze[0])):
                if maze[i][j].startswith('S') and maze[i][j][1:].isdigit():
                    starts.append((i, j))
                elif maze[i][j].startswith('G') and maze[i][j][1:].isdigit():
                    goals.append((i, j))

        return starts, goals

    def calculate_cost(self, current_cost, current_time, current_fuel, cell_value):
        if cell_value.isdigit():
            new_time_cost = int(cell_value)
            new_fuel_cost = int(cell_value)
        elif cell_value.startswith('F'):
            refuel_time = int(cell_value[1:])  # Extract the refuel time from 'F1', 'F2', ..., 'F10'
            new_time_cost = refuel_time
            new_fuel_cost = refuel_time
        else:
            new_time_cost = 1
            new_fuel_cost = 1

        return current_cost + 1, current_time - new_time_cost, current_fuel - new_fuel_cost

    def a_star_search_level4(self, start, goal, maze, max_fuel):
        rows, cols = maze.shape
        frontier = []
        heapq.heappush(frontier, (0, start, 0, max_fuel))  # (priority, current_node, current_cost, current_fuel)
        came_from = {}
        cost_to_get = {start: (0, 0)}  # (cost, time, fuel)

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while frontier:
            current_cost, current, current_time, current_fuel = heapq.heappop(frontier)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()  # Reverse to get the correct order from start to goal
                return path
            
            if self.is_start_location(start):               
                starts, goals = self.find_start_goal_pairs(maze)
                
                if current in starts or current in goals:
                    continue

            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                # Check if the neighbor is valid
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor] != '-1':
                    new_cost, new_time, new_fuel = self.calculate_cost(current_cost, current_time, current_fuel, maze[neighbor])

                    if neighbor not in cost_to_get or (new_cost < cost_to_get[neighbor][0] and new_fuel >= 0):
                        cost_to_get[neighbor] = (new_cost, new_time, new_fuel)
                        priority = new_cost + self.heuristic(neighbor, goal)
                        heapq.heappush(frontier, (priority, neighbor, new_cost, new_fuel))
                        came_from[neighbor] = current

        return None  # If the goal cannot be reached, return None
    
    def is_start_location(self, start_check):
        maze = self.ui_map.map
        
        start_location = np.where(maze == 'S')
        goal_location = np.where(maze == 'G')
        start = (start_location[0][0], start_location[1][0])
        goal = (goal_location[0][0], goal_location[1][0])
        
        if start_check == start:
            return True
        else:
            return False

    def find_all_paths(self, starts, goals, max_fuel):
        paths = []
        visited_goals = set()

        for start, goal in zip(starts, goals):
            # Check if the current goal has already been visited
            if goal in visited_goals:
                continue

            path = self.a_star_search_level4(start, goal, self.ui_map.map, max_fuel)
            if path:
                paths.append(path)
                visited_goals.add(goal)

        return paths

    def search_level4(self, max_time, max_fuel) -> list:
        maze = self.ui_map.map
        
        start_location = np.where(maze == 'S')
        goal_location = np.where(maze == 'G')
        start = (start_location[0][0], start_location[1][0])
        goal = (goal_location[0][0], goal_location[1][0])

        path_main = self.a_star_search_level4(start, goal, self.ui_map.map, max_fuel)
        
        starts, goals = self.find_start_goal_pairs(maze)

        all_paths = self.find_all_paths(starts, goals, max_fuel)
        
        all_paths.append(path_main)

        if all_paths:
            max_len = max(len(sublist) for sublist in all_paths)

            for col in range(1, max_len):
                for row in all_paths:
                    if col < len(row):
                        path = []
                        path.append(row[col - 1])
                        path.append(row[col])
                        self.ui_map.draw_path(path)

        path_result = path_main

        if path_result:
            self.ui_map.root.after(100, self.ui_map.draw_path(path_result))
            return path_result
        else: 
            self.ui_map.print_no_path()
            return None