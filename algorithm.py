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
        parent = np.full((rows, cols, 2), -1, dtype=int)
        path = []

        frontier = []
        frontier.append(start)
        visited[start] = True

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
                            path.append(current)
                            current = tuple(parent[current])

                        path.append(start)
                        path.reverse()

                        # Draw path
                        self.ui_map.root.after(100, self.ui_map.draw_path(path))
                        return path

                    # add neighbor to frontier, visited cell and the path
                    frontier.append(neighbor)
                    visited[neighbor] = True
                    parent[neighbor] = current
            
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

        while frontier:
            current = frontier.pop()

            # Color neighbors
            self.ui_map.color_cell(current, start, goal)

            if current == goal:  
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()

                # Draw path
                self.ui_map.root.after(100, self.ui_map.draw_path(path))
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

        return path
    
    def gbfs_level1(self) -> list:
        maze = self.ui_map.map
        start_location = np.where(maze == 'S')
        goal_location = np.where(maze == 'G')

        start = (start_location[0][0], start_location[1][0])
        goal = (goal_location[0][0], goal_location[1][0])
        
        visited = set()
        frontier = PriorityQueue()
        frontier.put((0,start,[start]))
        
        while frontier:
            _, current, path=frontier.get()

            self.ui_map.color_cell(current, start, goal)
                       
            if current == goal:
                self.ui_map.root.after(100, self.ui_map.draw_path(path))
                return path
            
            
            visited.add(current)
            
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
            
                if neighbor not in visited and 0 <= neighbor[0] < len(self.ui_map.map) and 0 <= neighbor[1] < len(self.ui_map.map[0]) and self.ui_map.map[neighbor[0]][neighbor[1]] != '-1':
                    new_path = path + [neighbor]
                    
                    # Manhattan distance from neighbor to goal
                    heuristic = abs(int(neighbor[0]) - int(goal[0])) + abs(int(neighbor[1]) - int(goal[1]))
                    
                    frontier.put((heuristic, neighbor, new_path))
                    visited.add(neighbor)
                    
        return None

        
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

        # Directions for moving in the maze (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        frontier = PriorityQueue()
        frontier.put((0,start))
        visited[start] = True
        
        while frontier:
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
                        path.append(start)
                        path.reverse()

                        # Draw path
                        self.ui_map.root.after(100, self.ui_map.draw_path(path))
                        return path
            
                    frontier.put((current_cost + int(maze[neighbor]), neighbor))
                    visited[neighbor] = True
                    parent[neighbor] = current
                    
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
                            
                        path.append(start)
                        path.reverse()

                        # draw path
                        self.ui_map.root.after(100, self.ui_map.draw_path(path))
                        return path
                
                    curr_cost = cost_to_get[current] + 1
                    # if neighbor is not in the frontier or the current cost smaller than the existed cost
                    if neighbor not in cost_to_get or curr_cost < cost_to_get[neighbor]:
                        came_from[neighbor] = current
                        cost_to_get[neighbor] = curr_cost
                        f_cost = curr_cost + self.heuristic(neighbor, goal)
                        heapq.heappush(frontier, (f_cost, neighbor))

        return None
    
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
        frontier.put((0,start))
        visited[start] = True
        
        while frontier:
            current_cost, current = frontier.get()
            for direction in directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                
                if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                    not visited[neighbor] and not maze[neighbor] in {'-1', 'S'}):

                    # Color neighbors
                    self.ui_map.color_cell(neighbor, start, goal)

                    if neighbor == goal:
                        totalTime = current_cost + 1
                        if totalTime > t:
                            return path
                        
                        path.append(neighbor)
                        while current != start:
                            path.append(current)
                            current = tuple(parent[current])
                        path.append(start)
                        path.reverse()

                        # Draw path
                        self.ui_map.root.after(100, self.ui_map.draw_path(path))
                        self.ui_map.print_result(totalTime)
                        return path

                    frontier.put((current_cost + int(maze[neighbor]) + 1, neighbor))
                    visited[neighbor] = True
                    parent[neighbor] = current
        
        return path
    
    # ===== LEVEL 3 =====
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

                            path.append(neighbor)
                            while current in came_from:
                                path.append(current)
                                current = came_from[current]
                                
                            path.append(start)
                            path.reverse()
                            return path
                        return -1
                    
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
        return None

    def search_level3(self, max_time, max_fuel) -> list:
        maze = self.ui_map.map

        # get the index of start and goal
        start_location = np.where(maze == 'S')
        goal_location = np.where(maze == 'G')
        start = (start_location[0][0], start_location[1][0])
        goal = (goal_location[0][0], goal_location[1][0])

        path = self.a_star_level3(start, goal, max_time, max_fuel)
        # if time and fuel is enough to get to the goal
        if path != -1:
            self.ui_map.root.after(100, self.ui_map.draw_path(path))
            return path

        # else find the indices of all fuel_station in the maze
        mask = np.char.startswith(maze.tolist(), 'F')
        rows = np.where(mask)[0]
        cols = np.where(mask)[1]

        # Generate again map
        self.ui_map.create_map()

        # sort the fuel_station's indices in the descending order of the heuristic value from start to that fuel_station
        fuel_station_indices = sorted(list(zip(rows, cols)), key=lambda x: self.heuristic(x, goal))

        # for each fuel_station
        for fuel_station in fuel_station_indices:
            cost_to_get = {start: 0}
            time_to_get = {start: 0}
            fuel_to_get = {start: 0}

            # find the path from start to that
            first_path = self.a_star_level3(start, fuel_station, max_time, max_fuel, cost_to_get, time_to_get, fuel_to_get)

            # if the path from start to that fuel_station exists
            if first_path and first_path != -1:
                # find the path from that fuel_station to goal
                second_path = self.a_star_level3(fuel_station, goal, max_time, max_fuel, cost_to_get, time_to_get, fuel_to_get)
                # if the path exist, return return the result path
                if second_path and second_path != -1:
                    result_path = first_path + second_path
                    self.ui_map.root.after(100, self.ui_map.draw_path(result_path))
                    return result_path
                
            # else move to next fuel_station
            continue

        return None