import numpy as np

class Algorithm:
    def __init__(self, deliveryMap) -> None:
        self.ui_map = deliveryMap

    def bfs_level1(self) -> list:  
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
        
        frontier = []
        frontier.append(start)
        visited[start] = True
    
        while frontier:
            current = frontier.pop(0)
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
                        self.ui_map.root.after(3000, self.ui_map.draw_path(path))

                        return path
            
                    frontier.append(neighbor)
                    visited[neighbor] = True
                    parent[neighbor] = current
            
        return path
    
    def dfs_level1(self) -> list:
        start_location = np.where(self.ui_map.map == 'S')
        goal_location = np.where(self.ui_map.map == 'G')

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
                self.ui_map.root.after(3000, self.ui_map.draw_path(path))
              
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
        
        from queue import PriorityQueue
        
        start_location = np.where(self.ui_map.map == 'S')
        goal_location = np.where(self.ui_map.map == 'G')

        start = (start_location[0][0], start_location[1][0])
        goal = (goal_location[0][0], goal_location[1][0])
        
        visited = set()
        frontier = PriorityQueue()
        frontier.put((0,start,[start]))
        
        while frontier:
            _, current, path=frontier.get()

            self.ui_map.color_cell(current, start, goal)
                       
            if current == goal:
                self.ui_map.root.after(3000, self.ui_map.draw_path(path))
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
        from queue import PriorityQueue
        
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
                        self.ui_map.root.after(3000, self.ui_map.draw_path(path))

                        return path
            
                    frontier.put((current_cost + int(maze[neighbor]), neighbor))
                    visited[neighbor] = True
                    parent[neighbor] = current
                    
        return path
