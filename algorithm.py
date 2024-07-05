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
                    self.ui_map.color_cell(neighbor, goal)

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
    
    def dfs_level1(self) -> list:
        start_location = np.where(self.ui_map.map == 'S')
        goal_location = np.where(self.ui_map.map == 'G')

        start = (start_location[0][0], start_location[1][0])
        goal = (goal_location[0][0], goal_location[1][0])

        stack = [start]
        visited = set()
        parent = {start: None}

        while stack:
            current = stack.pop()
            if current == goal:
                break

            visited.add(current)
            x, y = current

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                if neighbor in visited:
                    continue
         
                if (0 <= neighbor[0] < len(self.ui_map.map) and 0 <= neighbor[1] < len(self.ui_map.map[0])
                    and self.ui_map.map[neighbor] in {'0', 'G'}):
                    
                    stack.append(neighbor)
                    parent[neighbor] = current

                    # Color neighbors
                    #self.ui_map.color_cell(neighbor, goal)

        # Reconstruct the path from end to start
        current = goal
        path = []
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()

        return path
                    

