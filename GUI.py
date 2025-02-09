import tkinter as tk
from tkinter import *
from tkinter import ttk
from algorithm import Algorithm
import numpy as np

X_START = 150
Y_START = 100
MAP_WIDTH = 1000
MAP_HEIGHT = 900
mint = "#BDFCC9"

class GUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("GUI")
        self.root.iconbitmap('icon.ico')
        self.root.geometry("1550x900")
        self.level_option_list = ["Level 1", "Level 2", "Level 3", "Level 4"]
        self.input_option_list = ["1", "2", "3", "4", "5"]
        self.input_option_list_level4 = ["1"]
        self.widgets = []
        self.level_option = tk.StringVar()
        self.input_option = tk.StringVar()
        self.algo_option = tk.IntVar()
        self.file = "input1_level1.txt"
        self.deliveryMap = DeliveryMap(self.root, self.file)
        self.root.img = tk.PhotoImage(file="robot.png")
        self.root.result = []

    def run(self):
        level_option, algo_option = self.level_option.get(), self.algo_option.get()
        algorithm = Algorithm(self.deliveryMap)
        path, algo_name = [], []

        if level_option == "Level 1" and algo_option == 1:
            path.append(algorithm.dfs_level1())
            algo_name.append("DFS")

            # self.deliveryMap.create_map()

            path.append(algorithm.bfs_level1())
            algo_name.append("BFS")

            self.deliveryMap.create_map()

            path.append(algorithm.ucs_level1())
            algo_name.append("UCS")


            path.append(algorithm.gbfs_level1())
            algo_name.append("GBFS")


            path.append(algorithm.a_star_level1())
            algo_name.append("A*")

        elif level_option == "Level 1" and algo_option == 2:
            path.append(algorithm.bfs_level1())
            algo_name.append("BFS")


            path.append(algorithm.dfs_level1())
            algo_name.append("DFS")


            path.append(algorithm.ucs_level1())
            algo_name.append("UCS")


            path.append(algorithm.gbfs_level1())
            algo_name.append("GBFS")


            path.append(algorithm.a_star_level1())
            algo_name.append("A*")
        elif level_option == "Level 1" and algo_option == 3:
            path.append(algorithm.ucs_level1())
            algo_name.append("UCS")


            path.append(algorithm.dfs_level1())
            algo_name.append("DFS")


            path.append(algorithm.bfs_level1())
            algo_name.append("BFS") 

                    
            path.append(algorithm.gbfs_level1())
            algo_name.append("GBFS")


            path.append(algorithm.a_star_level1())
            algo_name.append("A*") 
        elif level_option == "Level 1" and algo_option == 4:
            path.append(algorithm.gbfs_level1())
            algo_name.append("GBFS")


            path.append(algorithm.dfs_level1())
            algo_name.append("DFS") 


            path.append(algorithm.bfs_level1())
            algo_name.append("BFS")


            path.append(algorithm.ucs_level1())
            algo_name.append("UCS")


            path.append(algorithm.a_star_level1())
            algo_name.append("A*")
        elif level_option == "Level 1" and algo_option == 5:
            path.append(algorithm.a_star_level1())
            algo_name.append("A*")


            path.append(algorithm.dfs_level1())
            algo_name.append("DFS")


            path.append(algorithm.bfs_level1())
            algo_name.append("BFS")


            path.append(algorithm.ucs_level1())
            algo_name.append("UCS")


            path.append(algorithm.gbfs_level1())
            algo_name.append("GBFS")
        # Level 2
        elif level_option == "Level 2" and algo_option == 3:
            path.append(algorithm.ucs_level2(self.deliveryMap.t))
            algo_name.append("UCS-2")
        # Level 3
        elif level_option == "Level 3" and algo_option == 5:
            result, _, _ = algorithm.search_level3(self.deliveryMap.t, self.deliveryMap.f)
            path.append(result)
            algo_name.append("A*-2")
        #Level 4
        elif level_option == "Level 4" and algo_option == 5:
            path.append(algorithm.search_level4(self.deliveryMap.t, self.deliveryMap.f))
            algo_name.append("A*-3")     

        writeResultPath(self.file, path, algo_name)   

    def level_option_changed(self, *args):
        for widget in self.widgets:
            widget.place_forget()
        self.widgets.clear()

        for res in self.root.result:
            res.place_forget()
        self.root.result.clear()

        if self.level_option.get() == "Level 1":
            dfs_button = ttk.Radiobutton(self.root, text = "Depth-First Search", variable = self.algo_option, value = 1)
            dfs_button.place(x = X_START - 40, y = Y_START + 300, anchor="nw")
            self.widgets.append(dfs_button)

            bfs_button = ttk.Radiobutton(self.root, text = "Breadth-First Search", variable = self.algo_option, value = 2)
            bfs_button.place(x = X_START + 160, y = Y_START + 300, anchor="nw")
            self.widgets.append(bfs_button)

            ucs_button = ttk.Radiobutton(self.root, text = "Uniform-Cost Search", variable = self.algo_option, value = 3)
            ucs_button.place(x = X_START - 40, y = Y_START + 360, anchor="nw")
            self.widgets.append(ucs_button)

            gbfs_button = ttk.Radiobutton(self.root, text = "Greedy Best First Search", variable = self.algo_option, value = 4)
            gbfs_button.place(x = X_START + 160, y = Y_START + 360, anchor="nw")
            self.widgets.append(gbfs_button)

            a_button = ttk.Radiobutton(self.root, text = "A* Search", variable = self.algo_option, value = 5)
            a_button.place(x = X_START + 80, y = Y_START + 420, anchor="nw")
            self.widgets.append(a_button)

            self.input_option.set(self.input_option_list[0])
            self.input_menu = tk.OptionMenu(self.root, self.input_option, *self.input_option_list, command=self.input_option_changed)
            self.input_menu.place(x = X_START + 275, y = Y_START + 200, anchor="nw") 
            self.widgets.append(self.input_menu)

        elif self.level_option.get() == "Level 2":
            ucs_button = ttk.Radiobutton(self.root, text = "Uniform-Cost Search-2", variable = self.algo_option, value = 3)
            ucs_button.place(x = X_START - 40, y = Y_START + 300, anchor="nw")
            self.widgets.append(ucs_button)

            self.input_option.set(self.input_option_list[0])
            self.input_menu = tk.OptionMenu(self.root, self.input_option, *self.input_option_list, command=self.input_option_changed)
            self.input_menu.place(x = X_START + 275, y = Y_START + 200, anchor="nw") 
            self.widgets.append(self.input_menu)

        elif self.level_option.get() == "Level 3":
            a_button = ttk.Radiobutton(self.root, text = "A* Search-2", variable = self.algo_option, value = 5)
            a_button.place(x = X_START - 40, y = Y_START + 300, anchor="nw")
            self.widgets.append(a_button)

            self.input_option.set(self.input_option_list[0])
            self.input_menu = tk.OptionMenu(self.root, self.input_option, *self.input_option_list, command=self.input_option_changed)
            self.input_menu.place(x = X_START + 275, y = Y_START + 200, anchor="nw") 
            self.widgets.append(self.input_menu)

        elif self.level_option.get() == "Level 4":
            a_button = ttk.Radiobutton(self.root, text = "A* Search-3", variable = self.algo_option, value = 5)
            a_button.place(x = X_START - 40, y = Y_START + 300, anchor="nw")
            self.widgets.append(a_button)

            self.input_option.set(self.input_option_list[0])
            option_list_lv4 = self.input_menu = tk.OptionMenu(self.root, self.input_option, *self.input_option_list_level4, command=self.input_option_changed)
            self.input_menu.place(x = X_START + 275, y = Y_START + 200, anchor="nw") 
            self.widgets.append(option_list_lv4)

        level_index = self.level_option.get().split(' ')[1]
        file_index = self.input_option.get()
        self.file = f"input{file_index}_level{level_index}.txt"
        self.deliveryMap = DeliveryMap(self.root, self.file)

    def input_option_changed(self, *arg):
        for res in self.root.result:
            res.place_forget()
        self.root.result.clear()

        level_index = self.level_option.get().split(' ')[1]
        file_index = self.input_option.get()
        self.file = f"input{file_index}_level{level_index}.txt"
        self.deliveryMap = DeliveryMap(self.root, self.file)

    def create(self):
        schoolName = tk.Label(self.root, text="University of Science - VNU-HCM", font=("Times New Roman", 13))
        schoolName.place(x = X_START, y = Y_START, anchor="nw")

        projectTitle = tk.Label(self.root, text="Project 1: Delivery System", font=("Times New Roman", 20, "bold"))
        projectTitle.place(x = X_START - 30, y = Y_START + 30, anchor="nw")

        subject_info = tk.Label(self.root, text="Subject: Fundamentals of Artificial Intelligence", font=("Times New Roman", 13))
        class_info = tk.Label(self.root, text="Class: 22CLC07", font=("Times New Roman", 13))
        group_info = tk.Label(self.root, text="Group: 09", font=("Times New Roman", 13))

        subject_info.place(x = X_START - 35, y = Y_START + 80, anchor="nw")
        class_info.place(x = X_START - 35, y = Y_START + 110, anchor="nw")
        group_info.place(x = X_START - 35, y = Y_START + 140, anchor="nw")

        level_info = tk.Label(self.root, text="Level: ", font=("Times New Roman", 13))
        level_info.place(x = X_START - 40, y = Y_START + 200, anchor="nw")
        
        self.level_option.set(self.level_option_list[0])
        self.level_menu = tk.OptionMenu(self.root, self.level_option, *self.level_option_list, command=self.level_option_changed)
        self.level_menu.place(x = X_START + 40, y = Y_START + 200, anchor="nw") 

        input_file = tk.Label(self.root, text="Input: ", font=("Times New Roman", 13))
        input_file.place(x = X_START + 200, y = Y_START + 200, anchor="nw")

        self.input_option.set(self.input_option_list[0])
        self.input_menu = tk.OptionMenu(self.root, self.input_option, *self.input_option_list, command=self.input_option_changed)
        self.input_menu.place(x = X_START + 275, y = Y_START + 200, anchor="nw") 
        self.widgets.append(self.input_menu)

        dfs_button = ttk.Radiobutton(self.root, text = "Depth-First Search", variable = self.algo_option, value = 1)
        dfs_button.place(x = X_START - 40, y = Y_START + 300, anchor="nw")
        self.widgets.append(dfs_button)

        bfs_button = ttk.Radiobutton(self.root, text = "Breadth-First Search", variable = self.algo_option, value = 2)
        bfs_button.place(x = X_START + 160, y = Y_START + 300, anchor="nw")
        self.widgets.append(bfs_button)

        ucs_button = ttk.Radiobutton(self.root, text = "Uniform-Cost Search", variable = self.algo_option, value = 3)
        ucs_button.place(x = X_START - 40, y = Y_START + 360, anchor="nw")
        self.widgets.append(ucs_button)

        gbfs_button = ttk.Radiobutton(self.root, text = "Greedy Best First Search", variable = self.algo_option, value = 4)
        gbfs_button.place(x = X_START + 160, y = Y_START + 360, anchor="nw")
        self.widgets.append(gbfs_button)

        a_button = ttk.Radiobutton(self.root, text = "A* Search", variable = self.algo_option, value = 5)
        a_button.place(x = X_START + 80, y = Y_START + 420, anchor="nw")
        self.widgets.append(a_button) 

        button = tk.Button( self.root, text="Run",
                            activebackground="blue", 
                            activeforeground="white",
                            bd=3,
                            bg="lightgray",
                            disabledforeground="gray",
                            fg="black",
                            font=("Arial", 12),
                            height=2,
                            highlightbackground="black",
                            width=15,
                            command = self.run
                            )
        button.place(x = X_START + 50, y = Y_START + 500, anchor="nw")

        self.root.mainloop()

class DeliveryMap:
    def __init__(self, root, file) -> None:
        self.root = root
        self.canvas = tk.Canvas(root, width = MAP_WIDTH, height = MAP_HEIGHT)
        self.map, self.t, self.f = readFile(filename=file)
        self.result = []

        # Calculate to center the delivery map
        self.cell_size = 40
        self.map_width = len(self.map[0]) * self.cell_size
        self.map_height = len(self.map) * self.cell_size
        self.x_offset = (MAP_WIDTH - self.map_width) // 2
        self.y_offset = (MAP_HEIGHT - self.map_height) // 2

        self.rectangles = {}
        
        self.create_map()

    def create_map(self): 
        for y in range(len(self.map)):
            for x in range(len(self.map[y])):
                color = "white"
                if self.map[y][x] == '0':
                    color = "white"
                elif self.map[y][x] == '-1':
                    color = "steelblue4"
                elif self.map[y][x][0] == 'S':
                    color = mint
                    self.start = (x, y)
                elif self.map[y][x][0] == 'G':
                    color = "lightpink"
                    self.goal = (x, y)
                elif self.map[y][x][0] == 'F':
                    color = "khaki1"
                else:
                    color = "lightblue2"
                                
                x1 = x * self.cell_size + self.x_offset
                y1 = y * self.cell_size + self.y_offset
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                self.rectangles[(y, x)] = self.canvas.create_rectangle(x1, y1, x2, y2, fill = color, outline = "black")
                if self.map[y][x] != '0' and self.map[y][x] != '-1':
                    self.canvas.create_text(x1 + 20, y1 + 20, text = self.map[y][x], fill = "black", font = "Times 12")  

        self.canvas.place(x = X_START + 350, y = 0, anchor="nw")

    def color_cell(self, cell, start, goal):
        if cell != goal and cell != start:
            self.canvas.itemconfig(self.rectangles[cell], fill="seashell3")
        elif cell == goal:
            self.canvas.itemconfig(self.rectangles[cell], fill="gold")

        self.pause(100)
    
    def pause(self, duration):
        self.canvas.update()
        self.canvas.after(duration)

    def draw_path(self, path : list):
        if len(path) == 2:
            for i in range(len(path) - 1):
                y1, x1 = path[i]
                y2, x2 = path[i + 1]
                x1 = x1 * self.cell_size + self.x_offset + 20
                y1 = y1 * self.cell_size + self.y_offset + 20
                x2 = x2 * self.cell_size + self.x_offset + 20
                y2 = y2 * self.cell_size + self.y_offset + 20

                self.line = self.canvas.create_line(x1, y1, x2, y2, fill="deeppink1", width=4)
                self.ai_img = self.canvas.create_image(x2, y2, image=self.root.img)
                self.pause(350)
                
                if i == len(path) - 2:
                    self.canvas.delete(self.ai_img)

        else:
            for i in range(len(path) - 1):
                y1, x1 = path[i]
                y2, x2 = path[i + 1]
                x1 = x1 * self.cell_size + self.x_offset + 20
                y1 = y1 * self.cell_size + self.y_offset + 20
                x2 = x2 * self.cell_size + self.x_offset + 20
                y2 = y2 * self.cell_size + self.y_offset + 20
             
                self.line = self.canvas.create_line(x1, y1, x2, y2, fill="darkseagreen1", width=4)
                if i == 0:
                    self.ai_img = self.canvas.create_image(x1, y1, image=self.root.img)                         
                    self.pause(250)
                    self.canvas.coords(self.ai_img, x2, y2)   
                else:
                    self.canvas.coords(self.ai_img, x2, y2)
                self.pause(250)

    def print_result_lv1(self, totalCost):
        for res in self.root.result:
            res.place_forget()
        self.root.result.clear()
        result = tk.Label(self.root, text=f"Total Cost: {totalCost}", font=("Times New Roman", 15, "bold"), foreground="red")
        result.place(x = X_START + 350, y = Y_START + 512, anchor="nw")
        self.root.result.append(result)

    def print_result_lv2(self, totalTime, totalCost):
        for res in self.root.result:
            res.place_forget()
        self.root.result.clear()

        result_time = tk.Label(self.root, text=f"Total time: {totalTime}", font=("Times New Roman", 15, "bold"), foreground="red")
        result_time.place(x = X_START + 250, y = Y_START + 512, anchor="nw")
        self.root.result.append(result_time)
        result = tk.Label(self.root, text=f"Total Cost: {totalCost}", font=("Times New Roman", 15, "bold"), foreground="red")
        result.place(x = X_START + 425, y = Y_START + 512, anchor="nw")
        self.root.result.append(result)

    def print_result_lv3(self, totalTime, totalCost):
        for res in self.root.result:
            res.place_forget()
        self.root.result.clear()
        
        result_time = tk.Label(self.root, text=f"Total time: {totalTime}", font=("Times New Roman", 15, "bold"), foreground="red")
        result_time.place(x = X_START + 250, y = Y_START + 512, anchor="nw")
        self.root.result.append(result_time)
        result = tk.Label(self.root, text=f"Total Cost: {totalCost}", font=("Times New Roman", 15, "bold"), foreground="red")
        result.place(x = X_START + 425, y = Y_START + 512, anchor="nw")
        self.root.result.append(result)

    def print_no_path(self, lv):
        for res in self.root.result:
            res.place_forget()
        self.root.result.clear()
        result = tk.Label(self.root, text="No path exists", font=("Times New Roman", 15, "bold"), foreground="red")
        result.place(x = X_START + 300, y = Y_START + 512, anchor="nw")
        self.root.result.append(result)

def readFile(filename):
    with open(filename, "r") as file:
        data = file.readlines()
        n, m, t, f = map(int, data[0].strip().split())
        maze = np.empty(shape=(n, m), dtype=object)
        for i in range(n):
            maze[i] = data[i + 1].strip().split()

    return maze, t, f

def writeResultPath(inputFilename, path, algo_name):
    filename = inputFilename.split('.')[0]
    filename = filename.split('_')
    input = filename[0][5]
    level = filename[1][5]
    outputFilename = "output" + input + "_level" + level + ".txt"
    with open(outputFilename, 'w') as file:
        for i in range(len(path)):
            file.write(algo_name[i] + ':\n')
            file.write("S\n")
            file.write(' '.join(str(cell) for cell in path[i]))
            file.write("\n")
        