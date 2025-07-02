import numpy as np
from agent.agent import *

class BFSAgent(Agent):
    def __init__(self, matrix):
        """
            Initializes the BFS agent class.

            Args:
                matrix (array): Initial game matrix
        """
        # Initializing the parent class
        super().__init__(matrix)
    
    def expand(self, node):
        nodes = [] # New created nodes will be added here

        for action in self.directions:  
            empty = (node.position[0] + action[0] , node.position[1] + action[1]) # position of empty tile in new matrix
            if empty[0] >= self.game_size or empty[1] >= self.game_size or empty[0] < 0 or empty[1] < 0: # checks if new empty tile is in matrix
                continue # if not do not add to list

            else: # if new empty tile is in matrix
                matrix = []
                for i in range(self.game_size): # this loop creates new matrix. changes empty tile to the new position
                    matrix.append([])
                    for j in range(self.game_size):
                        if (i,j) == empty:
                            matrix[i].append(0)
                        elif (i,j) == node.position:
                            matrix[i].append(node.matrix[empty[0]][empty[1]])
                        else:
                            matrix[i].append(node.matrix[i][j])

                nodes.append(Node(node, empty, matrix)) # creates new node and adds it to the list
                self.generated_node += 1
        return nodes # returns new nodes
    
    def tree_solve(self):
        """
            Solves the game using tree base BFS algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """
        self.frontier = [Node(None, self.empty_tile,self.initial_matrix)] # frontier set
        
        self.generated_node += 1
        self.maximum_node_in_memory = 1

        found = False # true if solution found
        while self.frontier: # Continue until find the solution or frontier is empty
            node = self.frontier.pop(0) # select first added node to expand (FIFO)
            self.expanded_node += 1

            for child in self.expand(node): # expand the node
                if self.checkEqual(self.desired_matrix, child.matrix): # solution has found if child has desired matrix 
                    return self.get_moves(child)
                else:
                    self.frontier.append(child) # add child node to frontier to expand

                    self.maximum_node_in_memory = max(len(self.frontier), self.maximum_node_in_memory) 
                    
        return None # return the path
    def graph_solve(self):
        """
            Solves the game using graph base BFS algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """

        self.frontier = [Node(None, self.empty_tile,self.initial_matrix)] # create frontier set
        self.explored = [Node(None, self.empty_tile,self.initial_matrix)] # create explored set
        self.maximum_node_in_memory += 1
        self.generated_node += 1

        found = False # true iif solution has found
        while self.frontier and not found: # continue until find the solution or frontier is empty
            node = self.frontier.pop(0) # select first added node to expand (FIFO)
            self.expanded_node += 1
            for child in self.expand(node): # expand child node
                if self.checkEqual(self.desired_matrix, child.matrix): # solution has found if node has desired matrix
                    self.maximum_node_in_memory += 1
                    found = True
                    break
                else:
                    seen = False
                    for exp in self.explored: # check if node already explored or not
                        if self.checkEqual(child.matrix, exp.matrix):
                            seen = True
                    if not seen: # if node has not been explored before add it to frontier and explored set
                        self.frontier.append(child)
                        self.explored.append(child)
                        self.maximum_node_in_memory += 1
        return self.get_moves(child) # return path
class DFSAgent(Agent):
    def __init__(self, matrix):
        """
            Initializes the DFS agent class.

            Args:
                matrix (array): Initial game matrix
        """
        # Initializing the parent class
        super().__init__(matrix)
    def expand(self, node):
        nodes = [] # New created nodes will be added here

        for action in self.directions:
            empty = (node.position[0] + action[0] , node.position[1] + action[1]) # position of empty tile in new matrix
            if empty[0] >= self.game_size or empty[1] >= self.game_size or empty[0] < 0 or empty[1] < 0: # checks if new empty tile is in matrix
                continue # if not do not add to list

            else:
                matrix = []
                for i in range(self.game_size): # this loop creates new matrix. changes empty tile to the new position
                    matrix.append([])
                    for j in range(self.game_size):
                        if (i,j) == empty:
                            matrix[i].append(0)
                        elif (i,j) == node.position:
                            matrix[i].append(node.matrix[empty[0]][empty[1]])
                        else:
                            matrix[i].append(node.matrix[i][j])

                nodes.append(Node(node, empty, matrix)) # creates new node and adds it to the list
                self.generated_node += 1

        return nodes # returns new nodes 
    def tree_solve(self):
        """
            Solves the game using tree base BFS algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """
        
        self.frontier = [Node(None, self.empty_tile,self.initial_matrix)] # frontier set
        
        self.generated_node += 1
        self.maximum_node_in_memory = 1

        found = False # true if solution found
        while self.frontier: # Continue until find the solution or frontier is empty
            node = self.frontier.pop(0) # select first added node to expand (FIFO)
            self.expanded_node += 1

            for child in self.expand(node): # expand the node
                if self.checkEqual(self.desired_matrix, child.matrix): # solution has found if child has desired matrix 
                    return self.get_moves(child)
                else:
                    self.frontier.append(child) # add child node to frontier to expand

                    self.maximum_node_in_memory = max(len(self.frontier), self.maximum_node_in_memory) 
                    
        return None # return the path
        ### YOUR CODE HERE ###
    def graph_solve(self):
        """
            Solves the game using graph base DFS algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """
        self.frontier = [Node(None, self.empty_tile,self.initial_matrix)] # create frontier set
        self.explored = [Node(None, self.empty_tile,self.initial_matrix)] # create explored set
        self.maximum_node_in_memory += 1
        self.generated_node += 1

        found = False # true iif solution has found
        while len(self.frontier) != 0 and not found: # continue until find the solution or frontier is empty
            node = self.frontier.pop() # select last added node to expand (LIFO)
            self.expanded_node += 1
            for child in self.expand(node): # expand child node
                if self.checkEqual(self.desired_matrix, child.matrix): # solution has found if node has desired matrix
                    self.maximum_node_in_memory += 1
                    found = True
                    break
                else:
                    seen = False
                    for exp in self.explored: # check if node already explored or not
                        if self.checkEqual(child.matrix, exp.matrix):
                            seen = True
                    if not seen: # if node has not been explored before add it to frontier and explored set
                        self.frontier.append(child)
                        self.explored.append(child)
                        self.maximum_node_in_memory += 1

        return self.get_moves(child) # return path

class AStarAgent(Agent):
    
    def __init__(self, matrix):
        """
            Initializes the A* agent class.

            Args:
                matrix (array): Initial game matrix
        """
        # Initializing the parent class
        super().__init__(matrix)

    def expand(self, node):
        nodes = []

        for action in self.directions:  
            empty = (node.position[0] + action[0] , node.position[1] + action[1]) # position of empty tile in new matrix
            if empty[0] >= self.game_size or empty[1] >= self.game_size or empty[0] < 0 or empty[1] < 0: # checks if new empty tile is in matrix
                continue # if not do not add to list

            else: # if new empty tile is in matrix
                matrix = []
                for i in range(self.game_size): # this loop creates new matrix. changes empty tile to the new position
                    matrix.append([])
                    for j in range(self.game_size):
                        if (i,j) == empty:
                            matrix[i].append(0)
                        elif (i,j) == node.position:
                            matrix[i].append(node.matrix[empty[0]][empty[1]])
                        else:
                            matrix[i].append(node.matrix[i][j])
                            
                h_score = self.h_function(matrix, self.desired_matrix) # calculates heuristic cost using given heuristic cost function.
                nodes.append(Node(node, empty, matrix, node.g_score + 1, h_score)) # creates new node and adds it to the list. updates g_score.
                self.generated_node += 1

        return nodes # returns new nodes
    
    def heuristic_cost_1(self, matrix, target):
        """
            Calculates number of wrong placed tiles.
        """ 
        cost = 0    

        for i in range(self.game_size): # traverse the matrices
            for j in range(self.game_size):
                if matrix[i][j] != target[i][j]: # update cost if not equal
                    cost += 1

        return cost # return cost

    def heuristic_cost_2(self, matrix, target):
        """
            calculates total manhattan distance between given matrixes.
        """
        cost = 0

        right_places = dict() # right places for every tile. key values are elements in tile and key values are right position.
        for i in range(self.game_size):
            for j in range(self.game_size):
                right_places[target[i][j]] = (i,j) 

        for i in range(self.game_size): # calculates manhattan distance for each tile and updates cost
            for j in range(self.game_size):
                if matrix[i][j] != 0:
                    location = right_places[matrix[i][j]]
                    cost += abs(location[0] - i) + abs(location[1] - j)

        return cost # returns cost

    def heuristic_cost_3(self, matrix, target):
        """
            calculates total euclidean distance between given matrixes.
        """
        cost = 0
        matrix_inidices = dict()
        for i in range(self.game_size):
            for j in range(self.game_size):
                matrix_inidices[matrix[i][j]] = np.array([i,j])
        
        for i in range(self.game_size):
            for j in range(self.game_size):
                cost += np.linalg.norm(matrix_inidices[target[i][j]] - [i,j])

        return cost
    def tree_solve(self):
        """
            Solves the game using tree base A* algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """
        self.h_function = self.heuristic_cost_3 # heuristic function that algorithm will use.
        h_value = self.h_function(self.initial_matrix, self.desired_matrix)
        initial_node = Node(None, self.empty_tile, self.initial_matrix, 0, h_value) # creates inital node
       
        self.frontier = PriorityQueue()
        self.frontier.push(initial_node, initial_node.f_score)

        self.generated_node = 1
        self.maximum_node_in_memory = 1

        min_path = Node(None,None,None, self.INFINITY, 0) # created for later comparison
        while not self.frontier.isEmpty():

            node = self.frontier.pop() # pick a node which has lowest f_score from priority queue
            if node.f_score >= min_path.f_score: # returns the "min_path" if the potential cost is greater than algorithm already find 
                break

            self.expanded_node += 1
            for child in self.expand(node): # expands node
                if self.checkEqual(self.desired_matrix, child.matrix): # update min path if desired matrix found and cost is lower than min_path's cost
                    if child.f_score < min_path.f_score:
                        min_path = child
                else:
                    self.frontier.push(child, child.f_score) # adds child to the frontier
                    if self.frontier.size() > self.maximum_node_in_memory: # update maximum number of nodes in memory
                        self.maximum_node_in_memory = self.frontier.size() 

        return self.get_moves(min_path) # return path
    
    def graph_solve(self):
        """
            Solves the game using graph base A* algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """
        """
            Solves the game using tree base A* algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """
        self.h_function = self.heuristic_cost_3 # heuristic function that algorithm will use.
        h_value = self.h_function(self.initial_matrix, self.desired_matrix)
        initial_node = Node(None, self.empty_tile, self.initial_matrix, 0, h_value) # creates inital node

        
        # create and initialize frontier and explored sets
        self.frontier = PriorityQueue()
        self.frontier.push(initial_node, initial_node.f_score)
        self.explored = PriorityQueue()
        self.explored.push(initial_node, initial_node.f_score)

        self.generated_node = 1
        self.maximum_node_in_memory = 1
        min_path = Node(None,None,None, self.INFINITY, 0)
        while not self.frontier.isEmpty():

            node = self.frontier.pop() # pick a node which has lowest f_score from priority queue
            if node.f_score >= min_path.f_score: # returns the "min_path" if the potential cost is greater than algorithm already find 
                break

            self.expanded_node += 1
            for child in self.expand(node): # expands node
                if self.checkEqual(self.desired_matrix, child.matrix): # update min path if desired matrix found and cost is lower than min_path's cost
                    if child.f_score < min_path.f_score:
                        self.maximum_node_in_memory += 1
                        min_path = child
                else:
                    if not self.explored.contains(child): # if child not in the explored set
                        self.frontier.push(child, child.f_score) # adds child to the frontier
                        self.explored.push(child, child.f_score) # adds child to the frontier
                        self.maximum_node_in_memory += 1

        return self.get_moves(min_path) # return path