from collections import defaultdict 
import itertools

class Graph:
    """
    Given binary 2D transition matrix, give functionality for finding unique paths between 2 vertices
    Source: https://www.geeksforgeeks.org/find-paths-given-source-destination/
    
    Used for finding link paths between any two servers 
    """
    def __init__(self, adj_matrix):
        """
        Input - adj_matrix : adjacency matrix with links lit binary (1/0) np.ndarray
        """
        
        #No. of vertices 
        self.V= adj_matrix.shape[0]
        
        # default dictionary to store graph 
        self.graph = defaultdict(list) 
        
        # Add all of edges in adj_matrix to graph
        self.addEdges(adj_matrix)
        
        # Add dictionary to hold paths of each s1,s2 index
        self.path_dict = {}
        
        # Find unique paths and place in dictionary
        self.getAllPaths(adj_matrix)
    
    
    def addEdges(self,adj_matrix):
        """
        Take all edges of the adj_matrix and add them to graph attribute
        """
        for s1 in range(adj_matrix.shape[0]):
            for s2 in range(adj_matrix.shape[1]):
                if adj_matrix[s1,s2] > 0:
                    self.addEdge(s1,s2)
                
    
    def addEdge(self,u,v): 
        """
        # function to add an edge to graph 
        """
        self.graph[u].append(v) 
    
    
    def getPathsUtil(self, u, d, visited, path): 
  
        # Mark the current node as visited and store in path 
        visited[u]= True
        path.append(u) 
  
        # If current vertex is same as destination, then print 
        # current path[] 
        if u == d: 
            # print(path)
            if path not in self.path_dict[self.curr_key]:
                self.path_dict[self.curr_key].append(path.copy())
        else: 
            # If current vertex is not destination 
            #Recur for all the vertices adjacent to this vertex 
            for i in self.graph[u]: 
                if visited[i]==False: 
                    self.getPathsUtil(i, d, visited, path) 
                      
        # Remove current vertex from path[] and mark it as unvisited 
        path.pop() 
        visited[u]= False
        
   
    # Prints all paths from 's' to 'd' 
    def getPaths(self,s, d): 
    
        # Mark all the vertices as not visited 
        visited =[False]*(self.V) 
  
        # Create an array to store paths 
        path = [] 
  
        # Call the recursive helper function to print all paths 
        self.getPathsUtil(s, d,visited, path) 
        
    def getAllPaths(self, adj_matrix):
        """
        For every source, destination server combination, write in dictionary all possible paths
        """
        for s1,s2 in itertools.product(range(adj_matrix.shape[0]),range(adj_matrix.shape[1])):
            if s1 != s2:
                self.curr_key = (s1,s2)
                self.path_dict[self.curr_key] = []
                self.getPaths(s1,s2)