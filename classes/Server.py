import numpy as np

class Server:
    """
    Server: generates one server in space/time with following characteristics
        - Existence of link
        - Number of paths between two servers
        - Resource constraint of each link
        - How many resources have been reserved at each timestep
    """
    
    def __init__(self, boundaries, level, rand_locs = True, locs = None):
        """
        boundaries - x,y coordinates showing limit for where 
        level - hierarchy level of server (cloud = 3, strong = 2, weak = 1)
        rand_locs - generate server using locations drawn from uniform locations
        locs - custom locations for servers (good for strong servers)
        """
        
        # Attributes
        self.level = level
        self.num_rsrc = None
        self.avail_rsrc = None
        self.svr_id = None
        self.svr_rsrc_cost = None
        
        # Generate/assign server locs
        if rand_locs is True:
            self.locs = self.generate_locs(boundaries)
        else:
            self.locs = locs
        
    """
    Misc. Callable Functions
    """
        
    def server_resources(self, num_resource, weak_range, strong_range):
        """
        generate matrix to define resource capacity for each timestep
        
        Input:
            num_resource - number of resources at a server (storage, ram, cpu)
            weak_range - level 1 resources, num_resource x 2 matrix
            strong_range - level 2 resources, num_resource x 2 matrix
            timesteps - number of timesteps in the system
            
        Attribute: 
            avail_rsrc - available resources at server (single timestep)
        """
        
        max_range = 1e9 # Placeholder for infinite resource
        avail = np.ones(num_resource)
        
        # define resource capacity for each server based on level
        if self.level == 1:
            lvl_range = weak_range
        elif self.level == 2:
            lvl_range = strong_range
        else: # If server level is cloud=3
            self.avail_rsrc = avail * max_range
            return
        
        # Draw each resource type from random distribution
        for i in range(num_resource):
            resource_draw = np.random.uniform(low = lvl_range[i,0], high = lvl_range[i,1], size = None)
            avail[i] = avail[i] * resource_draw
        
        self.num_rsrc = num_resource
        self.avail_rsrc = avail
        
    def server_resources_cost(self,num_resource,cost_per_resource):
        """
        generate cost per placement resource at this specific server
        
        Input: num_resource (integer), number of types of resources
               cost_per_resource (num_resource x 1) vector with monetary 
                   cost per unit resource per TS
        """
        
        assert cost_per_resource.shape[0] == num_resource
        
        self.svr_rsrc_cost = cost_per_resource
        
    def assign_id(self, id_no):
        """
        Assign ID to server. Make sure it doesn't overlap with other servers
        """
        self.svr_id = id_no
    
    """
    Init helper functions (non-callable)
    """
    
    def generate_locs(self, boundaries):
        """
        Use uniform distribution to set server location 
        """
        
        x_min, x_max = boundaries[0,0], boundaries[0,1]
        y_min, y_max = boundaries[1,0], boundaries[1,1]
        
        locs = np.zeros(2)
        
        locs[0] = np.random.uniform(low = x_min, high = x_max, size = None)
        locs[1] = np.random.uniform(low = y_min, high = y_max, size = None)
        
        return locs
