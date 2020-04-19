import numpy as np

class User:
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
        level - hierarchy level of server (cloud, strong, weak)
        rand_locs - generate server using locations drawn from uniform locations
        locs - custom locations for servers (good for strong servers)
        """
        
        # Generate/assign server locs
        
        # Assign server level
        
        return
        
    def server_resources(self, weak_range, strong_range, timesteps):
        """
        generate matrix to define resource capacity for each timestep
        """
        
        # define resource capacity for each server based on level
        
        # define capacity for each timestep
        
        return