import numpy as np

class Link:
    """
    Link: used for representing the following regarding links between servers
        - Existence of link
        - Number of paths between two servers
        - Resource constraint of each link
        - How many resources have been reserved at each timestep
    """
    
    def __init__(self, servers, num_link):
        """
        servers  - servers class holding server location information
        num_link - number of links each weak server should have with others
        """
        
        # invoke function for generating links
                
        # invoke function for counting number of unique paths between each server
        
        # invoke function for indicating which links are lit for each unique path
        
        return
        
    def link_resources(self, weak_range, strong_range, timesteps):
        """
        generate matrix to define resource capacity for each timestep
        weak_range - range of server capacity (mbps) between weak servers
        strong_rang - range of server capacity (mbps) between strong servers
        """
        
        # define resource capacity for each link
        
        # define capacity for each timestep
        
        return