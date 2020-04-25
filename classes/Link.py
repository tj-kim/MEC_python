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
        Inputs:
        servers  - servers class holding server location information
        num_link - number of links each weak server should have with others
        
        Attributes:
        valid_links = binary indication of which links exist in the system
        num_path = number of path available between any two servers
        lit_links = path indexed for 2 servers based on num_path
        
        """
        
        # Obtain distances between each server
        self.distances = self.calc_distance(servers)
        
        # invoke function for generating links
        self.valid_links = self.generate_links(servers, num_link)
                
        # invoke function for counting number of unique paths between each server
        
        # invoke function for indicating which links are lit for each unique path
        
        return
    
    """
    Initialization Functions
    """
        
    def link_resources(self, weak_range, strong_range, timesteps):
        """
        generate matrix to define resource capacity for each timestep
        
        Inputs:
        weak_range - range of server capacity (mbps) between weak servers
        strong_rang - range of server capacity (mbps) between strong servers
        
        Attributes:
        avail = number of available resources for each link at each timestep
        """
        
        # define resource capacity for each link
        
        # define capacity for each timestep
        
        return
    
    def calc_distance(self, servers):
        """
        Return S x S Matrix with distances between each server
        """
                
        # make list of locations
        locs = []
        x,y = np.zeros(len(servers)),np.zeros(len(servers))
        for i in range(len(servers)):
            x[i], y[i] = servers[i].locs[0], servers[i].locs[1]
            
        # Compute euclidean distance for every combination of servers        
        srv_dists = np.sqrt(np.square(x - x.reshape(-1,1)) + np.square(y - y.reshape(-1,1)))
        
        return srv_dists
        
    
    def generate_links(self, servers, num_link):
        """
        Generate SxS np array for which links exist in the system
        For now every link in the system is bi-directional
        """
        
        valid_links = np.zeros((len(servers),len(servers)))

        # Obtain server level for each index
        svr_lvls = np.zeros(len(servers))
        for i in range(len(servers)):
            svr_lvls[i] = servers[i].level
        
        # Generate links between all level 3 and 2 servers bidirectional
        lvl23_idx = np.where(svr_lvls>1)
        mesh = np.array(np.meshgrid(lvl23_idx, lvl23_idx))
        combinations = mesh.T.reshape(-1, 2)
        valid_links[combinations[:,0],combinations[:,1]] = 1
        
        # Generate links between level 2 and closest level 1 servers
        # Generate links between level 1 servers to other level 1 servers
        lvl2_idx = np.where(svr_lvls==2)[0]
        lvl1_idx = np.where(svr_lvls==1)[0]
        
        for i in lvl1_idx:
            # From level 1 to level 2 servers
            dists = self.get_distance(s1_idx=i,s2_idx=None)
            mask = np.zeros(len(servers))
            mask[lvl2_idx] = 1
            dists_lv2 = dists * mask
            min_idx = np.where(dists_lv2==np.min(dists_lv2[np.nonzero(dists_lv2)]))[0][0]
            valid_links[lvl1_idx,min_idx], valid_links[min_idx,lvl1_idx] = 1,1
            
            # From level 1 to level 1 servers
            mask = np.zeros(len(servers))
            mask[lvl1_idx] = 1
            dists_lvl1 = dists * mask
            
            # 4/24 RESUME CODING HERE
            
        
        
        
        
        # set identity to zero
        np.fill_diagonal(valid_links, 0)
        
        return valid_links
    
    """
    Utility Functions
    """
    
    def get_distance(self,s1_idx,s2_idx=None):
        """
        Returns distance between server s1 and server s2 (double)
        If only server s1 is inputted, returns distance to all other servers
        """
        
        if s2_idx is None:
            return self.distances[s1_idx]
        else:
            return self.distances[s1_idx][s2_idx]
        
