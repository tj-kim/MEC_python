import numpy as np
import itertools
from Graph import Graph 

class Link:
    """
    Link: used for representing the following regarding links between servers
        - Existence of link
        - Number of paths between two servers
        - Resource constraint of each link
        - How many resources have been reserved at each timestep
    """
    
    def __init__(self, servers, num_link, prob_link, lv_minmax, lv1_transmission = 1):
        """
        Inputs:
        servers  - servers class holding server location information
        num_link - number of links each weak server should have with others
        prob_link - pdf of each possible count in num_link
        lv_minmax - uniform distribution range for link capacity (mbps) based on server level
        
        Attributes:
        valid_links = binary indication of which links exist in the system
        num_path = number of path available between any two servers
        lit_links = path indexed for 2 servers based on num_path
        lv1_transmission - how many times lv1 paths can be traversed
        """
        
        # What level 2 server is closest to each level 1 server
        self.s1_s2_assoc = None
        
        # Obtain distances between each server
        self.distances = self.calc_distance(servers)
        
        # invoke function for generating links
        self.valid_links = self.generate_links(servers, num_link, prob_link)
        
        # Generate resource capacities for each link
        self.rsrc_avail = self.link_resources(servers, lv_minmax)
        
        # invoke function for indicating which links are lit for each unique path
        self.path_graph = Graph(self.valid_links)
        
        # Reduce the number of valid paths to follow link protocol
        self.trim_paths(servers, lv1_transmission)
        
        # Count number of paths between two servers
        self.num_path = self.count_paths()
        
    
    """
    Initialization Functions
    """
        
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
        
    
    def generate_links(self, servers, num_link, prob_link):
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
        
        # Associate each level 1 server to closest lvl 2 server
        lv1_lv2_assoc = {}
        
        for i in lvl1_idx:
            # From level 1 to level 2 servers
            dists = self.get_distance(s1_idx=i,s2_idx=None)
            mask = np.zeros(len(servers))
            mask[lvl2_idx] = 1
            dists_lv2 = dists * mask
            min_idx = np.where(dists_lv2==np.min(dists_lv2[np.nonzero(dists_lv2)]))[0][0]
            valid_links[i,min_idx], valid_links[min_idx,i] = 1,1
            lv1_lv2_assoc[i] = min_idx
        
        self.s1_s2_assoc = lv1_lv2_assoc
        
        for i in lvl1_idx:
            # From level 1 to level 1 servers
            mask = np.zeros(len(servers))
            mask[lvl1_idx] = 1
            dists_lvl1 = dists * mask
            
            # Draw number of links to be made and iteratively generate
            lvl1_xn = np.random.choice(num_link, 1, p=prob_link)[0]

            for j in range(lvl1_xn):
                # If too many links already just break
                curr_lv1_links = np.sum(valid_links[i,lvl1_idx])
                lv1_serv_count = len(lvl1_idx)
                
                if curr_lv1_links >= lvl1_xn:
                    break
                elif np.sum(dists_lvl1) <= 0:
                    break
                
                # Draw link for minimum and remove lv1 server from contention
                min_idx = np.where(dists_lvl1==np.min(dists_lvl1[np.nonzero(dists_lvl1)]))[0][0]
                
                # Only link level 1 servers if they are connected to same level 2 servers
                if lv1_lv2_assoc[i] == lv1_lv2_assoc[min_idx]:
                    valid_links[i,min_idx], valid_links[min_idx,i] = 1,1
                
                dists_lvl1[min_idx] = 0
                
        
        # set identity to zero
        np.fill_diagonal(valid_links, 0)
        
        return valid_links
    
    def link_resources(self, servers, lv_minmax):
        """
        Sets each link resource capacity for each timestep to use as constraint
        
        Inputs:
        servers - list of server objects
        lv_minmax has 3 rows
        lv1_minmax - uniform resource range (mbps) for lv1 link
        lv2_minmax - uniform resource range (mbps) for lv2 link
        lv3_minmax - uniform resource range (mbps) for lv3 link
        """
        
        # check if link capacity has already been created for server pair
        checklist = np.zeros(self.valid_links.shape)
        
        rsrc_avail = np.copy(self.valid_links)
        
        for s1, s2 in itertools.product(range(self.valid_links.shape[0]),
                                        range(self.valid_links.shape[1])):
            if checklist[s1,s2] == 0 and self.valid_links[s1,s2] > 0:

                checklist[s1,s2], checklist[s2,s1] = 1,1

                # Make resource capacity based on level
                max_lvl = np.max([servers[s1].level, servers[s2].level]) - 1
                resource_draw = np.random.uniform(low = lv_minmax[max_lvl,0],
                                                      high = lv_minmax[max_lvl,1],
                                                      size = None)

                # List resource capacity, same bidirectional
                rsrc_avail[s1,s2] = resource_draw
                rsrc_avail[s2,s1] = resource_draw
                    
        return rsrc_avail
    
    def count_paths(self):
        """
        Given the graph object holding dictionary of possible paths, count number
        """
        
        path_counts = np.zeros(self.valid_links.shape)
        
        for s1, s2 in itertools.product(range(self.valid_links.shape[0]),
                                        range(self.valid_links.shape[1])):
            if s1 != s2:
                path_counts[s1,s2] = len(self.path_graph.path_dict[(s1,s2)])
                
        return path_counts
    
    def trim_paths(self, servers, lv1_transmission):
        """
        Trim the number of paths the links can forward traffic from any 2 servers
        Protocol - Traffic from level 2 can only go to level 1 if its to destination
        """
        
        for s1 in range(len(servers)): 
            for s2 in range(len(servers)):
                if s1 != s2:
                    path_list = self.path_graph.path_dict[(s1,s2)]
                    new_path_list = []
                    for path in path_list:
                        curr_level = 1
                        break_flag = 0
                        lv1_counter = 0
                        for s in path:
                            if servers[s].level > curr_level: 
                                curr_level = servers[s].level
                            # Get rid of paths with too many connections between level 1 servers
                            elif servers[s].level == curr_level and curr_level == 1 and s != s1:
                                lv1_counter += 1
                                if lv1_counter == lv1_transmission + 1:
                                    break_flag = 1
                                    break
                            # Get rid of paths with back and forth between lv1 and lv2 servers
                            elif servers[s].level < curr_level and servers[s].level == 1 and s != s2:
                                break_flag = 1
                                break
                        
                        if break_flag != 1:
                            new_path_list += [path]
                    
                    self.path_graph.path_dict[(s1,s2)] = new_path_list
    
        
    
    
    """
    Utility Functions (Callable)
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
    
