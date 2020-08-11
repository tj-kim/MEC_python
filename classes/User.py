import numpy as np
import math

class User:
    """
    User: generates one user in space/time with following characteristics
        - Initial location, location at each timestep
        - User type (vehicle, pedestrian, public transport)
        - Markov chain
        - conditioning function
    """
    
    def __init__(self, boundaries, time_steps, mvmt_class, lambdas, max_speed, num_path = 1):
        """
        boundaries - x,y coordinates showing limit for where 
        time_steps - how many timesteps to simulate user movement for.
        mvmt_class - pedestrian, vehicle, or public transport (determines stochastic mvmt)
        lambdas - exponential distribution parameter for each mvmt class (list)
        numpath - number of random paths to simulate to make user markov chain
        """
        
        # Easy to store values
        self.num_path = num_path
        self.time_steps = time_steps
        self.mvmt_class = mvmt_class
        self.max_speed = max_speed
        self.lmda = lambdas[mvmt_class]
        self.num_servers = None
        self.user_id = None
        
        # Make user initial location
        init_loc = self.generate_locs(boundaries)
        
        # Draw future user location x numpath for travel
        self.all_paths = self.generate_all_paths(boundaries, init_loc, 
                                                 num_path, lambdas[mvmt_class], 
                                                 time_steps, max_speed)
        
        # Select a single path as true path of movement for user
        self.true_path_idx = np.random.randint(self.num_path,size=1)
        self.true_path = np.squeeze(self.all_paths[self.true_path_idx],axis=0)
        
        # User voronoi (All paths taken voronoi)
        self.user_voronoi = None
        self.user_voronoi_true = None
        self.MC_trans_matrix = None
        self.server_prob = None
        self.server_prob_true = None
        
    """
    Markov Chain Functions (Callable)
    """
    def generate_MC(self, servers):
        """
        Generate markov chain based on user movement patterns
        Take probabilistic conditioning on prior location to compute new location
        """
        
        # Assign closest server to each user location
        self.user_voronoi = self.find_closest_servs(servers)
        self.user_voronoi_true = np.squeeze(self.user_voronoi[self.true_path_idx],axis=0)
        self.num_servers = len(servers)
        
        self.server_prob_true = np.zeros((len(servers),self.time_steps))
        for t in range(self.user_voronoi_true.shape[0]):
            self.server_prob_true[int(self.user_voronoi_true[t]),t] = 1
        
        # Obtain transition probabilities based on user voronoi on paths
        self.MC_trans_matrix = self.generate_transition_matrix()
        self.update_voronoi_probs()

        
    def update_voronoi_probs(self, time_passed=0, self_rate = 0.05, raise_times = 1e7):
        """
        Generate probability of user being at each server at each timestep 
        based on Markov Chain
        Also update Markov chain based on where user is after certain 
        amount of times passed.
        
        Input:
        time_passed : amount of time passed in simulation/update by
        self_fate : In ergodic end node, self transition rate
        raise_times : Exponent of transition matrix to find mean settling prob
        """
        
        # Update Markov Chain based on user movement
        self.update_transition_matrix(time_passed)
        
        # Artificially make Markov Chain Ergodic (add end node and self loop)
        MC_ergodic = np.zeros((self.MC_trans_matrix.shape[0]+1,self.MC_trans_matrix.shape[1]+1))
        MC_ergodic[0:-1,0:-1] = np.copy(self.MC_trans_matrix)
        MC_ergodic[-1,-1] = self_rate
        MC_start_node = self.dict_st2node[(int(self.user_voronoi_true[time_passed]),time_passed)]
        MC_ergodic[-1,MC_start_node] = 1 - self_rate
        
        for s in self.user_voronoi[:,-1]:
            temp_node = self.dict_st2node[(int(s),self.time_steps-1)]
            MC_ergodic[temp_node,-1] = 1
            
        # Find stationary probabilities of ergodic markov chain
        stat_prob = np.linalg.matrix_power(MC_ergodic,int(raise_times))[0,:]
        
        # Find probability of user being at each server at each timestep
        server_prob = np.zeros((self.num_servers,self.time_steps))
        
        # Place probability of 1 for previous timesteps
        for t in range(time_passed + 1):
            visited_server = self.user_voronoi_true[t]
            server_prob[int(visited_server), t] = 1
        
        # Condition on stationary probability for future timesteps
        for t in range(time_passed + 1, self.time_steps):
            for s in range(self.num_servers):
                if (s,t) in self.dict_st2node.keys():
                    node_id = self.dict_st2node[(s,t)]
                    server_prob[s,t] = stat_prob[node_id]
            
            server_prob[:,t] = server_prob[:,t]/np.sum(server_prob[:,t]) 
        
        self.server_prob = server_prob
            
    """
    Misc. Callable Functions
    """
    def assign_id(self, id_no):
        """
        Assigns ID to user. 2 Users should not have the same IDs
        """
        
        self.user_id = id_no
        
        
    """
    Init helper Functions (Not Callable)
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

    def generate_all_paths(self, boundaries, init_loc, numpath, lmda, time_steps, max_speed):
        """
        Generate Random Movements for users starting at initial location
        """
        
        # Generate Random travel magnitude and direction from exponential distribution
        mags = np.random.exponential(1/lmda,size = (numpath, time_steps-1))
        mags[mags > max_speed] = max_speed
        angles = np.random.uniform(low = 0, high = 2 * math.pi, size = (numpath, time_steps-1))
        
        # Convert mag/angles to x,y displacements
        x_delta = np.expand_dims(np.multiply(mags, np.cos(angles)),axis=1)
        y_delta = np.expand_dims(np.multiply(mags, np.sin(angles)),axis=1)
        deltas = np.append(x_delta,y_delta,axis=1)
        
        # Add deltas to initial location while staying inside boundary
        locs = np.ones((self.num_path,2,time_steps)) * np.reshape(init_loc,(1,2,1))
        for t in np.arange(1,time_steps): # Offset first timestep (initloc)
            curr_locs = locs[:,:,t-1] + deltas[:,:,t-1]
            # Check if any of the boundaries are exceeded
            curr_locs = self.boundary_fix(curr_locs, boundaries)
            locs[:,:,t] = curr_locs
        
        return locs
    
    def boundary_fix(self, curr_locs,boundaries):
        """
        Shoves users to space boundary if they venture outside simulation space
        """
        
        x_min, x_max = boundaries[0,0], boundaries[0,1]
        y_min, y_max = boundaries[1,0], boundaries[1,1]
        
        x_vals = curr_locs[:,0]
        y_vals = curr_locs[:,1]
        
        x_vals[x_vals < x_min] = x_min
        x_vals[x_vals > x_max] = x_max
        y_vals[y_vals < y_min] = y_min
        y_vals[y_vals > y_max] = y_max
        
        output = np.append(np.expand_dims(x_vals,axis=1),
                           np.expand_dims(y_vals,axis=1),
                           axis=1)
        return output
        
    """
    Utility Functions for Markov CHain
    """
    def find_closest_servs(self, servers):
        """
        Find the closest server given all user locations through time
        servers - list of server objects
        """
        
        # Make array of server locations
        server_locs = np.zeros((len(servers),2))
        for i in range(len(servers)):
            curr_svr_locs = np.expand_dims(servers[i].locs,axis=0)
            server_locs[i,:] = curr_svr_locs
        
        # Make voronoi tesselation of user locations to servers
        user_voronoi = np.zeros((self.num_path,self.time_steps))
        for n in range(self.num_path):
            for t in range(self.time_steps):
                usr_loc = np.reshape(self.all_paths[n,:,t],(1,2))
                dist_2 = np.sum((server_locs - usr_loc)**2, axis=1)
                user_voronoi[n,t] =  np.argmin(dist_2)
                
        return user_voronoi
    
    def generate_transition_matrix(self):
        """
        Make transition matrix for user movement
        Inputs:
        - user_voronoi : user movement across all paths
        - node_count : number of nodes in the Markov Chain
        """
        
        # Dictionary transfers between server,timestep pairs to MC nodes
        self.dict_st2node = {}
        self.dict_node2st = {}
        
        node_count = 0
        for t in range(self.time_steps):
            for s in range(self.num_servers):
                if s in self.user_voronoi[:,t]:
                    self.dict_st2node[(int(s),int(t))] = node_count
                    self.dict_node2st[node_count] = (int(s),int(t))
                    node_count += 1
        
        trans_matrix = np.zeros((node_count,node_count))
        
        for t in range(self.time_steps-1):
            source_servers = np.unique(self.user_voronoi[:,t])
            for s in source_servers:
                s_idx = np.where(self.user_voronoi[:,t]==s)[0]
                dests = np.zeros(self.num_servers)
                for k in s_idx:
                    temp_dest = self.user_voronoi[k,t+1]
                    dests[int(temp_dest)] += 1/s_idx.shape[0]
                source_MC_node = self.dict_st2node[(s,t)]
                for j in range(dests.shape[0]):
                    if j in self.user_voronoi[:,t+1]:
                        dest_MC_node = self.dict_st2node[(j,t+1)]
                        trans_matrix[source_MC_node,dest_MC_node] = dests[j]
        
        return trans_matrix
    
    def update_transition_matrix(self,time_passed):
        """
        Update Markov Chain based on how the user has moved so far
        """
        
        # Obtain current timestep and server
        new_dict_st2node = {}
        new_dict_node2st = {}
        node_count = 0
        
        curr_serv = self.user_voronoi_true[time_passed]
        new_dict_st2node[(curr_serv,time_passed)] = node_count
        new_dict_node2st[node_count] = (curr_serv,time_passed)
        node_count += 1
        
        # Make new dictionary for new transition matrix
        for t in range(time_passed+1, self.time_steps):
            for s in range(self.num_servers):
                if s in self.user_voronoi[:,t]:
                    new_dict_st2node[(int(s),int(t))] = node_count
                    new_dict_node2st[node_count] = (int(s),int(t))
                    node_count += 1
        
        trans_matrix = np.zeros((node_count,node_count))
        
        # Update transition matrix based on old one
        for source_node in range(trans_matrix.shape[0]):
            source_s, source_t = new_dict_node2st[source_node]
            old_source_node = self.dict_st2node[(int(source_s),int(source_t))]
            for dest_node in range(trans_matrix.shape[1]):
                dest_s, dest_t = new_dict_node2st[dest_node]
                old_dest_node = self.dict_st2node[(dest_s, dest_t)]
                trans_matrix[source_node, dest_node] = self.MC_trans_matrix[old_source_node, old_dest_node]
                
        self.dict_st2node = new_dict_st2node
        self.dict_node2st = new_dict_node2st
        self.MC_trans_matrix = trans_matrix
        
class ONE_User(User):
    """
    Copy of user class that takes in ONE information instead
    """
# (boundaries, sim_param.time_steps, max_speed, num_path, num_path_orig, usr_info[30])]
        
    def __init__(self, boundaries, time_steps, max_speed, num_path, num_path_orig, one_sim_usr, mvmt_class):
        """
        time_steps - how many timesteps to simulate user movement for.
        numpath - number of random paths to simulate to make user markov chain
        
        """
        
        # Easy to store values
        self.num_path = num_path
        self.time_steps = time_steps
        self.num_servers = None
        self.user_id = None
        self.one_sim_usr = one_sim_usr
        self.mvmt_class = mvmt_class
        
        # Make user initial location
        init_loc = one_sim_usr[0,2:4]
        # Get average speed for lamda
        lambda_u = np.mean(one_sim_usr[:,4])        
        
        # Draw future user location x numpath for travel
        self.all_paths = self.generate_all_paths(boundaries, init_loc, 
                                                 num_path, 1/lambda_u, 
                                                 time_steps, max_speed)
        
        # Replace all_paths with num_path_orig
        real_path = one_sim_usr[:time_steps,2:4].T
        
        for i in range(num_path_orig):
            self.all_paths[i,:,:] = real_path
        
        # Select a single path as true path of movement for user
        self.true_path_idx = np.array([0])
        self.true_path = np.squeeze(self.all_paths[self.true_path_idx],axis=0)
        
        # User voronoi (All paths taken voronoi)
        self.user_voronoi = None
        self.user_voronoi_true = None
        self.MC_trans_matrix = None
        self.server_prob = None
        self.server_prob_true = None