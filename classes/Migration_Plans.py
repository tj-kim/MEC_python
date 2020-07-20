import numpy as np
import math

class Migration_Plans:
    """
    Migration_Plans: 
        - Collects all migration plans generated for an entire system
        - mig_plan_dict : 6 x time_steps np array with rows
            - timeslot, user_active_flag, usr_voronoi, source_svr, dest_svr, mig_rate
    """
    
    def __init__(self, users, jobs, sim_params):
        """
        users - list of user class (used to extract user location)
        time_steps - how many timesteps to simulate user movement for
        """
        
        self.mig_plan_dict = {}
        self.sim_params = sim_params
        
        # Initialize dictionary 
        self.dict_init(users,jobs,sim_params.time_steps)
    
    """
    Init Helper Function
    """
    
    def dict_init(self, users, jobs, time_steps):
        
        param_collection = ["time_slot", "user_active_flag", 
                            "user_voronoi", "source_server", 
                            "dest_server", "mig_rate",
                            "mig_link_id", "service_link_id",
                            "service_thruput", "latency"]
        
        for u in range(len(users)):
            temp_usr_dict = {}
            
            for p in param_collection:
                temp_usr_dict[p] = np.zeros(time_steps)
            
            # Record active time and user voronoi
            for t in range(int(time_steps)):
                temp_usr_dict["time_slot"][t] = t
                temp_usr_dict["user_voronoi"][t] = users[u].user_voronoi_true[t]
                temp_usr_dict["user_active_flag"][t] = jobs[u].active_time[t]
            
            self.mig_plan_dict[u] = temp_usr_dict
    
    """
    Extraction Functions
    """
    
    def from_ILP(self, ILP_prob):
        """
        From decision variables h we want to replace the zero vectors of 
        self.mig_plan_dict with relevant values based on the decision vars
        
        Input: ILP_prob - an Optim_PlanGenerator object that already has been optimized
        """
        
        # 1. Loop through all h_vars and obtain those that have been selected
        h_hit_keys = []
        
        for h_key in ILP_prob.h_vars.keys():
            if ILP_prob.h_vars[h_key].varValue>0:
                h_hit_keys += [h_key]
            
        
        # 2. Loop through each of the users and isolate variables that pertain to them
        for j in range(len(ILP_prob.jobs)):
            uh_keys = []
            
            # Collect all keys from a certain user
            for hit_key in h_hit_keys:
                if hit_key[0] == j: # If of the correct user
                    uh_keys += [hit_key]
            
            
            # Reorder the hit keys in terms of time
            uh_keys_ordered = []
            curr_time = -1 
            for l in range(len(uh_keys)):
                time_key = None
                for key in uh_keys:
                    if key[3] == curr_time:
                        break
                uh_keys_ordered += [key]
                curr_time = key[4]
                uh_keys.remove(key)
        
            self.ILP_plan_extract(uh_keys_ordered,j)
                    
        
        return
    
    def from_seq_greedy(self,SG_prob):
        """
        From the migration plan problem, do the entire procedure of reserving resources and 
        generating final migration plan -- how will we do this for batch?
        """
        
        # Should add convert node informations
        self.convert_node2st = SG_prob.convert_node2st
        self.convert_st2node = SG_prob.convert_st2node
        self.valid_links = SG_prob.valid_links
        self.num_edges = SG_prob.num_edges
        self.all_costs = SG_prob.all_costs
        self.edge_weights_min = SG_prob.edge_weights_min
        self.edge_weights_path_idx = SG_prob.edge_weights_path_idx
    
    """
    Extraction function helpers
    """
    
    # ILP
    
    def ILP_plan_extract(self, uh_keys_ordered, job_num):
        """
        Loop through the ordered selected keys for a single user 
        And generate np arrays that will correspond to plans
        Inputs:
            uh_keys_ordered: list of tupels that represent h-variables in ILP Solution
            job_num: job id 
        """
        
        active = True
        inactive = False
        
        # Loop through each of the keys (Use switch cases below)
        for uh_key in uh_keys_ordered:
            start_time, end_time = uh_key[3], uh_key[4]
            source_server,dest_server = uh_key[1], uh_key[2]
            link_path = uh_key[5]
            
            case = (source_server >= 0, dest_server >= 0) # server source, dest active/inactive
            
            if case == (active, active) or case == (inactive,inactive):
                self.mig_plan_dict[job_num]["source_server"][start_time:end_time] = source_server
                self.mig_plan_dict[job_num]["dest_server"][start_time:end_time] = dest_server
                
                # Migration length find
                if source_server != dest_server:
                    mig_length = end_time - start_time
                    self.mig_plan_dict[job_num]["mig_rate"][start_time:end_time] = 1/mig_length
                    self.mig_plan_dict[job_num]["mig_link_id"][start_time:end_time] = link_path
                    
            elif case == (inactive, active) or case == (active, inactive):
                # The entire column in the plan is considered inactive/active
                self.mig_plan_dict[job_num]["source_server"][start_time:end_time] = source_server
                self.mig_plan_dict[job_num]["dest_server"][start_time:end_time] = source_server
    
    # HEuristic
    
    def seq_greedy_plan_extract(self, node_orders, link_path_orders, job_num):
        """
        Loop through the ordered selected nodes for a single user 
        And generate np arrays that will correspond to plans
        Inputs:
            node_orders: Sequence of nodes in mig graph that make up a plan
            link_path_orders : which path was taken between two nodes in mig graph
            job_num: job id 
        """
        
        active = True
        inactive = False
        
        # Pull pairs of nodes that are connected together
        pair_list = []
        path_idx = 0
        for i in range(len(node_orders)-1):
            pair_list += [(node_orders[i],node_orders[i+1])]
        
        # Loop through each of the keys (Use switch cases below)
        for (node1,node2) in pair_list:
            (source_server, start_time) = self.convert_node2st[job_num][node1]
            (dest_server, end_time) = self.convert_node2st[job_num][node2]
            link_path = link_path_orders[path_idx]
            path_idx += 1
            
            case = (source_server >= 0, dest_server >= 0) # server source, dest active/inactive
            
            if case == (active, active) or case == (inactive,inactive):
                self.mig_plan_dict[job_num]["source_server"][start_time:end_time] = source_server
                self.mig_plan_dict[job_num]["dest_server"][start_time:end_time] = dest_server
                
                # Migration length find
                if source_server != dest_server:
                    mig_length = end_time - start_time
                    self.mig_plan_dict[job_num]["mig_rate"][start_time:end_time] = 1/mig_length
                    self.mig_plan_dict[job_num]["mig_link_id"][start_time:end_time] = link_path
                    
            elif case == (inactive, active) or case == (active, inactive):
                # The entire column in the plan is considered inactive/active
                self.mig_plan_dict[job_num]["source_server"][start_time:end_time] = source_server
                self.mig_plan_dict[job_num]["dest_server"][start_time:end_time] = source_server
    
    # General
        
    def reserve_service_bw(self,links,jobs):
        """
        take into a consideration the resources at each link at each timestep, and determine
        Inputs:
        links - a link instance of the simulation
        jobs - a list of job objects each with their job size 
        
        Updates migration plan to determine throughput of service at each instance
        """
        
        # Loop through each ts
        
        # Loop through each plan
        
        # 1. Select link randomly from available options
        
        # 2. If source/dest differ + active 
        
        # a. Loop through each of the links that is for this job
        
        # b. Loop through each of the active jobs that use this link
        
        # c. Find the bottleneck thruput based on proportions and reserve
        # we will have slight inefficiency due to sequential reserve system but it'll be redundant
        # across all users
        
        return