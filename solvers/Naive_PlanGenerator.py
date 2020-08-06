# Migration Plan Code goes here
import time
import itertools
import copy
from PlanGenerator import *
from SeqGreedy_PlanGenerator import *

# Dijkstra's Algorithm
from Dijkstra_Graph import *

class Naive_PlanGenerator(SeqGreedy_PlanGenerator):
    """
    Generate migration plans with basic heuristic approach.
    """
    def __init__(self, users, servers, links, jobs, sim_params):
        
        # Store all relevant parameters within class
        super(Naive_PlanGenerator,self).__init__(users=users, 
                                                     servers=servers, 
                                                     links=links,
                                                     jobs=jobs,
                                                     sim_params=sim_params)
                
        # Components of subclass
        self.convert_node2st = None
        self.convert_st2node = None
        self.valid_links = None
        self.num_edges = None
        self.all_costs = {}
        self.edge_weights_min = {}
        self.edge_weights_path_idx = {}
        self.num_path_limit = sim_params.num_path_limit # limit paths between 2 servers 
        
        # Make max edge length 1
        self.sim_params.max_edge_length = 1
        
        # Turn off refresh for all jobs
        refresh_rate = [0,0,0]
        refresh = False
        for j in range(len(jobs)):
            self.jobs[j].info_from_usr(self.users[j],refresh_rate,refresh)
        
        # Build components above
        self.build_mig_graph()

    
    def build_mig_graph(self):
        """
        Build the conversion table for all users that maps nodes in each user's migration graph 
        to a specific timestep and server (for job to be placed at)
        
        Also use this to make valid links table (matrix w 1 for valid, 0 for no edge)
        
        TODO --> Add dead nodes for every single timestep 
        """
        
        self.convert_node2st = []
        self.convert_st2node = []
        self.valid_links = []
        self.num_edges = []
        
        for j in range(len(self.jobs)):
            
            # --------- Build Conversion Table ----------#
            conv_idx = 0
            active_time = self.jobs[j].active_time
            u_node2st = {} # Nested dictionary
            
            # Loop through each timestep to make node
            # Start node 
            u_node2st[conv_idx] = (-1,-1) # (s,t)
            conv_idx += 1
            
            # All Other Nodes
            for t in range(self.sim_params.time_steps):
                if active_time[t] > 0:
                    # Add dead node for every single timestep
                    u_node2st[conv_idx] = (-1,t) # (s,t)
                    conv_idx += 1
                    for s in range(len(self.servers)):
                        u_node2st[conv_idx] = (s,t)
                        conv_idx += 1
                        
                else:
                    u_node2st[conv_idx] = (-1,t) # (s,t)
                    conv_idx += 1
            
            # End Node
            u_node2st[conv_idx] = (-1, self.sim_params.time_steps) #(s,t)
            
            u_st2node = dict([(value, key) for key, value in u_node2st.items()]) 
            self.convert_st2node += [u_st2node]
            self.convert_node2st += [u_node2st]
            
            
            # --------- Build Transition Matrix ----------#
            u_valid_links = np.zeros((len(u_node2st),len(u_node2st)))
            u_num_edges = np.zeros((len(u_node2st),len(u_node2st)))
            
            # Loop through every single node in the dictionary
            for key in u_st2node.keys():         
                server, active_s, time = key[0], key[0]>-1 ,key[1]
                time_check = time >= 0 and time < self.sim_params.time_steps
                
                if time_check:
                    active = self.jobs[j].active_time[time]
                else:
                    active = 0
                
                
                source_node = u_st2node[key]
                
                if time_check and active and active_s: # Source is active
                    max_edge_length = self.sim_params.max_edge_length
                    max_time = max(time+1, min(self.sim_params.time_steps, time + max_edge_length))
                    
                    for t2 in range(time+1,max_time+1):
                        if (t2 < self.sim_params.time_steps) and (self.jobs[j].active_time[t2] > 0):
                            for s in range(len(self.servers)):
                                if s == server and t2 == time + 1:
                                    dest_node =u_st2node[(s,t2)]
                                    u_valid_links[source_node,dest_node] = 1
                                    u_num_edges[source_node,dest_node] = 1
                            # Add the dummy dead node at t+1
                            if t2 == time + 1:
                                dest_node = u_st2node[(-1,t2)]
                                u_valid_links[source_node,dest_node]=1
                                u_num_edges[source_node,dest_node] = 1
                        elif (t2 == self.sim_params.time_steps) or self.jobs[j].active_time[t2] == 0:
                            if t2 == time + 1:
                                s = -1
                                dest_node = u_st2node[(s,t2)]
                                u_valid_links[source_node,dest_node] = 1
                                u_num_edges[source_node,dest_node] = 1
                            break # Break the loop as all other nodes don't matter
                        
                    
                # Source is inactive
                elif (time_check and not active) or (time == -1): 
                    time2 = time + 1
                    if time2 >= self.sim_params.time_steps:
                        active_2 = 0
                    else:
                        active_2 = self.jobs[j].active_time[time2]
                    
                    if active_2:
                        for s in range(len(self.servers)):
                            dest_node = u_st2node[(s,time2)]
                            u_valid_links[source_node,dest_node] = 1
                            u_num_edges[source_node,dest_node] = 1
                            
                    # else: # Both source and destination are inactive
                    # Lead to inactive node every single time now
                    dest_node = u_st2node[(-1,time2)]
                    u_valid_links[source_node,dest_node] = 1
                    u_num_edges[source_node,dest_node] = 1

            self.valid_links += [u_valid_links]
            self.num_edges += [u_num_edges]
 