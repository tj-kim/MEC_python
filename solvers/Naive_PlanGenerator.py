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
 
    def calc_all_costs(self,j,start_node,end_node):
        """
        For every edge and path variation, calculate the cost and store in
        dictionary
        """
        
        # Truncate valid links matrix based on start and end time
        temp_valid_links = copy.deepcopy(self.valid_links[j])
        (start_svr, start_time) = self.convert_node2st[j][start_node]
        (end_svr, end_time) = self.convert_node2st[j][end_node]
        
        
        # Collect all nodes that are outside timezone
        ban_nodes = []
        for node_id in range(temp_valid_links.shape[0]):
            curr_svr,curr_time = self.convert_node2st[j][node_id]
            if curr_time > end_time or curr_time < start_time:
                temp_valid_links[node_id,:] = 0
                temp_valid_links[:,node_id] = 0
        
        
        one_coor = zip(*np.where(temp_valid_links == 1))
        job_placement_rsrc = self.jobs[j].placement_rsrc
        job_migration_rsrc = self.jobs[j].migration_rsrc

        # Make Dictionary for job j key = [node1, node2, linkid], val = cost
        dict_n1n2 = {} 

        for (node1, node2) in one_coor:
            (server1, time1) = self.convert_node2st[j][node1]
            (server2, time2) = self.convert_node2st[j][node2]
            num_path = self.num_edges[j][(node1,node2)]
            
            if num_path > self.num_path_limit:
                num_path = self.num_path_limit

            (active1, active2) = (server1 > -1, server2 > -1)
            time_diff = time2-time1

            # Case 1 - Active to Active
            if active1 and active2:
                # Placement and Migration Cost
                placement_cost_s1 = np.multiply(np.dot(job_placement_rsrc,self.servers[server1].svr_rsrc_cost),time_diff)
                if server1 != server2:
                    placement_cost_s2 = np.dot(job_placement_rsrc,self.servers[server2].svr_rsrc_cost)*time_diff
                    migration_cost = []
                    for n in range(int(num_path)):
                        num_path_links = self.links.get_subpath(server1,server2,n)
                        path_mig_cost = np.multiply(self.links.cost_links, num_path_links)
                        migration_cost += [job_migration_rsrc * np.sum(np.sum(path_mig_cost,axis=1),axis=0)]
                else:
                    placement_cost_s2 = 0
                    migration_cost = [0]

                # Service BW Cost - Expectation of user loc
                service_bw_cost = 0
                curr_latency = 0
                latency_cost = 0

                latency_list = []
                for t in range(time1+1,time2): # Offset as we already calc current timestep in prev edge
                    temp_sbw, temp_cL = self.service_latency_cost(j,server1,t)
                    latency_list += [temp_cL]
                    service_bw_cost += temp_sbw

                temp_sbw, temp_cL = self.service_latency_cost(j,server2,time2)
                service_bw_cost += temp_sbw
                latency_list += [temp_cL]


                for t in range(time_diff):
                    leftover_latency = latency_list[t] - self.jobs[j].latency_req
                    if leftover_latency > 0:
                        latency_cost += self.jobs[j].latency_penalty * leftover_latency


                # Record cost
                cost_etc = latency_cost + 0.01
                for n in range(int(num_path)):
                    dict_n1n2[(node1,node2,n)] = cost_etc

            # Case 2 - Inactive to Active
            elif (not active1) and active2:
                placement_cost = np.dot(job_placement_rsrc, self.servers[server2].svr_rsrc_cost) 

                # Service BW Cost - Expectation of user loc
                service_bw_cost, curr_latency = self.service_latency_cost(j,server2,time2)

                leftover_latency = curr_latency - self.jobs[j].latency_req
                latency_cost = 0
                if leftover_latency > 0:
                    latency_cost = self.jobs[j].latency_penalty * leftover_latency

                cost = latency_cost + 0.01
                dict_n1n2[(node1,node2,0)] = cost

            # Case 3 - Inactive to Inactive
            elif (not active1) and (not active2):
                cost = 0.01
                dict_n1n2[(node1,node2,0)] = cost

            # Case 4 - Active to Inactive
            elif active1 and (not active2):
                cost = 0.01
                dict_n1n2[(node1,node2,0)] = cost

            self.all_costs[j] = dict_n1n2
            
            
     # Subcost helper for latency and service bw
    def service_latency_cost(self,j,server,t):

        # Skew the latency calc process to consider previous ts server
        t = 0
        
        service_bw_cost = 0
        curr_latency = 0
        for s_var in range(len(self.servers)):
            if s_var != server:
                avg_link = self.links.get_avgpath(server,s_var)

                usr_job_flag = self.users[j].server_prob_true[s_var,t]
                expected_link_cost = np.multiply(self.links.cost_links, avg_link)
                total_link_cost = np.sum(np.sum(expected_link_cost,axis=1),axis=0)
                service_bw_cost += self.jobs[j].thruput_req * usr_job_flag * total_link_cost

                for s3, s4 in itertools.product(range(len(self.servers)),range(len(self.servers))):
                    delay = self.links.switch_delay + self.links.dist_delay * self.links.get_distance(s3,s4)
                    curr_latency += avg_link[s3,s4] * delay *usr_job_flag

        return service_bw_cost, curr_latency