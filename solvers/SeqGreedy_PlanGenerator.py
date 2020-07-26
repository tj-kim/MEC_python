# Migration Plan Code goes here
import time
import itertools
from PlanGenerator import *

# Dijkstra's Algorithm
from Dijkstra_Graph import *

class SeqGreedy_PlanGenerator(PlanGenerator):
    """
    Generate migration plans with basic heuristic approach.
    """
    def __init__(self, users, servers, links, jobs, sim_params):
        
        # Store all relevant parameters within class
        super(SeqGreedy_PlanGenerator,self).__init__(users=users, 
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
        
        # Build components above
        self.build_mig_graph()
        
        # self.calc_all_costs()
        # self.obtain_minimum_cost()
    
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
                                if s != server:
                                    dest_node = u_st2node[(s,t2)]
                                    u_valid_links[source_node,dest_node] = 1
                                    u_num_edges[source_node,dest_node] = self.links.num_path[server,s]
                                elif s == server and t2 == time + 1:
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
            
    def calc_all_costs(self,j):
        """
        For every edge and path variation, calculate the cost and store in
        dictionary
        """
        
        one_coor = zip(*np.where(self.valid_links[j] == 1))
        job_placement_rsrc = self.jobs[j].placement_rsrc
        job_migration_rsrc = self.jobs[j].migration_rsrc

        # Make Dictionary for job j key = [node1, node2, linkid], val = cost
        dict_n1n2 = {} 

        for (node1, node2) in one_coor:
            (server1, time1) = self.convert_node2st[j][node1]
            (server2, time2) = self.convert_node2st[j][node2]
            num_path = self.num_edges[j][(node1,node2)]

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
                cost_etc = placement_cost_s1 + placement_cost_s2 + service_bw_cost + latency_cost
                for n in range(int(num_path)):
                    dict_n1n2[(node1,node2,n)] = cost_etc + migration_cost[n]

            # Case 2 - Inactive to Active
            elif (not active1) and active2:
                placement_cost = np.dot(job_placement_rsrc, self.servers[server2].svr_rsrc_cost) 

                # Service BW Cost - Expectation of user loc
                service_bw_cost, curr_latency = self.service_latency_cost(j,server2,time2)

                leftover_latency = curr_latency - self.jobs[j].latency_req
                latency_cost = 0
                if leftover_latency > 0:
                    latency_cost = self.jobs[j].latency_penalty * leftover_latency

                cost = placement_cost + service_bw_cost + latency_cost
                dict_n1n2[(node1,node2,0)] = cost

            # Case 3 - Inactive to Inactive
            elif (not active1) and (not active2):
                cost = 1
                dict_n1n2[(node1,node2,0)] = cost

            # Case 4 - Active to Inactive
            elif active1 and (not active2):
                cost = 1
                dict_n1n2[(node1,node2,0)] = cost

            self.all_costs[j] = dict_n1n2
        
    # Subcost helper for latency and service bw
    def service_latency_cost(self,j,server,t):

        service_bw_cost = 0
        curr_latency = 0
        for s_var in range(len(self.servers)):
            if s_var != server:
                avg_link = self.links.get_avgpath(server,s_var)

                usr_job_flag = self.users[j].server_prob[s_var,t]
                expected_link_cost = np.multiply(self.links.cost_links, avg_link)
                total_link_cost = np.sum(np.sum(expected_link_cost,axis=1),axis=0)
                service_bw_cost += self.jobs[j].thruput_req * usr_job_flag * total_link_cost

                for s3, s4 in itertools.product(range(len(self.servers)),range(len(self.servers))):
                    delay = self.links.switch_delay + self.links.dist_delay * self.links.get_distance(s3,s4)
                    curr_latency += avg_link[s3,s4] * delay *usr_job_flag

        return service_bw_cost, curr_latency
            

    def obtain_minimum_cost_j(self,j):
        """
        For a specific user u
        """
        output_weight = np.zeros(self.valid_links[j].shape)
        output_path_idx = np.zeros(self.valid_links[j].shape)
        cost_dict = self.all_costs[j]
        
        one_coor = zip(*np.where(self.valid_links[j] == 1))
        
        for node1,node2 in one_coor:
            num_path = self.num_edges[j][(node1,node2)]
            edges_list = []
            
            # Obtain all costs in an array
            for n in range(int(num_path)):
                edges_list += [cost_dict[(node1,node2,n)]]
            
            # Min value
            min_val = min(i for i in edges_list if i > 0)
            # Min index
            idx = [i for i, j in enumerate(edges_list) if j == min_val][0]
            
            # obtain minimum link for specific link taken between two nodes
            output_weight[node1,node2] = min_val
            output_path_idx[node1,node2] = idx
            
        
        self.edge_weights_min[j] = output_weight
        self.edge_weights_path_idx[j] = output_path_idx
    
    def dijkstra_j(self,j,start_node,end_node):
        """
        Find shortest point between start and end node 
        """
        
        edge_weights = self.edge_weights_min[j]
        d_graph = Dijkstra_Graph()
        
        one_coor = zip(*np.where(self.valid_links[j] == 1))
        
        start_node = self.convert_st2node[j][(-1,-1)]
        end_node = self.convert_st2node[j][(-1,self.sim_params.time_steps)]
        
        for node1,node2 in one_coor:
            weight = edge_weights[node1,node2]
            d_graph.add_edge(node1,node2,weight)
        
        shortest_path = dijsktra(d_graph,start_node,end_node)
        shortest_path_link_idx = []
        
        for i in range(len(shortest_path)-1):
            p1, p2 = shortest_path[i],shortest_path[i+1]
            shortest_path_link_idx += [int(self.edge_weights_path_idx[j][p1,p2])]
        
        return shortest_path, shortest_path_link_idx
    
    def check_reserve_resource(self,j,shortest_path,shortest_path_link_idx):
        """
        Look through system resources given edge sequence and return:
        - (s,t) (node) combinations we should strike out from valid links
        - (node1, node2, numpath) specific path numbers we should zero out in all cost
        - Flag whether or not plan has been reserved or not
        """
        
        node_bans = []
        path_bans = []
        plan_reserved = False
        
        # Get job and mig sizes
        placement_rsrc = self.jobs[j].placement_rsrc
        mig_rsrc = self.jobs[j].migration_rsrc
        service_bw = self.jobs[j].thruput_req
        
        for i in range(len(shortest_path)-1):
            start_node = shortest_path[i]
            end_node = shortest_path[i+1]
            path_idx = shortest_path_link_idx[i]
            
            (server1, time1) = self.convert_node2st[j][start_node]
            (server2, time2) = self.convert_node2st[j][end_node]
            
            s1_active, s2_active = server1 >-1, server2 >-1
            
            valid_times = np.arange(time1,time2)
            
            # 1. Check server resources for all timesteps
            if server1 == server2 and s1_active:
                for t in valid_times:
                    # Server Check
                    avail_rsrc_s2 = self.resource_constraints.server_rsrc[server1,:,t] # 1d shape
                    for sr in range(avail_rsrc_s2.shape[0]):
                        if (avail_rsrc_s2[sr]-placement_rsrc[sr] < 0) and (start_node not in node_bans):
                            node_bans += [start_node]
                
            elif server1 != server2 and s1_active and s2_active:
                
                for t in valid_times:
                    avail_rsrc_s1 = self.resource_constraints.server_rsrc[server1,:,t]
                    avail_rsrc_s2 = self.resource_constraints.server_rsrc[server2,:,t] # 1d shape
                    
                    for sr in range(avail_rsrc_s1.shape[0]):
                        if (avail_rsrc_s1[sr]-placement_rsrc[sr] < 0) and (start_node not in node_bans):
                            node_bans += [start_node]
                    
                    for sr in range(avail_rsrc_s2.shape[0]):
                        if (avail_rsrc_s2[sr]-placement_rsrc[sr] < 0) and (end_node not in node_bans):
                            node_bans += [end_node]
                
            elif (not s1_active) and s2_active:
                continue 
                
            elif s1_active and (not s2_active):
                for t in valid_times:
                    avail_rsrc_s2 = self.resource_constraints.server_rsrc[server1,:,t] # 1d shape
                    for sr in range(avail_rsrc_s2.shape[0]):
                        if (avail_rsrc_s2[sr]-placement_rsrc[sr] < 0) and (start_node not in node_bans):
                            node_bans += [start_node] 
                
            elif server1 == server2 and (not s1_active):
                continue # Go to next node, no resources here
            
            
            # Set migration rate if migration
            mig_rate = 0
            exp_service = np.zeros((len(self.servers),len(self.servers)))
            
            
            # 2. Check Link Resources for all timesteps            
            if server1 == server2 and s1_active:
                for t in valid_times:
                    
                    avail_link = self.resource_constraints.link_rsrc[:,:,t]
                    
                    for s_var in range(len(self.servers)):
                        if s_var != server2:
                            avg_link = self.links.get_avgpath(server1,s_var)
                            usr_job_flag = self.users[j].server_prob[s_var,t]
                            expected_sbw = np.multiply(service_bw, avg_link)
                            exp_service += expected_sbw
                    
                    remain_link = avail_link - exp_service
                    check = np.all(remain_link >= 0)
                    
                    if (not check) and ((start_node,end_node,path_idx) not in path_bans):
                        path_bans += [(start_node,end_node,path_idx)]   

                
            elif server1 != server2 and s1_active and s2_active:
                mig_rate = 1/(time2-time1)
                mig_amt = mig_rsrc * mig_rate
                
                mig_links = self.links.get_subpath(server1,server2,path_idx)
                                   
                for t in valid_times:
                    avail_link = self.resource_constraints.link_rsrc[:,:,t]
                    
                    for s_var in range(len(self.servers)):
                        if s_var != server1:
                            avg_link = self.links.get_avgpath(server1,s_var)
                            usr_job_flag = self.users[j].server_prob[s_var,t]
                            expected_sbw = np.multiply(service_bw, avg_link)
                            exp_service += expected_sbw
                    
                    remain_link = avail_link - (mig_amt*mig_links + exp_service)
                    check = np.all(remain_link >= 0)

                    if (not check) and ((start_node,end_node,path_idx) not in path_bans):
                        path_bans += [(start_node,end_node,path_idx)]                   
                
            elif (not s1_active) and s2_active:
                continue   
                
            elif s1_active and (not s2_active):
                for t in valid_times:
                    
                    avail_link = self.resource_constraints.link_rsrc[:,:,t]
                    
                    for s_var in range(len(self.servers)):
                        if s_var != server1:
                            avg_link = self.links.get_avgpath(server1,s_var)
                            usr_job_flag = self.users[j].server_prob[s_var,t]
                            expected_sbw = np.multiply(service_bw, avg_link)
                            exp_service += expected_sbw
                    
                    remain_link = avail_link - exp_service
                    check = np.all(remain_link >= 0)
                    
                    if (not check) and ((start_node,end_node,path_idx) not in path_bans):
                        path_bans += [(start_node,end_node,path_idx)] 
                
            elif server1 == server2 and (not s1_active):
                continue # Go to next node, no resources here               
                                     

        # REserve resources and return    
        if (len(node_bans),len(path_bans)) == (0,0):
            self.reserve_resources(j,shortest_path,shortest_path_link_idx)
            plan_reserved = True
            
        return node_bans, path_bans, plan_reserved
    
    def reserve_resources(self,j,shortest_path,shortest_path_link_idx):
        """
        Subtract resource reservations from existing resources
        """
        
        # Get job and mig sizes
        placement_rsrc = self.jobs[j].placement_rsrc
        mig_rsrc = self.jobs[j].migration_rsrc
        service_bw = self.jobs[j].thruput_req
        
        for i in range(len(shortest_path)-1):
            start_node = shortest_path[i]
            end_node = shortest_path[i+1]
            path_idx = shortest_path_link_idx[i]
            
            (server1, time1) = self.convert_node2st[j][start_node]
            (server2, time2) = self.convert_node2st[j][end_node]
            
            s1_active, s2_active = server1 >-1, server2 >-1
            
            valid_times = np.arange(time1,time2)
                        
            # 1. Check server resources for all timesteps
            if server1 == server2 and s1_active:
                for t in valid_times:
                    self.resource_constraints.server_rsrc[server1,:,t] -= placement_rsrc# 1d shape
                
            elif server1 != server2 and s1_active and s2_active:                
                for t in valid_times:
                    self.resource_constraints.server_rsrc[server1,:,t] -= placement_rsrc
                    self.resource_constraints.server_rsrc[server2,:,t] -= placement_rsrc
                
            elif (not s1_active) and s2_active:
                for t in valid_times:
                    self.resource_constraints.server_rsrc[server2,:,t] -= placement_rsrc
          

            # 2. Check Link Resources for all timesteps
            # Set migration rate if migration
            mig_rate = 0
            exp_service = np.zeros((len(self.servers),len(self.servers)))
            
            
            # 2. Check Link Resources for all timesteps            
            if server1 == server2 and s1_active:
                for t in valid_times:
                    
                    avail_link = self.resource_constraints.link_rsrc[:,:,t]
                    
                    for s_var in range(len(self.servers)):
                        if s_var != server2:
                            avg_link = self.links.get_avgpath(server1,s_var)
                            usr_job_flag = self.users[j].server_prob[s_var,t]
                            expected_sbw = np.multiply(service_bw, avg_link)
                            exp_service += expected_sbw
                    
                    remain_link = avail_link - exp_service
                    self.resource_constraints.link_rsrc[:,:,t] = remain_link 

                
            elif server1 != server2 and s1_active and s2_active:
                mig_rate = 1/(time2-time1)
                mig_amt = mig_rsrc * mig_rate
                
                mig_links = self.links.get_subpath(server1,server2,path_idx)
                                   
                for t in valid_times:
                    avail_link = self.resource_constraints.link_rsrc[:,:,t]
                    
                    for s_var in range(len(self.servers)):
                        if s_var != server1:
                            avg_link = self.links.get_avgpath(server1,s_var)
                            usr_job_flag = self.users[j].server_prob[s_var,t]
                            expected_sbw = np.multiply(service_bw, avg_link)
                            exp_service += expected_sbw
                    
                    remain_link = avail_link - (mig_amt*mig_links + exp_service)
                    self.resource_constraints.link_rsrc[:,:,t] = remain_link             
                
            elif (not s1_active) and s2_active:
                continue   
                
            elif s1_active and (not s2_active):
                for t in valid_times:
                    
                    avail_link = self.resource_constraints.link_rsrc[:,:,t]
                    
                    for s_var in range(len(self.servers)):
                        if s_var != server1:
                            avg_link = self.links.get_avgpath(server1,s_var)
                            usr_job_flag = self.users[j].server_prob[s_var,t]
                            expected_sbw = np.multiply(service_bw, avg_link)
                            exp_service += expected_sbw
                    
                    remain_link = avail_link - exp_service
                    self.resource_constraints.link_rsrc[:,:,t] = remain_link
                    
            elif server1 == server2 and (not s1_active):
                continue
                
    def update_costs(self, j, node_bans, path_bans):
        """
        Update total cost matrix based on bans to system.
        This is in response to the node and link resource constraints during
        resource reservation stages
        """
        
        # Eliminate valid node based on server (to and from)
        for node in node_bans:
            self.valid_links[j][node,:] = 0
            self.valid_links[j][:,node] = 0
        
        # Eliminate All links - replace specific edge weights with zero
        for (start_node,end_node,path_idx) in path_bans:
            if self.valid_links[j][start_node,end_node] == 1:
                self.all_costs[j][(start_node,end_node,path_idx)] = 0