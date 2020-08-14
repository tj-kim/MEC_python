# Migration Plan Code goes here
import time
import itertools
import copy
from PlanGenerator import *

# Dijkstra's Algorithm
from Dijkstra_Graph import *
from SeqGreedy_PlanGenerator import *

class Myopic_PlanGenerator(SeqGreedy_PlanGenerator):
    """
    Generate migration plans with basic heuristic approach.
    """
    def __init__(self, users, servers, links, jobs, sim_params):
        
        # Store all relevant parameters within class
        super(Myopic_PlanGenerator,self).__init__(users=users, 
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
        
        # Custom - make max edge length 1
        self.sim_params.max_edge_length = 1
        
        # Alter all jobs to have refresh rate of 1
        refresh_rate = [1,1,1]
        refresh = True
        for j in range(len(jobs)):
            self.jobs[j].info_from_usr(self.users[j],refresh_rate,refresh)
        
        # Build components above
        self.build_mig_graph()
            
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
        if t > 0:
            t = t-1
        
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