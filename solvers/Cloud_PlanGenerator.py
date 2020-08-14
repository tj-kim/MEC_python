# Migration Plan Code goes here
import time
import itertools
import copy
from PlanGenerator import *

# Dijkstra's Algorithm
from Dijkstra_Graph import *
from SeqGreedy_PlanGenerator import *

class Cloud_PlanGenerator(SeqGreedy_PlanGenerator):
    """
    Generate migration plans with basic heuristic approach.
    """
    def __init__(self, users, servers, links, jobs, sim_params):
        
        # Store all relevant parameters within class
        super(Cloud_PlanGenerator,self).__init__(users=users, 
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
        refresh_rate = [0,0,0]
        refresh = False
        for j in range(len(jobs)):
            self.jobs[j].info_from_usr(self.users[j],refresh_rate,refresh)
        
        # Build components above
        self.build_mig_graph()

    def update_costs(self, j, node_bans, path_bans):
        """
        Update total cost matrix based on bans to system.
        This is in response to the node and link resource constraints during
        resource reservation stages
        """
        
        t_list = []
        
        # Eliminate valid node based on server (to and from)
        for node in node_bans:
            node_s, node_t = self.convert_node2st[j][node]
            t_list += [node_t]
            
            print("Struck Placement Resource (j,s,t):", j, node_s, node_t)
            
            self.valid_links[j][node,:] = 0
            self.valid_links[j][:,node] = 0
        
        
        # Eliminate All links - replace specific edge weights with zero
        for (start_node,end_node,path_idx) in path_bans:
            (s_start,t_start) = self.convert_node2st[j][start_node]
            (s_end,t_end) = self.convert_node2st[j][end_node]
            t_list += [t_end] 
            if self.valid_links[j][start_node,end_node] == 1:
                self.all_costs[j][(start_node,end_node,path_idx)] = 0
                print("Struck Link Resource (j,s1,s2,t,p):", j, s_start,s_end,t_end,path_idx)
                
        t_list.sort()
        # Eliminate all nodes that are not part of cloud
        num_no_cloud_svr = len(self.servers) - 1
        for s,t in itertools.product(range(num_no_cloud_svr), t_list):
            if t >= 0 and t < self.sim_params.time_steps:
                node = self.convert_st2node[j][(s,t)]
                print("ban_node:", node)
                self.valid_links[j][node,:] = 0
                self.valid_links[j][:,node] = 0
        
