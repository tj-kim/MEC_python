import numpy as np
import math
import itertools

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
    
    def from_ILP(self, ILP_prob, solve_flag = True):
        """
        From decision variables h we want to replace the zero vectors of 
        self.mig_plan_dict with relevant values based on the decision vars
        
        Input: ILP_prob - an Optim_PlanGenerator object that already has been optimized
        """
        
        if solve_flag:
            ILP_prob.prob.solve()
        
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
        
        # 3. Reserve Resources From Resource Constraints
        for j in range(len(ILP_prob.jobs)):
            placement_rsrc = ILP_prob.jobs[j].placement_rsrc
            mig_rsrc = ILP_prob.jobs[j].migration_rsrc
            service_bw = ILP_prob.jobs[j].thruput_req
            
            plan_dict = self.mig_plan_dict[j]
            for t in range(self.sim_params.time_steps):
                # Reserve Placement Cost & Mig link cost
                source_svr = int(plan_dict['source_server'][t])
                dest_svr = int(plan_dict['dest_server'][t])
                mig_rate = plan_dict['mig_rate'][t]
                path_idx = int(plan_dict['mig_link_id'][t])
                
                if source_svr == dest_svr:
                    ILP_prob.resource_constraints.server_rsrc[source_svr,:,t] -= placement_rsrc
                else:
                    ILP_prob.resource_constraints.server_rsrc[source_svr,:,t] -= placement_rsrc
                    ILP_prob.resource_constraints.server_rsrc[dest_svr,:,t] -= placement_rsrc
                    
                    avail_link = ILP_prob.resource_constraints.link_rsrc[:,:,t]
                    mig_links = ILP_prob.links.get_subpath(source_svr,dest_svr,path_idx)
                    remain_link = avail_link - (mig_rsrc*mig_rate*mig_links)
                    
                    ILP_prob.resource_constraints.link_rsrc[:,:,t] = remain_link
                
                # Reserve Expected Service BAndwidth Cost
                avail_link = ILP_prob.resource_constraints.link_rsrc[:,:,t]
                exp_service = np.zeros((len(ILP_prob.servers),len(ILP_prob.servers)))
                
                for s_var in range(len(ILP_prob.servers)):
                    if s_var != source_svr:
                        avg_link = ILP_prob.links.get_avgpath(source_svr,s_var)
                        usr_job_flag = ILP_prob.users[j].server_prob[s_var,t]
                        expected_sbw = np.multiply(service_bw, avg_link)
                        exp_service += expected_sbw
                
                remain_link = avail_link - exp_service
                ILP_prob.resource_constraints.link_rsrc[:,:,t] = remain_link
                
        
        self.prob = ILP_prob
        self.service_path_selection()
        self.thruput_selection()
        return
    
    def from_seq_greedy(self,SG_prob):
        """
        From the migration plan problem, do the entire procedure of reserving resources and 
        generating final migration plan -- how will we do this for batch?
        """
        
        # Should add convert node informations

        self.prob = SG_prob
        
        # Loop through time and user to generate incoming plans
        for t in range(self.sim_params.time_steps):
            for j in range(len(self.prob.jobs)):
                
                refresh_flag = self.prob.jobs[j].refresh_flags[t]
                # 1. Check for user arrival time
                if refresh_flag == 1:
                    
                    # 0. Update user server prob
                    self.prob.users[j].update_voronoi_probs(time_passed = t)
                    
                    # Resource Reservation/Constraints Prep
                    node_bans = []
                    path_bans = []
                    approved_flag = False
                    
                    # Flag if this is the first batch for this job
                    fresh_plan = self.prob.jobs[j].fresh_plan
                    if fresh_plan:
                        self.prob.jobs[j].fresh_plan = False
                    
                    # Start Node and End Node of Mig plan
                    start_node1 = self.prob.jobs[j].refresh_start_nodes[t]
                    end_node1 = self.prob.jobs[j].refresh_end_nodes[t]
                    
                    # Edit server of start node based on current node
                    if t > self.prob.jobs[j].arrival_time:
                        start_node1 = (self.mig_plan_dict[j]["source_server"][t], start_node1[1])
                    
                    start_node = self.prob.convert_st2node[j][start_node1]
                    end_node = self.prob.convert_st2node[j][end_node1]
                    self.prob.calc_all_costs(j,start_node,end_node)
                    
                    while_idx = 0
                    while not approved_flag:
                        # print("usr:",j,"reserve:",while_idx,"t:",t)
                        while_idx += 1
                    
                        # 2. If user arrives, make plan
                        self.prob.obtain_minimum_cost_j(j,start_node,end_node)

                        node_num, link_num = self.prob.dijkstra_j(j=j,start_node=start_node,
                                                                     end_node=end_node)
                        
                        
                        # 3. Repeat resource reservation until no conflicts --> or reject job
                        node_bans, path_bans, approved_flag = self.prob.check_reserve_resource(j,
                                                              node_num,link_num,fresh_plan)
                        
                        # print("start_node:",start_node)
                        # print("end_node:",end_node)
                        # print("dijkstra:",node_num)
                        # print("node_bans",node_bans)
                        # print("path_bans",path_bans)
                        
                        # Update cost graph
                        if not approved_flag:
                            self.prob.update_costs(j, node_bans,path_bans)   
                            
                        # set_trace()
                    
                    # Extract plan and record to system
                    self.seq_greedy_plan_extract(node_orders=node_num, 
                                                 link_path_orders=link_num, 
                                                 job_num=j)

        self.service_path_selection()
        self.thruput_selection()
    
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
            (source_server, start_time) = self.prob.convert_node2st[job_num][node1]
            (dest_server, end_time) = self.prob.convert_node2st[job_num][node2]
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
    
        
    def service_path_selection(self):
        """
        take into a consideration the resources at each link at each timestep, and determine
        Inputs:
        links - a link instance of the simulation
        jobs - a list of job objects each with their job size 
        
        Updates migration plan to determine throughput of service at each instance
        """
        
        switch_latency = self.prob.links.switch_delay
        dist_latency = self.prob.links.dist_delay
        server_distances = self.prob.links.calc_distance(self.prob.servers)
        
        # Loop thru plan - pick service and calc latency for each ts
        for j,t in itertools.product(range(len(self.prob.jobs)),range(self.sim_params.time_steps)):
            usr_svr = int(self.mig_plan_dict[j]["user_voronoi"][t])
            job_svr = int(self.mig_plan_dict[j]["source_server"][t])
            
            if usr_svr != job_svr and job_svr != -1:
                # Calculate which path
                num_path = int(self.prob.links.num_path[job_svr,usr_svr])
                # select_path = np.random.randint(0,self.sim_params.num_path_limit)
                select_path = 0
                self.mig_plan_dict[j]['service_link_id'][t] = select_path
            
                # Calculate Latency
                service_path = self.prob.links.get_subpath(job_svr,usr_svr,select_path)
                num_switch = np.sum(np.sum(service_path,axis=1),axis=0)
                
                latency_dists = np.multiply(service_path,server_distances)
                num_dist = np.sum(np.sum(latency_dists,axis=1),axis=0)
               
                self.mig_plan_dict[j]['latency'][t] = switch_latency * num_switch + num_dist * dist_latency
            
            else:
                self.mig_plan_dict[j]["service_link_id"][t] = -1
                
    def thruput_selection(self):
        """
        After running service_path_selection() we can pick thruput of each job at each timestep
        """
        
        # Loop through each timestep 
        for t in range(self.sim_params.time_steps):
            
            service_thruput = np.zeros(self.prob.links.distances.shape)
            job_thruputs = []
            
            # Add all 
            for j in range(len(self.prob.jobs)):
                # Add to list job idx and thruput
                job_thruputs += [self.prob.jobs[j].thruput_req]
            
            approved_flag = False
            
            # Adjust throughputs for this timestep
            while not approved_flag:
                for j in range(len(self.prob.jobs)):
                    if self.mig_plan_dict[j]["service_link_id"][t] > -1:
                        usr_svr = int(self.mig_plan_dict[j]["user_voronoi"][t])
                        job_svr = int(self.mig_plan_dict[j]["source_server"][t])
                        path_id = int(self.mig_plan_dict[j]["service_link_id"][t])

                        service_links_j = self.prob.links.get_subpath(job_svr,usr_svr,path_id)
                        service_thruput += job_thruputs[j] * service_links_j

                remainder_link = self.prob.resource_constraints.link_rsrc[:,:,t] - service_thruput
                
                one_coor = zip(*np.where(remainder_link < 0))

                if len(list(one_coor)) == 0:
                    approved_flag = True
                    continue
                    
                struck_jobs = []
                
                for (s1,s2) in one_coor:
                    for j in range(len(self.prob.jobs)):
                        if self.mig_plan_dict[j]["service_link_id"][t] > -1:
                            usr_svr = int(self.mig_plan_dict[j]["user_voronoi"][t])
                            job_svr = int(self.mig_plan_dict[j]["source_server"][t])
                            path_id = int(self.mig_plan_dict[j]["service_link_id"][t])

                            service_links_j = self.prob.links.get_subpath(job_svr,usr_svr,path_id)

                            if service_links_j[s1,s2] > 0 and (j not in struck_jobs):
                                job_thruputs[j] *= 0.9
                                struck_jobs += [j]
                                
            # Record throughput for each job
            for j in range(len(self.prob.jobs)):
                thru_flag = (self.mig_plan_dict[j]["service_link_id"][t] > -1)
                self.mig_plan_dict[j]["service_thruput"][t] = thru_flag * job_thruputs[j]