from pulp import *
from PlanGenerator import *
import time

class Optim_PlanGenerator(PlanGenerator):
    """
    Generate migration plans with optimization approach.
    """
    def __init__(self, users, servers, links, jobs, sim_params):
        
        # Store all relevant parameters within class
        super().__init__(users=users, servers=servers, links=links, jobs=jobs, sim_params=sim_params)
        
        # Components of subclass
        self.h_vars = None    # Decision var - traverse what paths on migration graph
        self.max_vars = None  # Decision var - proxy var to get maximum function 
        self.q_vars = None
        self.g_vars = None
        self.h_vars = None
        
        # Measure Run time
        self.run_time = None
        
        # Declare Optimization Problem
        self.prob = LpProblem("Migration Plan Problem",LpMinimize)
        
        # Build Linear Optimization Problem
        self.opt_decision_var()
        self.opt_auxiliary_vars()
        self.opt_feasibility_constraints()
        self.opt_resource_constraints()
        self.opt_objective_function()
    
    """
    Build optimization problem 
    """
    def opt_decision_var(self):
        """
        Write dictionary decision variable for variable "h"
        """        
        # Keys are tuples (job id, s1, s2, t1, t2, path_no)
        idxs = []
        
        # Start End Node settings
        start_node_time = -1
        end_node_time = self.sim_params.time_steps

        # Loop through all possible combinations of h variables outside of start and end node
        for j in range(len(self.jobs)):
            
            # Add start node
            first_time_active = (self.jobs[j].active_time[0] == 1)
            if first_time_active:
                for s1 in range(len(self.servers)):
                    idxs += [(j, -1, s1, start_node_time, 0, 0)]
            else:
                idxs += [(j, -1, -1, start_node_time, 0, 0)]
                
            # All other nodes
            for t1 in range(self.sim_params.time_steps):
                # Limit how far edge can go
                end_steps = min(t1+1+self.sim_params.max_edge_length, self.sim_params.time_steps)
                t1_active = (self.jobs[j].active_time[t1] == 1)
                
                for t2 in range(t1+1,end_steps):    
                    t2_active = (self.jobs[j].active_time[t2] == 1)
                    ts_one = (t2 == t1+1)
                    
                    # Case 1 - Active to Active node
                    if t1_active and t2_active:
                        for s1 in range(len(self.servers)):
                            for s2 in range(len(self.servers)):
                                if s1 != s2:
                                    for p in range(int(self.links.num_path[s1,s2])):
                                        idxs += [(j, s1, s2, t1, t2, p)]
                                elif ts_one:
                                    idxs += [(j, s1, s2, t1, t2, 0)]

                    # Case 2 - Inactive to Inactive Node
                    elif not t1_active and not t2_active and ts_one:
                        idxs += [(j, -1, -1, t1, t2, 0)]

                    # Case 3 - Inactive to Active Node
                    elif not t1_active and t2_active and ts_one:
                        for s2 in range(len(self.servers)):
                            idxs += [(j, -1, s2, t1, t2, 0)]

                # Case 4 - Active to Inactive Node
                    elif t1_active and not t2_active and ts_one:
                        for s1 in range(len(self.servers)):
                            idxs += [(j, s1, -1, t1, t2, 0)]
                            
            
                
            # Add end node
            end_time_active = (self.jobs[j].active_time[end_node_time-1] == 1)
            if end_time_active:
                for s1 in range(len(self.servers)):
                    idxs += [(j, s1, -1, end_node_time-1, end_node_time, 0)]
            else:
                idxs += [(j, -1, -1, end_node_time-1, end_node_time, 0)]

        
        self.h_vars = LpVariable.dicts("h",idxs,lowBound=0,upBound = 1, cat='Integer')
    
    def opt_auxiliary_vars(self):
        """
        Define auxiliary variables based on decision variable "h"
        - q (job location at server)
        - g (migration destination)
        - j (migration rate from server to server)
        """
        
        self.q_vars = {}
        self.g_vars = {}
        self.j_vars = {}
        self.i_vars = {}
        
        # Loop through every possible combination of Q
        q_list = []
        for u in range(len(self.users)):
            for s in range(-1, len(self.servers)):
                for t in range(self.sim_params.time_steps):
                    q_list += [(u,s,t)]
        
        for (u,s,t) in q_list:
            temp = 0
            # Check for active/inactive servers
            compute_flag = False
            if self.jobs[u].active_time[t] == 1:
                if s != -1:
                    compute_flag = True
            else:
                if s== -1:
                    temp = 1
            
            # Compute arrivals and departures to node 
            if compute_flag:
                for s1 in range(-1,len(self.servers)):
                    for t1 in range(-1,t):
                        for t2 in range(t1+1,t+1):
                            if s1 != -1 and s != -1:
                                for n in range(int(self.links.num_path[s1,s])+1):
                                    if (u,s1,s,t1,t2,n) in self.h_vars.keys():
                                        temp += self.h_vars[(u,s1,s,t1,t2,n)]
                                for n in range(int(self.links.num_path[s,s1]+1)):
                                    if (u,s,s1,t1,t2,n) in self.h_vars.keys():
                                        temp -= self.h_vars[(u,s,s1,t1,t2,n)]
                            else:
                                n = 0
                                if (u,s1,s,t1,t2,n) in self.h_vars.keys():
                                    temp += self.h_vars[(u,s1,s,t1,t2,n)]
                                if (u,s,s1,t1,t2,n) in self.h_vars.keys():
                                    temp -= self.h_vars[(u,s,s1,t1,t2,n)]
                        
            self.q_vars[(u,s,t)] = temp
            
        # Find J Variables - how much migration transition is taking place in 1ts
        j_list = []
        for u in range(len(self.users)):
            for s1 in range(0, len(self.servers)):
                for s2 in range(0, len(self.servers)):
                    for t in range(self.sim_params.time_steps):
                        if s1 != -1 and s2 != -1:
                            for n in range(int(self.links.num_path[s1,s2])+1):
                                j_list += [(u,s1,s2,t,n)]
        
        for (u,s1,s2,t,n) in j_list:
            temp = 0
            for t1 in range(0,t+1):
                for t2 in range(t+1,self.sim_params.time_steps):
                    if (u,s1,s2,t1,t2,n) in self.h_vars.keys():
                        temp += self.h_vars[(u,s1,s2,t1,t2,n)]*(1/(t2-t1))
            if temp != 0:
                self.j_vars[(u,s1,s2,t,n)] = temp
            
        
        # Find G Variables - if a destination is occuring for a migration
        g_list = []
        for u in range(len(self.users)):
            for s in range(len(self.servers)):
                for t in range(self.sim_params.time_steps):
                    g_list += [(u,s,t)]
        
        for (u,s,t) in g_list:
            temp = 0
            for s1 in range(len(self.servers)):
                if s1 != s:
                    for t1 in range(0,t+1):
                        for t2 in range(t+1,self.sim_params.time_steps):
                            for n in range(int(self.links.num_path[s1,s])):
                                if (u,s1,s,t1,t2,n) in self.h_vars.keys():
                                    temp += self.h_vars[(u,s1,s,t1,t2,n)]
            self.g_vars[(u,s,t)] = temp
            
        
    def opt_feasibility_constraints(self):
        """
        Make migration graph based constraints on "h" for real solution
        """
        
        # Restriction 1. All nodes in MG have same enter and leaving values
        
        
        # Pull all valid nodes of migration graph
        r1_nodes = []
        for u in range(len(self.users)):
            for t in range(self.sim_params.time_steps):
                if self.jobs[u].active_time[t] > 0:
                    for s in range(len(self.servers)):
                        r1_nodes += [(u,s,t)]
                else:
                    s = -1
                    r1_nodes += [(u,s,t)]
        
        # For each node make enter/departure same
        for (u,s,t) in r1_nodes:
            temp1, temp2 = 0, 0
            for s1 in range(-1,len(self.servers)):
                # Departures
                for t1 in range(-1,t):
                    if s1 != -1 and s != -1:
                        for n in range(int(self.links.num_path[s1,s])+1):
                            if (u,s1,s,t1,t,n) in self.h_vars.keys():
                                temp1 += self.h_vars[(u,s1,s,t1,t,n)]
                    else:
                        n = 0
                        if (u,s1,s,t1,t,n) in self.h_vars.keys():
                            temp1 += self.h_vars[(u,s1,s,t1,t,n)]
                
                # Arrivals
                for t2 in range(t+1,self.sim_params.time_steps+1):
                    if s1 != -1 and s != -1:
                        for n in range(int(self.links.num_path[s,s1])+1):
                            if (u,s,s1,t,t2,n) in self.h_vars.keys():
                                temp2 += self.h_vars[(u,s,s1,t,t2,n)]
                    else:
                        n = 0
                        if (u,s,s1,t,t2,n) in self.h_vars.keys():
                            temp2 += self.h_vars[(u,s,s1,t,t2,n)]
                            
            self.prob += (temp1 - temp2) == 0
        
        
        # Restriction 2. Make leave/arrive to start/end node just once
        
        for u in range(len(self.users)):
            temp_start = 0
            temp_end = 0
            for s in range(-1,len(self.servers)):
                # Start Node Equal
                if (u,-1,s,-1,0,0) in self.h_vars.keys():
                    temp_start += self.h_vars[(u,-1,s,-1,0,0)]
                # End Node Equal
                if (u,s,-1,self.sim_params.time_steps-1,self.sim_params.time_steps,0) in self.h_vars.keys():
                    temp_end += self.h_vars[(u,s,-1,self.sim_params.time_steps-1,self.sim_params.time_steps,0)]
            
            self.prob += (temp_start == 1)
            self.prob += (temp_end == 1)
                
        
    def opt_resource_constraints(self):
        """
        Make resource constraints based on decision variable
        """
        
        # 1. Placement Cost restriction
        
        # Build all combinations of (s,t) on migration graph
        place_list = []
        for s in range(len(self.servers)):
            for t in range(self.sim_params.time_steps):
                place_list += [(s,t)]
        
        # Loop through each and sum resources across users
        num_rsrc = self.servers[0].avail_rsrc.shape[0]
        for (s,t) in place_list:
            temp = []
            for i in range(num_rsrc):
                temp+= [0]
            for j in range(len(self.jobs)):
                for i in range(num_rsrc):
                    temp[i] += (self.q_vars[(j,s,t)] + self.g_vars[(j,s,t)])*self.jobs[j].placement_rsrc[i]
            for i in range(num_rsrc):
                if temp[i] != 0:
                    self.prob += (temp[i] <= self.servers[s].avail_rsrc[i])
                
                
        # 2. Bandwidth Cost restriction
        
        # Make server, path combination
        idx_list = []
        bw_list = []
        
        # Make migration index
        for t in range(self.sim_params.time_steps):
            for s3 in range(len(self.servers)):
                for s4 in range(len(self.servers)):
                    if s3 != s4:
                        bw_list += [(s3,s4,t)]
        
        # Make user index
        for u in range(len(self.users)):
            for s1 in range(len(self.servers)):
                for s2 in range(len(self.servers)):
                    if s1 != s2:
                        idx_list += [(u,s1,s2)]
        
        for (s3,s4,t) in bw_list:
            temp_service_bw = 0
            temp_mig_bw = 0
            
            for (u,s1,s2) in idx_list:
                # Service BAndwidth
                service_capacity = self.jobs[u].thruput_req * self.links.get_avgpath(s1,s2)[s3,s4]
                # i(u,s2,t) * q(u,s1,t)
                indication_service = self.users[u].server_prob[s2,t] * self.q_vars[(u,s1,t)]
                if service_capacity > 0:
                    temp_service_bw += service_capacity * indication_service
                
                # Migration Bandwidth
                for n in range(int(self.links.num_path[s1,s2])):
                    migration_capacity = self.jobs[u].migration_rsrc * self.links.get_subpath(s1,s2,n)[s3,s4]
                    if (u,s1,s2,t,n) in self.j_vars.keys():
                        indication_migration = self.j_vars[(u,s1,s2,t,n)]
                    else:
                        indication_migration = 0
                    if migration_capacity > 0:
                        temp_mig_bw += migration_capacity * indication_migration
            
            # Place total restriction
            if temp_service_bw != 0 or temp_mig_bw != 0:
                self.prob += (temp_service_bw + temp_mig_bw) <= self.links.rsrc_avail[s3,s4]
                        
        
        
    def opt_objective_function(self):
        """
        Make objective function to minimize
        """
        
        # Make all lists to loop through later
        placement_cost_list = []
        mig_bw_cost_list = []
        service_bw_cost_list = []
        latency_list1 = []
        latency_list2 = []
        
        # Placement cost list loop
        for t in range(self.sim_params.time_steps):
            for u in range(len(self.jobs)):
                for s in range(len(self.servers)):
                    placement_cost_list += [(u,s,t)]
                    
        # Bandwidth, latency loops
        for u in range(len(self.jobs)):
            for t in range(self.sim_params.time_steps):
                for s1 in range(len(self.servers)):
                    for s2 in range(len(self.servers)):
                        for n in range(int(self.links.num_path[s1,s2])):
                            for s3 in range(len(self.servers)):
                                for s4 in range(len(self.servers)):
                                    if s1 != s2 and s3 != s4:
                                        mig_bw_cost_list += [(u,t,s1,s2,n,s3,s4)]
                                        if (u,t,s1,s2,s3,s4) not in service_bw_cost_list:
                                            service_bw_cost_list += [(u,t,s1,s2,s3,s4)]
                                        if (u,t,s1,s2) not in latency_list1:
                                            latency_list1 += [(u,t,s1,s2)]
                                        if (s3,s4) not in latency_list2:
                                            latency_list2 += [(s3,s4)]
        
        # 1. Placemnt Cost
        
        placement_cost = 0
        for (u,s,t) in placement_cost_list:
            check1 = (u,s,t) in self.q_vars.keys()
            check2 = (u,s,t) in self.g_vars.keys()
            s_placement_cost = np.dot(self.jobs[u].placement_rsrc, self.servers[s].svr_rsrc_cost)
            
            if check1:
                placement_cost += s_placement_cost * self.q_vars[(u,s,t)]
            if check2:
                placement_cost += s_placement_cost * self.g_vars[(u,s,t)]
        
        # 2. Migration Bandwidth Cost
        
        mig_bw_cost = 0
        for (u,t,s1,s2,n,s3,s4) in mig_bw_cost_list:
            if (u,s1,s2,n,t) in self.j_vars.keys():
                mig_amount = self.jobs[u].migration_rsrc * self.j_vars[(u,s1,s2,n,t)]
                lit_link_costs = self.links.get_subpath(s1,s2,n)[s3,s4] * self.links.cost_links[s3,s4]
                mig_bw_cost += mig_amount * lit_link_costs
        
        # 3. Service Bandwidth Cost
        serv_bw_cost = 0
        for (u,t,s1,s2,s3,s4) in service_bw_cost_list:
            if (u,s2,t) in self.q_vars.keys():
                usr_job_flag = self.users[u].server_prob[s1,t] * self.q_vars[(u,s2,t)]
                expected_link = self.links.cost_links[s3,s4] * self.links.get_avgpath(s2,s1)[s3,s4]
                serv_bw_cost += self.jobs[u].thruput_req * usr_job_flag * expected_link
        
        # 4. Latency Cost
        # Make new decision variable for latency every (u,t,s1,s2) for max(leftover,0)
        self.max_var = LpVariable.dicts("max",latency_list1,lowBound=0,upBound = None, cat='Continuous')
        
        usr_latency_cost = 0
        for (u,t,s1,s2) in latency_list1:
            curr_latency = 0
            avg_link = self.links.get_avgpath(s1,s2)
            indication_var = self.q_vars[(u,s1,t)] * self.users[u].server_prob[s2,t]
            
            # Obtain current latency value 
            for (s3,s4) in latency_list2:
                if (u,s1,t) in self.q_vars.keys():
                    delay = self.links.switch_delay + self.links.dist_delay * self.links.get_distance(s3,s4)
                    curr_latency += avg_link[s3,s4] * indication_var * delay
            
            # Take maximum of total latency - latency requirement
            leftover_latency = curr_latency - self.jobs[u].latency_req
            self.prob += (self.max_var[(u,t,s1,s2)]) >= leftover_latency
            
            usr_latency_cost += self.jobs[u].latency_penalty * self.max_var[(u,t,s1,s2)]
            
        
        # Sum all the costs together
        self.prob += placement_cost + mig_bw_cost + serv_bw_cost + usr_latency_cost
       

    """
    Callable Functions
    """
    def solve_ILP(self):
        """
        solve the ILP problem built
        - measure the total time it takes to solve the problem
        """
        
        # Measure Time of running
        start_time = time.time()
        self.prob.solve()
        self.run_time = time.time() - start_time
        
        # Print Status
        print("Status:", constants.LpStatus[optim_prob.prob.status])
        print("Run Time:", self.run_time, "s")
        
        
    def extract_plan(self):
        """
        Alter the decision variable "h" into a migration plan
        """
        
        