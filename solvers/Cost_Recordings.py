import numpy as np
import math

class Cost_Recordings:
    
    """
    Migration_Plans: 
        - Collects all migration plans generated for an entire system
        - mig_plan_dict : 6 x time_steps np array with rows
            - timeslot, user_active_flag, usr_voronoi, source_svr, dest_svr, mig_rate
    """
    
    def __init__(self, mig_plan_obj):
        """
        mig_plan_obj - is the migration plan object with all the plans calculated
        """
        
        self.categories = ["placement_cost","mig_bw_cost",
                          "srv_bw_cost", "latency_cost",
                          "thruput_cost", "total_cost"]
        
        # Import information
        self.mig_plan = mig_plan_obj
        self.prob = mig_plan_obj.prob
        self.servers = self.prob.servers
        self.users = self.prob.users
        self.links = self.prob.links
        self.jobs = self.prob.jobs
        self.sim_params = self.mig_plan.sim_params
        
        # Cost dicts to compute
        self.user_cost_cumulative = None
        self.user_cost_instance = None
        self.all_cost_cumulative = None
        self.all_cost_instance = None
        
        # Functions to calculate all costs
        self.calc_user_instance()
        self.calc_user_cumulative()
        self.calc_all_instance()
        self.calc_all_cumulative()
        
        
    """
    Automatic Functions to Record Cost
    """
    
    def calc_user_instance(self):
        
        self.user_cost_instance = []
        
        for j in range(len(self.jobs)):
            usr_instance_cost_dict = {}
            
            placement_cost = np.zeros(self.sim_params.time_steps)
            mig_bw_cost = np.zeros(self.sim_params.time_steps)
            srv_bw_cost = np.zeros(self.sim_params.time_steps)
            latency_cost = np.zeros(self.sim_params.time_steps)
            thruput_cost = np.zeros(self.sim_params.time_steps)
            total_cost = np.zeros(self.sim_params.time_steps)
            
            mig_plan = self.mig_plan.mig_plan_dict[j]
            
            for t in range(self.sim_params.time_steps):
                if self.jobs[j].active_time[t] == 1:
                    
                    # Record Placment Cost
                    s1 = int(mig_plan["source_server"][t])
                    s2 = int(mig_plan["dest_server"][t])
                    usr_svr = int(mig_plan["user_voronoi"][t])
                    
                    if s1 != s2:
                        placement_cost[t] += np.dot(self.jobs[j].placement_rsrc,
                                                    self.servers[s1].svr_rsrc_cost)
                        placement_cost[t] += np.dot(self.jobs[j].placement_rsrc,
                                                    self.servers[s2].svr_rsrc_cost)
                    else:
                        placement_cost[t] += np.dot(self.jobs[j].placement_rsrc,
                                                    self.servers[s1].svr_rsrc_cost)
                        
                    # Record Mig_BW_cost
                    if mig_plan["mig_rate"][t] > 0:
                        num_path = int(mig_plan["mig_link_id"][t])
                        mig_links = self.links.get_subpath(s1,s2,num_path)
                        cost = self.jobs[j].migration_rsrc * np.multiply(mig_links,self.links.cost_links)
                        cost = np.sum(np.sum(cost))
                        mig_bw_cost[t] += cost
                    
                    # Record Service BW Cost & THRUPUT Cost
                    num_path = int(mig_plan["service_link_id"][t])
                    if num_path != -1:
                        # Bandwidth Cost
                        serv_links = self.links.get_subpath(s1,usr_svr,num_path)
                        thruput = mig_plan['service_thruput'][t]
                        cost = thruput * np.multiply(serv_links, self.links.cost_links)
                        cost = np.sum(np.sum(cost))
                        srv_bw_cost[t] += cost
                        
                        # UE Cost
                        req = self.jobs[j].thruput_req
                        penalty_rate = self.jobs[j].thruput_penalty
                        remain = req - thruput
                        cost = 0
                        
                        if remain > 0:
                            cost += remain * penalty_rate
                        
                        thruput_cost[t] = cost 
                    
                    
                    # Latency Cost
                    latency = mig_plan['latency'][t]
                    if latency > 0:
                        req = self.jobs[j].latency_req
                        penalty_rate = self.jobs[j].latency_penalty
                        remain = latency - req
                        cost = 0
                        
                        if remain > 0:
                            cost += remain * penalty_rate
                        
                        latency_cost[t] += cost
                    
                    # Total Cost
                    total_cost[t] = placement_cost[t] + mig_bw_cost[t] + srv_bw_cost[t] + thruput_cost[t] + latency_cost[t]
                        
            # Record Cost for this user
            usr_instance_cost_dict["placement_cost"] = placement_cost
            usr_instance_cost_dict["mig_bw_cost"] = mig_bw_cost
            usr_instance_cost_dict["srv_bw_cost"] = srv_bw_cost
            usr_instance_cost_dict["latency_cost"] = latency_cost
            usr_instance_cost_dict["thruput_cost"] = thruput_cost
            
            usr_instance_cost_dict["total_cost"] = total_cost
            
            self.user_cost_instance += [usr_instance_cost_dict]

    def calc_user_cumulative(self):
        """
        Take user_cost_instance and add from previous timestep
        """

        self.user_cost_cumulative = []
        
        for j in range(len(self.jobs)):
            usr_cumul_cost_dict = {}
            
            for label in self.categories:
                label_cost_instance = self.user_cost_instance[j][label]
                label_cumulative = np.zeros(label_cost_instance.shape)
                
                for t in range(self.sim_params.time_steps):
                    cumul_val = np.sum(label_cost_instance[:t+1])
                    label_cumulative[t] = cumul_val
                
                usr_cumul_cost_dict[label] = label_cumulative
            
            self.user_cost_cumulative += [usr_cumul_cost_dict]
    
    def calc_all_instance(self):
        """
        Total cost per instance across all jobs
        """
        
        self.all_cost_instance = {}
        
        for label in self.categories:
            label_all_instance = np.zeros(self.user_cost_instance[0][label].shape)
            for j in range(len(self.jobs)):
                usr_cost_label = self.user_cost_instance[j][label]
                label_all_instance += usr_cost_label
            
            self.all_cost_instance[label] = label_all_instance
    
    def calc_all_cumulative(self):
        """
        Cumulative Cost over all jobs
        """
        self.all_cost_cumulative = {}
        
        
        for label in self.categories:
            label_all_cumulative = np.zeros(self.user_cost_cumulative[0][label].shape)
            for j in range(len(self.jobs)):
                usr_cost_label = self.user_cost_cumulative[j][label]
                label_all_cumulative += usr_cost_label
            
            self.all_cost_cumulative[label] = label_all_cumulative
                        
    """
    Functions for calling cost
    """
    
    def call_cost(self, indiv_no = -1, resource_type = "total_cost", cumulative = True, time_step = -1):
        
        val = None
        
        if resource_type not in self.categories:
            resource_type = "total_cost"
            
        if time_step < 0 and time_step >= self.sim_params.time_steps:
            time_step = self.sim_params.time_steps - 1
        
        # Case from all jobs
        if indiv_no < 0 and indiv_no > len(self.jobs):
            if cumulative:
                val = self.all_cost_cumulative[resource_type][time_step]
            else:
                val = self.all_cost_instance[resource_type][time_step]
        else:
            if cumulative:
                val = self.user_cost_cumulative[indiv_no][resource_type][time_step]
            else:
                val = self.user_cost_instance[indiv_no][resource_type][time_step]
        
        return val