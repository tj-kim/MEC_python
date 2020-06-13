import numpy as np
import math

class Migration_Plans:
    """
    Migration_Plans: 
        - Collects all migration plans generated for an entire system
        - mig_plan_dict : 6 x time_steps np array with rows
            - timeslot, user_active_flag, usr_voronoi, source_svr, dest_svr, mig_rate
    """
    
    def __init__(self, users, jobs, time_steps):
        """
        users - list of user class (used to extract user location)
        time_steps - how many timesteps to simulate user movement for
        """
        
        self.mig_plan_dict = {}
        
        # Initialize dictionary 
        self.dict_init(users,jobs,time_steps)
    
    """
    Init Helper Function
    """
    
    def dict_init(self, users, jobs, time_steps):
        
        param_collection = ["time_slot", "user_active_flag", 
                            "user_voronoi", "source_server", 
                            "dest_server", "mig_rate"]
        
        for u in range(len(users)):
            temp_usr_dict = {}
            
            for p in param_collection:
                temp_usr_dict[p] = np.zeros(time_steps)
            
            # Record active time and user voronoi
            for t in range(int(time_steps)):
                temp_usr_dict["user_voronoi"][t] = users[u].user_voronoi_true[t]
                temp_usr_dict["user_active_flag"][t] = jobs[u].active_time[t]
            
            self.mig_plan_dict[u] = temp_usr_dict
    
    """
    Extraction Functions
    """
    
    def from_ILP(self):
        """
        From decision variables h we want to replace the zero vectors of 
        self.mig_plan_dict with relevant values based on the decision vars
        """
        return