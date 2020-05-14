import numpy as np

class Job:
    """
    Job: Associated with each user id and define
    - Job type, resource requirements, UE requirements, arrival, departure times
    """
    
    def __init__(self, job_type, user_id, time_steps, job_profiles):
        """
        job_type - integer [0,1,2] based on the sample profiles we have 
        user_id - associate job with user id
        """
        
        self.user_id = user_id
        self.job_type = job_type
        self.time_steps = time_steps
        self.job_profile = job_profiles[job_type]
        
        # User Experience Requirements for this job
        self.latency_req, self.thruput_req = self.draw_UE_req()
        
        # Draw Arrival and Departure time of job
        self.arrival_time, self.departure_time, self.active_time = self.draw_active_time()
        
        # Draw placement resources
        self.placement_rsrc = self.draw_placement_rsrc()
        
        # Draw Migration BW Resources
        self.migration_rsrc = self.draw_migration_rsrc()
        
    def draw_UE_req(self):
        """
        Tap into job profile and obtain user experience thresholds
        Draw from uniform distribution to determine
        """

        latency_req = np.random.uniform(self.job_profile.latency_req_range[0],
                                        self.job_profile.latency_req_range[1])
        
        thruput_req = np.random.uniform(self.job_profile.thruput_req_range[0],
                                        self.job_profile.thruput_req_range[1])
        
        return latency_req, thruput_req
    
    def draw_active_time(self):
        """
        draw the active time for the job for the duration of the simulation
        """
        
        job_length = np.random.randint(self.job_profile.length_range[0],
                                       self.job_profile.length_range[1]+1)
        arrival_time = np.random.randint(0, self.time_steps - job_length + 1)
        departure_time = arrival_time + job_length
        active_time = np.zeros(self.time_steps)
        active_time[arrival_time:departure_time] = 1
        
        return arrival_time, departure_time, active_time
    
    def draw_placement_rsrc(self):
        """
        Draw placement resources based on the number of resources we are using
        """
        
        num_rsrc = self.job_profile.placement_rsrc_range.shape[0]
        placement_rsrc = np.zeros(num_rsrc)
        
        for i in range(num_rsrc):
            placement_rsrc[i] = np.random.uniform(self.job_profile.placement_rsrc_range[i,0],
                                                  self.job_profile.placement_rsrc_range[i,1])
        
        return placement_rsrc

    def draw_migration_rsrc(self):
        """
        Draw total amount of memory that has to be migrated 
        """
        
        return np.random.uniform(self.job_profile.migration_amt_range[0],
                                 self.job_profile.migration_amt_range[1])
    

class Job_Profile:
    """
    Make list of job profiles with
    - UE properties (latency, thruput requirement)
    - Length Properties
    - Resource Consumption
    """
    
    def __init__(self, job_name,
                    latency_req_range,
                    thruput_req_range,
                    length_range,
                    placement_rsrc_range,
                    migration_amt_range):
        """
        Add job profile to list 
        """
        
        self.job_name = job_name
        self.latency_req_range = latency_req_range # milisecond
        self.thruput_req_range = thruput_req_range # MBPS
        self.length_range = length_range # In units of 5 minuts
        self.placement_rsrc_range = placement_rsrc_range
        self.migration_amt_range = migration_amt_range