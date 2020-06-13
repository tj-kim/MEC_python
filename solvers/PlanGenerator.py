import numpy as np
import copy

class PlanGenerator:
    """
    PlanGenerator
    - Given information on users, servers, links, jobs, simulation parameters, resources
    - Make plan for each of the users and record cost
    """
    
    def __init__(self, users, servers, links, jobs, sim_params):
        
        # Store all components within the plan generator
        self.users = copy.copy(users) # Copy so that conditioning MC doesn't overwrite for other sims
        self.servers = servers
        self.links = links
        self.jobs = jobs
        self.sim_params = copy.copy(sim_params)
        
        # Define resource restrictions per timestep
        self.resource_constraints = self.set_resource_constraints(servers, links)
        self.user_experience = self.set_user_experience(jobs)
        
    """
    Init Helper Functions
    """
    def set_resource_constraints(self, servers, links):
        """
        Invoke resource constraint class to set resource capacities 
        servers and links
        """
        
        return Resource_Constraints(servers, links, self.sim_params.time_steps)
    
    def set_user_experience(self, jobs):
        """
        Store information about latency and throughput requirements for each job
        """
        
        return User_Experience(jobs)
    
"""
Helper Classes to generate plan generator
"""
    
class Resource_Constraints:
    """
    Resource constraints given server and links
    """
    def __init__(self, servers, links, time_steps):
        
        # Calculate server resources
        server_rsrc = np.zeros((len(servers),servers[0].num_rsrc))
        
        for i in range(len(servers)):
            server_rsrc[i,:] = servers[i].avail_rsrc
            
        # Set server and link resource
        self.server_rsrc = np.repeat(np.expand_dims(server_rsrc,axis=-1),repeats=time_steps, axis=2)
        self.link_rsrc = np.repeat(np.expand_dims(links.rsrc_avail,axis=-1), repeats=time_steps, axis=2)
    
    
class User_Experience:
    """
    User Experience thresholds given in latency and service bandwidth throughput
    """
    def __init__(self, jobs):
        
        self.latency = np.zeros(len(jobs))
        self.thruput = np.zeros(len(jobs))
        
        for i in range(len(jobs)):
            self.latency[i] = jobs[i].latency_req
            self.thruput[i] = jobs[i].thruput_req

class Sim_Params:
    """
    Simulation params hold information about system setting for simulation
    - timestep - 5 min per timestep
    - length - 1 mile per unit length
    """
    
    def __init__(self, time_steps, x_length, y_length, max_edge_length):
        
        self.time_steps = time_steps
        self.x_length = x_length
        self.y_length = y_length
        self.max_edge_length = max_edge_length