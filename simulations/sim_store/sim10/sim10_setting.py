# Import All Custom Classes
from Server import *
from User import *
from Link import *
from Job import *
from PlanGenerator import *
from get_one_sim_usr import *
import os, sys

import numpy as np
import random

def simulation_setting(num_server,num_user):    
    """
    Make Simulation Parameters
    """
    sim_param = Sim_Params(time_steps=20, x_length = 2.4, y_length = 1.7, max_edge_length=1,num_path_limit=1)
    boundaries = np.array([[0,sim_param.x_length],[0,sim_param.y_length]])


    """
    Make Job Profiles
    """
    # REsources used are CPU (no. cores) storage (GB), and RAM (GB)
    # througput is in mb/s
    # Latency is in ms

    job_profile1 = Job_Profile(job_name = "VR",
                               latency_req_range=[25/1000*0, 40/1000*0], 
                               thruput_req_range=[100/1000, 400/1000], 
                               length_range=[sim_param.time_steps,sim_param.time_steps],  
                               placement_rsrc_range = np.array([[2,3],[8,16],[2,5]]),
                               migration_amt_range = [2, 3],
                               latency_penalty_range = [0.8*1.5,1.2*1.5],
                               thruput_penalty_range = [0.4,0.8]) 

    job_profile2 = Job_Profile(job_name = "Assistant",
                               latency_req_range=[100/1000 * 0 , 200/1000 * 0 ],
                               thruput_req_range=[50/1000, 100/1000],
                               length_range=[sim_param.time_steps,sim_param.time_steps],
                               placement_rsrc_range = np.array([[1,1],[0.5,1],[0.5,1]]),
                               migration_amt_range = [1, 2],
                               latency_penalty_range = [0.5*1.5, 0.8*1.5],
                               thruput_penalty_range = [0.2,0.3])

    job_profile3 = Job_Profile(job_name = "AR",
                               latency_req_range=[50/1000 * 0 , 80/1000 * 0 ], 
                               thruput_req_range=[90/1000, 300/1000],
                               length_range=[sim_param.time_steps,sim_param.time_steps],
                               placement_rsrc_range = np.array([[1,2],[2,4],[1,2]]),
                               migration_amt_range = [1.5, 2.5],
                               latency_penalty_range = [0.6*1.5, 1*1.5],
                               thruput_penalty_range = [0.4, 0.6])

    job_profiles = [job_profile1, job_profile2, job_profile3]


    job_profiles = [job_profile1, job_profile2, job_profile3]


    """
    Make Servers
    """

    # Server Settings
    num_server_l1 = num_server-4
    num_server_l2 = 3
    num_server_l3 = 1

    num_resource = 3
    # (cores, storage GB, ram)
    weak_range = np.array([[30,50],[1000,1500],[4,16]])
    strong_range = np.array([[100,300],[10000,20000],[1000,1500]])

    rsrc_cost = np.array([0.02, 0.01, 0.02])

    rsrc_cost_scale_lv1 = 1.1
    rsrc_cost_scale_lv2 = 1
    rsrc_cost_scale_lv3 = 0.9

    # Generate Server
    servers_l1 = []
    servers_l2 = []
    servers_l3 = []
    idx_counter = 0

    for i in range(num_server_l1):
        servers_l1.append(Server(boundaries,level=1,rand_locs=True,locs=None))
        servers_l1[-1].server_resources(num_resource, weak_range, strong_range)
        servers_l1[-1].assign_id(idx_counter)
        servers_l1[-1].server_resources_cost(num_resource,rsrc_cost*rsrc_cost_scale_lv1)
        idx_counter += 1

    for i in range(num_server_l2):
        servers_l2.append(Server(boundaries,level=2,rand_locs=True,locs=None))
        servers_l2[-1].server_resources(num_resource, weak_range, strong_range)
        servers_l2[-1].assign_id(idx_counter)
        servers_l2[-1].server_resources_cost(num_resource,rsrc_cost*rsrc_cost_scale_lv2)
        idx_counter += 1

    for i in range(num_server_l3):
        servers_l3.append(Server(boundaries,level=3,rand_locs=False,locs=np.array([200,200])))
        servers_l3[-1].server_resources(num_resource, weak_range, strong_range)
        servers_l3[-1].assign_id(idx_counter)
        servers_l3[-1].server_resources_cost(num_resource,rsrc_cost*rsrc_cost_scale_lv3)
        idx_counter += 1

    servers = servers_l1 + servers_l2 + servers_l3


    """
    Make Links
    """

    # Link Settings
    num_link = [0,1,2,3]
    prob_link = [0.5,0.2,0.2,0.1]
    lv_minmax = np.array(([[100,200],[1000,3000],[3000,5000]]))
    lv1_transmission = 1
    link_costs = np.array([0.06, 0.06, 0.06])
    latency_settings = [250 * 1e-3, 50 * 1e-3] #[ms per switch, ms per mile]

    links = Link(servers, num_link, prob_link, lv_minmax, link_costs, latency_settings,lv1_transmission)


    """
    Make Users
    """

    # User Settings
    num_user_m0 = int(num_user/2) # stochastic
    num_user_m1 = 0 # deterministic
    num_user_m0_ONE = int(num_user/2) # stochastic - ONE
    num_user_m1_ONE = 0 # deterministic - ONE
    total_count = num_user_m0 + num_user_m1

    max_speed = 2.5
    lamdas = [1/0.9,1/0.9] # 3 mph, 10 mph, 20 mph

    # Generate Server
    users_m0 = []
    users_m1 = []
    idx_counter = 0
    
    # Select which one users to draw from
    usr_info = get_one_sim_usr()
    key_list = []
    for key in usr_info:
        key_list += [key]
    
    random.shuffle(key_list)
    
    # Generate Stochastic users
    mvmt_class = 0
    num_path = 10
    num_path_orig = 1
    
    for i in range(num_user_m0):
        users_m0 += [User(boundaries, sim_param.time_steps, mvmt_class, lamdas, 
                          max_speed, num_path)]
        users_m0[-1].generate_MC(servers)
        users_m0[-1].assign_id(idx_counter)
        idx_counter += 1
    
    for i in range(num_user_m0_ONE):
        usr_idx = key_list[idx_counter]
        users_m0 += [ONE_User(boundaries, sim_param.time_steps, 
                              max_speed, num_path, num_path_orig, 
                              usr_info[usr_idx], mvmt_class)]
        users_m0[-1].generate_MC(servers)
        users_m0[-1].assign_id(idx_counter)
        idx_counter += 1

        
    # Generate Deterministic Users
    mvmt_class = 1
    num_path = 1
    num_path_orig = 1
    
    for i in range(num_user_m1):
        users_m1 += [User(boundaries, sim_param.time_steps, mvmt_class, lamdas, 
                          max_speed, num_path)]
        users_m1[-1].generate_MC(servers)
        users_m1[-1].assign_id(idx_counter)
        idx_counter += 1
        
    
    for i in range(num_user_m1_ONE):
        usr_idx = key_list[idx_counter]
        users_m1 += [ONE_User(boundaries, sim_param.time_steps,
                              max_speed, num_path, num_path_orig, 
                              usr_info[usr_idx], mvmt_class)]
        users_m1[-1].generate_MC(servers)
        users_m1[-1].assign_id(idx_counter)
        idx_counter += 1

    users = users_m0 + users_m1


    """
    Make Jobs
    - "I'm just going to do it"
    """

    # Job settings
    job_type0 = 20 # VR
    job_type1 = 0 # Assistant
    job_type2 = 0 # AR

    jobs0 = []
    jobs1 = []
    jobs2 = []
    idx_counter = 0

    total_job_count = job_type0+job_type1+job_type2
    draw_job_id = np.random.choice(total_job_count, total_job_count, replace=False)

    for i in range(job_type0):
        jobs0 += [Job(job_type = 0,
                      user_id = draw_job_id[idx_counter],
                      time_steps=sim_param.time_steps,
                      job_profiles = job_profiles)]
        idx_counter += 1

    for i in range(job_type1):
        jobs1 += [Job(job_type = 1,
                      user_id = draw_job_id[idx_counter],
                      time_steps=sim_param.time_steps,
                      job_profiles = job_profiles)]
        idx_counter += 1

    for i in range(job_type2):
        jobs2 += [Job(job_type = 2,
                      user_id = draw_job_id[idx_counter],
                      time_steps=sim_param.time_steps,
                      job_profiles=job_profiles)]
        idx_counter += 1

    jobs = jobs0 + jobs1 + jobs2

    """
    Mix Jobs and Users
    """

    random.shuffle(users)
    random.shuffle(jobs)

    """
    Refresh rate
    - Add batch functionality to jobs
    """

    refresh_rate = [0,0]
    refresh = True

    for j in range(len(jobs)):
        jobs[j].info_from_usr(users[j],refresh_rate,refresh)

    return users, servers, links, jobs, sim_param