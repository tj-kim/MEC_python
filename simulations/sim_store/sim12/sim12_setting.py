# Import All Custom Classes
from Server import *
from User import *
from Link import *
from Job import *
from PlanGenerator import *
from get_one_sim_usr import *
from load_crawdad_usr import *

import os, sys

import numpy as np
import random

def simulation_setting(db):    
    
    """
    Make Simulation Parameters
    """
    sim_param = Sim_Params(time_steps=10, x_length = 6.3, y_length = 11.2, max_edge_length=2,num_path_limit=5)
    boundaries = np.array([[-sim_param.x_length,sim_param.x_length],[-sim_param.y_length,sim_param.y_length]])


    """
    Make Job Profiles
    """
    # REsources used are CPU (no. cores) storage (GB), and RAM (GB)
    # througput is in mb/s
    # Latency is in ms

    job_profile1 = Job_Profile(job_name = "VR",
                               latency_req_range=[25/1000*0, 40/1000*0], 
                               thruput_req_range=[100/1000, 400/1000], 
                               length_range=[8,10],  
                               placement_rsrc_range = np.array([[2,3],[8,16],[2,5]]),
                               migration_amt_range = [2, 3],
                               latency_penalty_range = [0.8*1.5,1.2*1.5],
                               thruput_penalty_range = [0.4,0.8]) 

    job_profile2 = Job_Profile(job_name = "Assistant",
                               latency_req_range=[100/1000 * 0 , 200/1000 * 0 ],
                               thruput_req_range=[50/1000, 100/1000],
                               length_range=[6,9],
                               placement_rsrc_range = np.array([[1,1],[0.5,1],[0.5,1]]),
                               migration_amt_range = [1, 2],
                               latency_penalty_range = [0.5*1.5, 0.8*1.5],
                               thruput_penalty_range = [0.2,0.3])

    job_profile3 = Job_Profile(job_name = "AR",
                               latency_req_range=[50/1000 * 0 , 80/1000 * 0 ], 
                               thruput_req_range=[90/1000, 300/1000],
                               length_range=[7,9],
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
    num_server_l1 = 6
    num_server_l2 = 3
    num_server_l3 = 1

    num_resource = 3
    # (cores, storage GB, ram)
    weak_range = np.array([[6,8],[1000,1500],[100,100]])
    strong_range = np.array([[14,20],[10000,20000],[1000,1500]])

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
        servers_l3.append(Server(boundaries,level=3,rand_locs=False,locs=np.array([10,0])))
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
    lv_minmax = np.array(([[25,40],[1000,3000],[3000,5000]]))
    lv1_transmission = 1
    link_costs = np.array([0.06, 0.06, 0.06])
    latency_settings = [250 * 1e-3, 50 * 1e-3] #[ms per switch, ms per mile]

    links = Link(servers, num_link, prob_link, lv_minmax, link_costs, latency_settings,lv1_transmission)


    """
    Make Users
    """

    # User Settings
    num_user = 20

    max_speed = 80
    num_path = 5
    num_path_orig = 1
    mvmt_class = 0 # Dummy

    # Generate Server
    users = []
    idx_counter = 0

    for i in range(num_user):

        confirmed = False

        while confirmed is False:
            db_idx = np.random.randint(len(db))
            mvmt_array = random.choice(list(db[db_idx].values()))
            # print(mvmt_array.shape[0])
            if mvmt_array.shape[0] > sim_param.time_steps:
                confirmed = True

        users += [Crawdad_User(boundaries, sim_param.time_steps, max_speed, num_path, num_path_orig, mvmt_array, mvmt_class = 0)]
        users[-1].generate_MC(servers)
        users[-1].assign_id(idx_counter)
        idx_counter += 1


    """
    Make Jobs
    - "I'm just going to do it"
    """

    # Job settings
    job_type0 = 7 # VR
    job_type1 = 7 # Assistant
    job_type2 = 6 # AR

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

    refresh_rate = [2,0]
    refresh = True

    for j in range(len(jobs)):
        jobs[j].info_from_usr(users[j],refresh_rate,refresh)

    return users, servers, links, jobs, sim_param