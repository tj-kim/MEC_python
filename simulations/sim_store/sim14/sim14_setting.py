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

def simulation_setting(db):    
    """
    Make Simulation Parameters
    """
    sim_param = Sim_Params(time_steps=12, x_length = 5, y_length = 5, max_edge_length=3,num_path_limit=5)
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


    """
    Make Servers
    """

    # Server Settings
    num_server_l1 = 5
    num_server_l2 = 1
    num_server_l3 = 1
    
    custom_locs = [[1,1],[1,4],[4,1],[4,4], [2.5,1], [2.5,4]]

    num_resource = 3
    # (cores, storage GB, ram)
    weak_range = np.array([[4,6],[1000,1500],[100,100]])
    strong_range = np.array([[6,8],[10000,20000],[1000,1500]])

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
        servers_l1.append(Server(boundaries,level=1,rand_locs=False,locs=np.array(custom_locs[idx_counter])))
        servers_l1[-1].server_resources(num_resource, weak_range, strong_range)
        servers_l1[-1].assign_id(idx_counter)
        servers_l1[-1].server_resources_cost(num_resource,rsrc_cost*rsrc_cost_scale_lv1)
        idx_counter += 1

    for i in range(num_server_l2):
        servers_l2.append(Server(boundaries,level=2,rand_locs=False,locs=np.array(custom_locs[idx_counter])))
        servers_l2[-1].server_resources(num_resource, weak_range, strong_range)
        servers_l2[-1].assign_id(idx_counter)
        servers_l2[-1].server_resources_cost(num_resource,rsrc_cost*rsrc_cost_scale_lv2)
        idx_counter += 1

    for i in range(num_server_l3):
        servers_l3.append(Server(boundaries,level=3,rand_locs=False,locs=np.array([10,10])))
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
    lv_minmax = np.array(([[25,40],[1000,3000],[3000,5000]]))*100
    lv1_transmission = 1
    link_costs = np.array([0.06, 0.06, 0.06]) * 2 # multiply cost by 2
    latency_settings = [250/2 * 1e-3, 50/2 * 1e-3] #[ms per switch, ms per mile] - div cost by 2

    links = Link(servers, num_link, prob_link, lv_minmax, link_costs, latency_settings,lv1_transmission)

    """
    Make Users
    """

    # User Settings
    num_user = 10

    max_speed = 80
    num_path = 30
    num_path_orig = 1
    mvmt_class = 0 # Dummy

    spd_thresh = [0.2, 3]
    boundary_thresh = [boundaries[0,1],boundaries[1,1]]

    # Generate Server
    users = []
    idx_counter = 0
    infos = np.zeros([num_user, 5]) # <min x, max x, min y, max y, meanspd>

    for i in range(num_user):

        confirmed = False

        # Add filtering based on mean loc and start location randomization
        while confirmed is False:
            db_idx = np.random.randint(len(db))
            mvmt_array = random.choice(list(db[db_idx].values()))
            if mvmt_array.shape[0] > sim_param.time_steps:
                # Take mean speed here:
                new_mvmt_array = draw_ts(mvmt_array, sim_param.time_steps)
                mean_spd = avg_speed(new_mvmt_array)
                xmin = min(new_mvmt_array[:,0])
                xmax = max(new_mvmt_array[:,0])
                ymin = min(new_mvmt_array[:,1])
                ymax = max(new_mvmt_array[:,1])

                if mean_spd >= spd_thresh[0] and mean_spd <= spd_thresh[1]:
                    if (abs(xmax-xmin) <= boundary_thresh[0]) and (abs(ymax-ymin) <= boundary_thresh[1]):
                        confirmed = True

        # Edit initialization point of travel for more even distribution
        # Assuming all boundaries go from 0-x, 0-y
        xlow = -1*min(new_mvmt_array[:,0])
        xhigh = (boundary_thresh[0] - max(new_mvmt_array[:,0]))
        ylow = -1*min(new_mvmt_array[:,1])
        yhigh = (boundary_thresh[1] - max(new_mvmt_array[:,1]))
        x_offset = np.random.uniform(low= xlow, high=xhigh)
        y_offset = np.random.uniform(low=ylow,high=yhigh)

        new_mvmt_array[:,0] += x_offset
        new_mvmt_array[:,1] += y_offset

        xmin = min(new_mvmt_array[:,0])
        xmax = max(new_mvmt_array[:,0])
        ymin = min(new_mvmt_array[:,1])
        ymax = max(new_mvmt_array[:,1])
        infos[i] = np.array([xmin,xmax,ymin,ymax,mean_spd])

        users += [Crawdad_User(boundaries, sim_param.time_steps, max_speed, num_path, 
                               num_path_orig, new_mvmt_array, mean_spd, mvmt_class)]
        users[-1].generate_MC(servers)
        users[-1].assign_id(idx_counter)
        idx_counter += 1


    """
    Make Jobs
    - "I'm just going to do it"
    """

    # Job settings
    job_type0 = 3 # VR
    job_type1 = 3 # Assistant
    job_type2 = 4 # AR

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

    refresh_rate = [3,0]
    refresh = False

    for j in range(len(jobs)):
        jobs[j].info_from_usr(users[j],refresh_rate,refresh)

    return users, servers, links, jobs, sim_param