import numpy as np
import os, sys

def get_one_sim_usr():
    """
    Get users and their average speeds in a list in onesim
    """
    # Import Data and eliminate z axis
    one_sim_data = np.genfromtxt('ONE_SIM.csv', delimiter=',')
    one_sim_data = one_sim_data[:,:4]

    # Data polishing
    meter2mile = 0.000621371
    # Convert Time from seconds --> 5 min intervals, or 200s interval for ns3 sim
    # one_sim_data[:,0] *= (1/300)
    one_sim_data[:,0] = np.floor(one_sim_data[:,0])
    # Convert distance from meters --> Miles
    one_sim_data[:,2] *= meter2mile
    one_sim_data[:,3] *= meter2mile

    # Data Dividing and sampling
    ids = one_sim_data[:,1]
    unique_ids = np.unique(ids) 

    one_users = {}
    for idx in unique_ids:
        # Take id locations and take out relevant id data
        curr_locs = np.where(ids == idx)[0]
        start_loc, end_loc = curr_locs[0], curr_locs[-1]
        one_sim_data_u = one_sim_data[start_loc:end_loc,:]

        # Sampling (time seems to go from zero to 36)
        times = np.arange(0,37)
        sampled_data = np.zeros((times.shape[0],5))

        for t in times:
            # print("idx,t", idx, t)
            curr_time_loc = np.where(one_sim_data_u == t)[0]
            if curr_time_loc.size != 0:
                sampled_data[t,0:4] = one_sim_data_u[curr_time_loc[0],:]
            else:
                sampled_data[t,0:4] = sampled_data[t-1,0:4]
                sampled_data[t,0] = t

            # Find mean displacement from prev step
            if t > 0:
                curr_loc = sampled_data[t,2:4]
                prev_loc = sampled_data[t-1,2:4]
                diff = curr_loc - prev_loc
                summed = np.sqrt(diff[0]**2 + diff[1]**2)
                sampled_data[t,4] = summed

        one_users[idx] = sampled_data

    mean_spd_list = []
    for key in one_users:
        usr = one_users[key]
        mean_spd_list += [np.mean(usr[:,4])]
        
    return one_users