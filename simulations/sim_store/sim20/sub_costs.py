import numpy as np

def obtain_sub_costs(data):
    
    SG_cost_stores = data["SG"]
    try:
        Batch_cost_stores=data["Batch"]
    except:
        Batch_cost_stores = data["SG"]
    Myopic_cost_stores= data["Myopic"]
    Naive_cost_stores=data["Naive"]
    Cloud_cost_stores=data["Cloud"]
    
    num_trials = len(Batch_cost_stores)

    SG_total_sum = np.zeros(num_trials)
    SG_placement_sum = np.zeros(num_trials)
    SG_bw_sum = np.zeros(num_trials)
    SG_UE_sum = np.zeros(num_trials)

    Batch_total_sum = np.zeros(num_trials)
    Batch_placement_sum = np.zeros(num_trials)
    Batch_bw_sum = np.zeros(num_trials)
    Batch_UE_sum = np.zeros(num_trials)

    Myopic_total_sum = np.zeros(num_trials)
    Myopic_placement_sum = np.zeros(num_trials)
    Myopic_bw_sum = np.zeros(num_trials)
    Myopic_UE_sum = np.zeros(num_trials)

    Naive_total_sum = np.zeros(num_trials)
    Naive_placement_sum = np.zeros(num_trials)
    Naive_bw_sum = np.zeros(num_trials)
    Naive_UE_sum = np.zeros(num_trials)

    Cloud_total_sum = np.zeros(num_trials)
    Cloud_placement_sum = np.zeros(num_trials)
    Cloud_bw_sum = np.zeros(num_trials)
    Cloud_UE_sum = np.zeros(num_trials)

    for i in range(num_trials):

        SG_total_sum[i] += SG_cost_stores[i].all_cost_cumulative["total_cost"][-1]
        SG_placement_sum[i] += SG_cost_stores[i].all_cost_cumulative["placement_cost"][-1]
        SG_bw_sum[i] += SG_cost_stores[i].all_cost_cumulative["srv_bw_cost"][-1]
        SG_bw_sum[i] += SG_cost_stores[i].all_cost_cumulative["mig_bw_cost"][-1]
        SG_UE_sum[i] += SG_cost_stores[i].all_cost_cumulative["thruput_cost"][-1]
        SG_UE_sum[i] += SG_cost_stores[i].all_cost_cumulative["latency_cost"][-1]

        Batch_total_sum[i] += Batch_cost_stores[i].all_cost_cumulative["total_cost"][-1]
        Batch_placement_sum[i] += Batch_cost_stores[i].all_cost_cumulative["placement_cost"][-1]
        Batch_bw_sum[i] += Batch_cost_stores[i].all_cost_cumulative["srv_bw_cost"][-1]
        Batch_bw_sum[i] += Batch_cost_stores[i].all_cost_cumulative["mig_bw_cost"][-1]
        Batch_UE_sum[i] += Batch_cost_stores[i].all_cost_cumulative["thruput_cost"][-1]
        Batch_UE_sum[i] += Batch_cost_stores[i].all_cost_cumulative["latency_cost"][-1]

        Myopic_total_sum[i] += Myopic_cost_stores[i].all_cost_cumulative["total_cost"][-1]
        Myopic_placement_sum[i] += Myopic_cost_stores[i].all_cost_cumulative["placement_cost"][-1]
        Myopic_bw_sum[i] += Myopic_cost_stores[i].all_cost_cumulative["srv_bw_cost"][-1]
        Myopic_bw_sum[i] += Myopic_cost_stores[i].all_cost_cumulative["mig_bw_cost"][-1]
        Myopic_UE_sum[i] += Myopic_cost_stores[i].all_cost_cumulative["thruput_cost"][-1]
        Myopic_UE_sum[i] += Myopic_cost_stores[i].all_cost_cumulative["latency_cost"][-1]

        Naive_total_sum[i] += Naive_cost_stores[i].all_cost_cumulative["total_cost"][-1]
        Naive_placement_sum[i] += Naive_cost_stores[i].all_cost_cumulative["placement_cost"][-1]
        Naive_bw_sum[i] += Naive_cost_stores[i].all_cost_cumulative["srv_bw_cost"][-1]
        Naive_bw_sum[i] += Naive_cost_stores[i].all_cost_cumulative["mig_bw_cost"][-1]
        Naive_UE_sum[i] += Naive_cost_stores[i].all_cost_cumulative["thruput_cost"][-1]
        Naive_UE_sum[i] += Naive_cost_stores[i].all_cost_cumulative["latency_cost"][-1]

        Cloud_total_sum[i] += Cloud_cost_stores[i].all_cost_cumulative["total_cost"][-1]
        Cloud_placement_sum[i] += Cloud_cost_stores[i].all_cost_cumulative["placement_cost"][-1]
        Cloud_bw_sum[i] += Cloud_cost_stores[i].all_cost_cumulative["srv_bw_cost"][-1]
        Cloud_bw_sum[i] += Cloud_cost_stores[i].all_cost_cumulative["mig_bw_cost"][-1]
        Cloud_UE_sum[i] += Cloud_cost_stores[i].all_cost_cumulative["thruput_cost"][-1]
        Cloud_UE_sum[i] += Cloud_cost_stores[i].all_cost_cumulative["latency_cost"][-1]
        
    SG_cost = [np.mean(SG_total_sum), np.mean(SG_placement_sum),np.mean(SG_bw_sum), np.mean(SG_UE_sum)]
    Batch_cost = [np.mean(Batch_total_sum), np.mean(Batch_placement_sum),np.mean(Batch_bw_sum), np.mean(Batch_UE_sum)]
    Myopic_cost = [np.mean(Myopic_total_sum), np.mean(Myopic_placement_sum),np.mean(Myopic_bw_sum), np.mean(Myopic_UE_sum)]
    Naive_cost = [np.mean(Naive_total_sum), np.mean(Naive_placement_sum),np.mean(Naive_bw_sum), np.mean(Naive_UE_sum)]
    Cloud_cost = [np.mean(Cloud_total_sum), np.mean(Cloud_placement_sum),np.mean(Cloud_bw_sum), np.mean(Cloud_UE_sum)]
    return SG_cost, Batch_cost, Myopic_cost, Naive_cost, Cloud_cost

def obtain_sub_vars(data):
    
    SG_cost_stores = data["SG"]
    try:
        Batch_cost_stores=data["Batch"]
    except:
        Batch_cost_stores = data["SG"]
    Myopic_cost_stores= data["Myopic"]
    Naive_cost_stores=data["Naive"]
    Cloud_cost_stores=data["Cloud"]
    
    num_trials = len(Batch_cost_stores)

    SG_total_sum = np.zeros(num_trials)
    SG_placement_sum = np.zeros(num_trials)
    SG_bw_sum = np.zeros(num_trials)
    SG_UE_sum = np.zeros(num_trials)

    Batch_total_sum = np.zeros(num_trials)
    Batch_placement_sum = np.zeros(num_trials)
    Batch_bw_sum = np.zeros(num_trials)
    Batch_UE_sum = np.zeros(num_trials)

    Myopic_total_sum = np.zeros(num_trials)
    Myopic_placement_sum = np.zeros(num_trials)
    Myopic_bw_sum = np.zeros(num_trials)
    Myopic_UE_sum = np.zeros(num_trials)

    Naive_total_sum = np.zeros(num_trials)
    Naive_placement_sum = np.zeros(num_trials)
    Naive_bw_sum = np.zeros(num_trials)
    Naive_UE_sum = np.zeros(num_trials)

    Cloud_total_sum = np.zeros(num_trials)
    Cloud_placement_sum = np.zeros(num_trials)
    Cloud_bw_sum = np.zeros(num_trials)
    Cloud_UE_sum = np.zeros(num_trials)

    for i in range(num_trials):

        SG_total_sum[i] += SG_cost_stores[i].all_cost_cumulative["total_cost"][-1]
        SG_placement_sum[i] += SG_cost_stores[i].all_cost_cumulative["placement_cost"][-1]
        SG_bw_sum[i] += SG_cost_stores[i].all_cost_cumulative["srv_bw_cost"][-1]
        SG_bw_sum[i] += SG_cost_stores[i].all_cost_cumulative["mig_bw_cost"][-1]
        SG_UE_sum[i] += SG_cost_stores[i].all_cost_cumulative["thruput_cost"][-1]
        SG_UE_sum[i] += SG_cost_stores[i].all_cost_cumulative["latency_cost"][-1]

        Batch_total_sum[i] += Batch_cost_stores[i].all_cost_cumulative["total_cost"][-1]
        Batch_placement_sum[i] += Batch_cost_stores[i].all_cost_cumulative["placement_cost"][-1]
        Batch_bw_sum[i] += Batch_cost_stores[i].all_cost_cumulative["srv_bw_cost"][-1]
        Batch_bw_sum[i] += Batch_cost_stores[i].all_cost_cumulative["mig_bw_cost"][-1]
        Batch_UE_sum[i] += Batch_cost_stores[i].all_cost_cumulative["thruput_cost"][-1]
        Batch_UE_sum[i] += Batch_cost_stores[i].all_cost_cumulative["latency_cost"][-1]

        Myopic_total_sum[i] += Myopic_cost_stores[i].all_cost_cumulative["total_cost"][-1]
        Myopic_placement_sum[i] += Myopic_cost_stores[i].all_cost_cumulative["placement_cost"][-1]
        Myopic_bw_sum[i] += Myopic_cost_stores[i].all_cost_cumulative["srv_bw_cost"][-1]
        Myopic_bw_sum[i] += Myopic_cost_stores[i].all_cost_cumulative["mig_bw_cost"][-1]
        Myopic_UE_sum[i] += Myopic_cost_stores[i].all_cost_cumulative["thruput_cost"][-1]
        Myopic_UE_sum[i] += Myopic_cost_stores[i].all_cost_cumulative["latency_cost"][-1]

        Naive_total_sum[i] += Naive_cost_stores[i].all_cost_cumulative["total_cost"][-1]
        Naive_placement_sum[i] += Naive_cost_stores[i].all_cost_cumulative["placement_cost"][-1]
        Naive_bw_sum[i] += Naive_cost_stores[i].all_cost_cumulative["srv_bw_cost"][-1]
        Naive_bw_sum[i] += Naive_cost_stores[i].all_cost_cumulative["mig_bw_cost"][-1]
        Naive_UE_sum[i] += Naive_cost_stores[i].all_cost_cumulative["thruput_cost"][-1]
        Naive_UE_sum[i] += Naive_cost_stores[i].all_cost_cumulative["latency_cost"][-1]

        Cloud_total_sum[i] += Cloud_cost_stores[i].all_cost_cumulative["total_cost"][-1]
        Cloud_placement_sum[i] += Cloud_cost_stores[i].all_cost_cumulative["placement_cost"][-1]
        Cloud_bw_sum[i] += Cloud_cost_stores[i].all_cost_cumulative["srv_bw_cost"][-1]
        Cloud_bw_sum[i] += Cloud_cost_stores[i].all_cost_cumulative["mig_bw_cost"][-1]
        Cloud_UE_sum[i] += Cloud_cost_stores[i].all_cost_cumulative["thruput_cost"][-1]
        Cloud_UE_sum[i] += Cloud_cost_stores[i].all_cost_cumulative["latency_cost"][-1]
        
    cost_var_dict = {}
    cost_var_dict['SG'] = np.std(SG_total_sum)
    cost_var_dict['Batch'] = np.std(Batch_total_sum)
    cost_var_dict['Myopic'] = np.std(Myopic_total_sum)
    cost_var_dict['Naive'] = np.std(Naive_total_sum)
    cost_var_dict['Cloud'] = np.std(Cloud_total_sum)
    
    lat_var_dict = {}
    lat_var_dict['SG'] = np.std(SG_UE_sum)
    lat_var_dict['Batch'] = np.std(Batch_UE_sum)
    lat_var_dict['Myopic'] = np.std(Myopic_UE_sum)
    lat_var_dict['Naive'] = np.std(Naive_UE_sum)
    lat_var_dict['Cloud'] = np.std(Cloud_UE_sum)

    return cost_var_dict, lat_var_dict
    