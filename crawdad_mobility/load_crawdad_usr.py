import pandas as pd
import numpy as np
import sqlite3

def load_usr(db, ts_min_size = 5):
    cnx = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT * FROM locationTable", cnx)

    cnx.close()

    df = df[df['_activity'] == 2]
    df = df[df['_latitude']>0]
    df = df[df['_longitude']>0]
    
    times = np.array(df['_time_location'])
    date_list = []
    hr_list = []
    min_list = []
    ts_list = []
    

    for t in times:
        date_list+= [t[4:-9]]
        hr_list+= [int(t[8:-7])]
        min_list += [int(t[10:-5])]
        ts_list += [np.round((hr_list[-1]*60 + min_list[-1])/ts_min_size)]
        
    df['Date'] = date_list
    df['Hr'] = hr_list
    df['Min'] = min_list
    df['ts'] = ts_list

    
    lat = np.array(df['_latitude']*10e-7)
    long = np.array(df['_longitude']*10e-7)

    mean_lat = np.mean(lat)
    mean_long =  np.mean(long)

    dx,dy = longlat2km(lat,long,mean_lat,mean_long)

    df['x_loc'] = dx/1.6
    df['y_loc']= dy/1.6 # Change to miles

    df = df.drop(['_node_id','_latitude_gps','_longitude_gps','_latitude_wifi','_longitude_wifi',
             '_altitude','_accuracy','_accuracy_gps','_accuracy_wifi','_place_name','_place_comment',
                 '_latitude','_longitude','_activity','_time_location'], axis=1)

    
    traces = {}
    
    for date in date_list:
        temp_df = np.array(df[df["Date"]==date])[:,3:]
        
        # Normalize X and Y here to be average for that date (mean = 0)
        ts = temp_df[:,0]
        x = temp_df[:,1] # - np.mean(np.unique(temp_df[:,1]))
        y = temp_df[:,2] # - np.mean(np.unique(temp_df[:,2]))
        
        temp_df[:,1] = x
        temp_df[:,2] = y
        
        temp_final = np.empty([0,3])
        temp_final = np.append(temp_final,np.reshape(temp_df[0],[1,3]),axis=0)
                
        for i in range(ts.shape[0]-1):
            t2 = int(ts[i+1])
            t1 = int(ts[i])
            
            row_2 = temp_df[i+1,:]
            row_1 = temp_df[i,:]
            
            t_diff = t2 - t1
            
            # Extrapolate
            if t_diff > 1:
                times = range(t1+1,t2+1)
                del_x = x[i+1]-x[i]
                del_y = y[i+1]-y[i]
                
                ratio_count = 1
                
                for t in times:
                    temp_x = x[i] + (ratio_count/t_diff) * del_x
                    temp_y = y[i] + (ratio_count/t_diff) * del_y
                    
                    temp = np.array([[t,temp_x,temp_y]])
                    temp_final = np.append(temp_final,temp,axis=0)
                    
                    ratio_count += 1
                
            # Delete one of the cases    
            elif t_diff == 0:
                continue
                
            # Proceed as normal    
            elif t_diff == 1:
                temp_final = np.append(temp_final,np.reshape(row_2,[1,3]),axis=0)
            
        traces[date] = temp_final
    
    return df, traces

def longlat2km(lat1, long1, lat2, long2):
    D = 40075 # km
    dy = (lat1-lat2) * 111.32
    dx = (long1 - long2)*(D * np.cos((lat1+lat2)/(2 * 180 )))/(360)
    
    return dy, dx