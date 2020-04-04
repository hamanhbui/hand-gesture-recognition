import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import glob, os

def read_data_sets(file_path):
    column_names = ['timestamp','x-axis', 'y-axis', 'z-axis','x1-axis', 'y1-axis', 'z1-axis','x2-axis', 'y2-axis', 'z2-axis','activity']
    data = pd.read_csv(file_path,header = None, names = column_names)
    data=data.sort_values(by="timestamp")
    return data

def convert_time_stamp(data):
    times=[0]
    for idx in range(len(data["timestamp"])-1):
        raw_1=data["timestamp"][idx]
        raw_2=data["timestamp"][idx+1]
        datetime_object_0 = datetime.strptime(raw_1, '%Y-%m-%d;%H:%M:%S.%f')
        datetime_object_1 = datetime.strptime(raw_2, '%Y-%m-%d;%H:%M:%S.%f')
        diff=datetime_object_1-datetime_object_0
        times.append(times[-1]+diff.total_seconds())
    
    data["timestamp"]=times
    return data

def interpolate_axis(time, data):
    time=time.tolist()
    data=data.tolist()
    func=interpolate.CubicSpline(time,data)
    time_new=np.arange(0, time[-1], 0.04)
    data_new=func(time_new)
    return data_new

def gen_label(data,new_data):
    rs=[]
    st_idx=0
    for idx in range(len(data["timestamp"])):
        if(data['activity'][idx]!=data['activity'][st_idx]):
            raw=[]
            raw.append(data["timestamp"][st_idx])
            raw.append(data["timestamp"][idx])
            raw.append(data["activity"][st_idx])
            rs.append(raw)
            st_idx=idx
        
    raw=[]
    raw.append(data["timestamp"][st_idx])
    raw.append(data["timestamp"][data.index[-1]])
    raw.append(data["activity"][st_idx])
    rs.append(raw)    
    elan_data = pd.DataFrame(rs,columns=['s_time','e_time','activity'])

    activity=[]
    e_idx=0

    for idx in range(len(new_data["timestamp"])):
        if (new_data["timestamp"][idx]>=elan_data["e_time"][e_idx]):
            e_idx+=1
        if(e_idx>=len(elan_data["e_time"])):
            break
        activity.append(elan_data["activity"][e_idx])

    new_data['activity']=activity
    return new_data

def drop_duplicate_timestamp(data):
    time_stamp=data["timestamp"].tolist()
    x_acc=data["x-axis"].tolist()
    y_acc=data["y-axis"].tolist()
    z_acc=data["z-axis"].tolist()
    x_gyr=data["x1-axis"].tolist()
    y_gyr=data["y1-axis"].tolist()
    z_gyr=data["z1-axis"].tolist()
    x_mag=data["x2-axis"].tolist()
    y_mag=data["y2-axis"].tolist()
    z_mag=data["z2-axis"].tolist()
    activity=data["activity"].tolist()
    idx=0
    while True:
        if(idx+1>=len(time_stamp)):
            break
        if(time_stamp[idx]==time_stamp[idx+1]):
            del time_stamp[idx]
            del x_acc[idx]
            del y_acc[idx]
            del z_acc[idx]
            del x_gyr[idx]
            del y_gyr[idx]
            del z_gyr[idx]
            del x_mag[idx]
            del y_mag[idx]
            del z_mag[idx]
            del activity[idx]
        else:
            idx+=1

    data={'timestamp':time_stamp,'x-axis':x_acc,'y-axis':y_acc,'z-axis':z_acc,
                    'x1-axis':x_gyr,'y1-axis':y_gyr,'z1-axis':z_gyr,
                    'x2-axis':x_mag,'y2-axis':y_mag,'z2-axis':z_mag,'activity':activity}
    return pd.DataFrame(data) 

def main():
    for filename in glob.iglob('original_dataset/**', recursive=True):
        if os.path.isfile(filename):
            data=read_data_sets(filename)
            data=drop_duplicate_timestamp(data)
            data=convert_time_stamp(data)
            times=data["timestamp"].tolist()
            data_timestamp=np.arange(0, times[-1], 0.04)
            data_x_axis=interpolate_axis(data["timestamp"],data["x-axis"])
            data_y_axis=interpolate_axis(data["timestamp"],data["y-axis"])
            data_z_axis=interpolate_axis(data["timestamp"],data["z-axis"])
            data_x1_axis=interpolate_axis(data["timestamp"],data["x1-axis"])
            data_y1_axis=interpolate_axis(data["timestamp"],data["y1-axis"])
            data_z1_axis=interpolate_axis(data["timestamp"],data["z1-axis"])
            data_x2_axis=interpolate_axis(data["timestamp"],data["x2-axis"])
            data_y2_axis=interpolate_axis(data["timestamp"],data["y2-axis"])
            data_z2_axis=interpolate_axis(data["timestamp"],data["z2-axis"])
            new_data={'timestamp':data_timestamp,'x-axis':data_x_axis,'y-axis':data_y_axis,'z-axis':data_z_axis,
                    'x1-axis':data_x1_axis,'y1-axis':data_y1_axis,'z1-axis':data_z1_axis,
                    'x2-axis':data_x2_axis,'y2-axis':data_y2_axis,'z2-axis':data_z2_axis}
            new_data = pd.DataFrame(new_data) 
            new_data=gen_label(data,new_data)
 
            list_str_filename=filename.split('/',1)
            filename="interpolated_dataset/"+list_str_filename[1]
            directory=filename.replace("data.txt","")
            if not os.path.exists(directory):
                os.makedirs(directory)
            data.to_csv(filename,header=None, sep='\t', encoding='utf-8', index=False, float_format='%.6f')

main()