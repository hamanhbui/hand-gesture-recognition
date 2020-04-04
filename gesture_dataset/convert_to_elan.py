import pandas as pd
from datetime import datetime

def read_data_sets(file_path):
    column_names = ['timestamp','x-axis', 'y-axis', 'z-axis','x1-axis', 'y1-axis', 'z1-axis','x2-axis', 'y2-axis', 'z2-axis','activity']
    data = pd.read_csv(file_path,header = None, names = column_names)
    data=data.sort_values(by="timestamp")
    return data

data=read_data_sets("data.txt")
data.to_csv("data.txt",header=None, sep=',', encoding='utf-8', index=False, float_format='%.6f')

times=[0]
for idx in range(len(data["timestamp"])-1):
    raw_1=data["timestamp"][idx]
    raw_2=data["timestamp"][idx+1]
    datetime_object_0 = datetime.strptime(raw_1, '%Y-%m-%d;%H:%M:%S.%f')
    datetime_object_1 = datetime.strptime(raw_2, '%Y-%m-%d;%H:%M:%S.%f')
    diff=datetime_object_1-datetime_object_0
    times.append(times[-1]+diff.total_seconds())

data["timestamp"]=times
data.to_csv("data.csv",header=None, sep='\t', encoding='utf-8', index=False, float_format='%.6f')

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

df = pd.DataFrame(rs,columns=['Begin Time - ss.msec','End Time - ss.msec','default'])
print(df)
df.to_csv("label.txt", sep="\t", encoding='utf-8', index=False, float_format='%.6f')

# out_data=read_data_sets("out.txt")
# out_data.to_csv("out.csv",header=None, sep='\t', encoding='utf-8', index=False, float_format='%.6f')
