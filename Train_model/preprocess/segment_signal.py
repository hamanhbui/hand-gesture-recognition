import pandas as pd
import numpy as np
from scipy import stats
import glob, os

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

dt = 1.0/25
#Transition
F = np.array([[1, -dt], [0, 1]])
#Observation
H = np.array([1, 0]).reshape(1, 2)
Q = np.array([[0.001, 0.000], [0.000, 0.003]])
R = np.array([0.03]).reshape(1, 1)

# F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
# H = np.array([1, 0, 0]).reshape(1, 3)
# Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
# R = np.array([0.5]).reshape(1, 1)

def compute_kalman(measurements):
    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
    predictions = []
    for z in measurements:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)

    predictions=np.asarray(predictions).reshape(-1)
    return predictions

def read_data_sets(file_path):
    column_names = ['timestamp','x-axis', 'y-axis', 'z-axis','x1-axis', 'y1-axis', 'z1-axis','x2-axis', 'y2-axis', 'z2-axis','activity']
    data = pd.read_csv(file_path,header = None, names = column_names)
    return data

def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size/2)
        
def segment_signal(data,window_size,num_channels):
    data["x-axis"]=compute_kalman(data["x-axis"])
    data["y-axis"]=compute_kalman(data["y-axis"])
    data["z-axis"]=compute_kalman(data["z-axis"])
    data["x1-axis"]=compute_kalman(data["x1-axis"])
    data["y1-axis"]=compute_kalman(data["y1-axis"])
    data["z1-axis"]=compute_kalman(data["z1-axis"])
    data["x2-axis"]=compute_kalman(data["x2-axis"])
    data["y2-axis"]=compute_kalman(data["y2-axis"])
    data["z2-axis"]=compute_kalman(data["z2-axis"])
    segments = np.empty((0,window_size,num_channels))
    labels = np.empty((0))
    for (start, end) in windows(data["timestamp"], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        x1 = data["x1-axis"][start:end]
        y1 = data["y1-axis"][start:end]
        z1 = data["z1-axis"][start:end]
        x2 = data["x2-axis"][start:end]
        y2 = data["y2-axis"][start:end]
        z2 = data["z2-axis"][start:end]
        if(len(data["timestamp"][start:end]) == window_size):
            activity=stats.mode(data["activity"][start:end])[0][0]
            if(activity!="Start_gesture;" and activity!="Unknown;"):
                segments = np.vstack([segments,np.dstack([x,y,z,x1,y1,z1,x2,y2,z2])])
                labels = np.append(labels,activity)

    return segments, labels    

def main():
    np_set_x=np.empty((0,74,9))
    np_set_y=np.empty((0))

    for filename in glob.iglob('Dataset/origin/train_set/**', recursive=True):
        if os.path.isfile(filename):
            dataset = read_data_sets(file_path=filename)
            x, y = segment_signal(data=dataset,window_size=74,num_channels=9)
            np_set_x=np.append(np_set_x,x,axis=0)
            np_set_y=np.append(np_set_y,y,axis=0)
    
    print(np_set_x.shape)
    print(np_set_y.shape)
    np.save("Dataset/out/train_set/x.npy",np_set_x)
    np.save("Dataset/out/train_set/y.npy",np_set_y)

main()