import numpy as np
import matplotlib.pyplot as plt
from plotly.graph_objs import *
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

def map_coord_from_pts(filename, res):
    # Initialize 3D matrix with shape of (res, res, res)
    sim_box = np.zeros((res, res, res))
    # Read input
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.read().split('\n')[:-1]
        # Read box size
        
        coord = [[float(i) for i in line.split()] for line in lines]
        coord = normalize_pts(coord)
        
        for i in range(coord.shape[0]):
            index_x = int(coord[i, 0] * res)
            index_y = int(coord[i, 1] * res)
            index_z = int(coord[i, 2] * res)

            if index_x >= res: index_x -= res
            if index_y >= res: index_y -= res
            if index_z >= res: index_z -= res
                
            sim_box[index_x, index_y, index_z] += 1
    return sim_box, coord

# def normalize_pts(X):
#     # Min_max scaling
#     X = np.array(X, dtype=np.float32)
#     X_min, X_max = np.min(X), np.max(X)
#     X_scaled = (X - X_min) / (X_max - X_min)
#     return X_scaled

def normalize_pts(X):
    # Min_max scaling in all three dimensions, resulting coordinates have a unit cubic boundary
    X = np.array(X, dtype=np.float32)
    for i in range(3):
        X_min_i, X_max_i = np.min(X[:, i]), np.max(X[:, i])
        X[:, i] = (X[:, i] - X_min_i) / (X_max_i - X_min_i)   
    return X

def standardize_pts(X):
    scaler = StandardScaler()
    X = np.array(X, dtype=np.float32)
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    return X

def plot_FT(FT_shifted):
    fig, axs = plt.subplots(1,3,figsize = (10, 30))
    fig.subplots_adjust(hspace =.1, wspace=.0)
    axs = axs.ravel()

    for i in range(3):
        axs[i].imshow(np.sum(FT_shifted,i))
        _ = axs[i].yaxis.set_ticklabels([])
        _ = axs[i].xaxis.set_ticklabels([])
        axs[i].tick_params(axis='both', which='both', bottom=True, top=True, right=True, direction='in')
        
def FT_calc(sim_box, keep=1):
    FT = np.fft.fftn(sim_box) 
    FTsort = np.sort(np.abs(FT.reshape(-1)))
    thresh = FTsort[int(np.floor((1 - keep) * len(FTsort)))]
    ind = np.abs(FT) > thresh
    FT_low = FT * ind
    FT_shifted = np.abs(np.fft.fftshift(FT_low))
    return FT_shifted
    
def display_minority(coord):
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
    fig = go.Figure(data =[go.Scatter3d(x = x,
                                       y = y,
                                       z = z,
                                       mode ='markers', 
                                       marker = dict(
                                         size = 2,
                                         color = 'orange',
                                         opacity = 0.4 
                                       ))])
    fig.update_layout(scene=dict(xaxis=dict(showticklabels=False, visible=False),
                                 yaxis=dict(showticklabels=False, visible=False),
                                 zaxis=dict(showticklabels=False, visible=False)),
                      template='plotly_white',
                      showlegend=False,
                      scene_camera_projection=dict(type='orthographic'),
                      margin=dict(l=0, r=0, t=0, b=0))
    fig.show()