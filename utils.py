import numpy as np
import itertools
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from plotly.graph_objs import *
import plotly.graph_objects as go
from itertools import product
from sklearn.preprocessing import StandardScaler

def map_coord_from_pts(filename, res, fit=False):
    # Initialize 3D matrix with shape of (res, res, res)
    sim_box = np.zeros((res, res, res))
    # Read input
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.read().split('\n')[:-1]
        # Read box size
        
        coord = [[float(i) for i in line.split()] for line in lines]
        if fit:
            coord = normalize_pts_fit(coord)
        else:
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

def normalize_pts(X):
    # Min_max scaling
    X = np.array(X, dtype=np.float32)
    X_min, X_max = np.min(X), np.max(X)
    X_scaled = (X - X_min) / (X_max - X_min)
    return X_scaled

def normalize_pts_fit(X):
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

def get_bounding_box(L):
    # Get the bounding box of the N*3 coordinates for display
    rx, ry, rz = [0, L[0]], [0, L[1]], [0, L[2]]
    ZZ = np.asarray(list(product(rx, ry, rz)))
    pairs = [(0,2), (2,3), (3,1), (1,0),
             (0,1), (1,5), (5,4), (4,0),
             (1,3), (3,7), (7,5), (5,1),
             (2,6), (6,7), (7,3), (3,2),
             (0,2), (2,6), (6,4), (4,0),
             (0,4), (4,5), (5,1), (1,0)]
    
    Zx, Zy, Zz = ZZ[:,0], ZZ[:,1], ZZ[:,2]
    x_lines, y_lines, z_lines = [], [], []

    #create the coordinate list for the lines
    for p in pairs:
        for i in range(len(p)):
            x_lines.append(Zx[p[i]])
            y_lines.append(Zy[p[i]])
            z_lines.append(Zz[p[i]])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    return x_lines, y_lines, z_lines

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
    
    fig.add_trace(go.Scatter3d(x=get_bounding_box(np.max(coord, 0))[0],
                           y=get_bounding_box(np.max(coord, 0))[1],
                           z=get_bounding_box(np.max(coord, 0))[2],
                           mode='lines',
                           name='lines',
                           line=dict(width=3,color='black')))
    fig.show()
    

def show_slice(array):
    # show heatmap of a 2-D array
    fig, ax = plt.subplots(figsize=(5,5))
    im_region = ax.imshow(array)
    
class VisUnwrapClusters:   
    def __call__(self, voxel):
        self.voxel = voxel
        cluster_2 = self.find_clusters() 
        ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
        ax.voxels(self.voxel, facecolor='deepskyblue', alpha = 0.6, edgecolors='midnightblue')
        ax.voxels(cluster_2, facecolor='tomato', alpha = 0.6, edgecolors='maroon')
        ax.axis('off')
        plt.show()
           
    def is_out_of_bound(self, i, j, k):
        L1, L2, L3 = len(self.voxel), len(self.voxel[0]), len(self.voxel[0][0])
        return (i < 0 or i >= L1) or (j < 0 or j >= L2) or (k < 0 or k >= L3) 

    def dfs(self, i, j, k):
        # Check if current position is out-of-boundary, or has been visited, or is empty.
        if self.is_out_of_bound(i, j, k) or self.voxel[i][j][k] == 0:
            return
        # Mark as visited
        self.voxel[i][j][k] = 0
        # Get neighbor indexes (26 directions)
        dirs = (p for p in itertools.product([-1,0,1], repeat=3) if any(p))
        # Recurse through all neighbors
        for di, dj, dk in dirs:
            self.dfs(i+di, j+dj, k+dk)       

    def find_clusters(self):
        # Return two cluster for structurs such as double gyroid
        count = 0
        voxel_prev = deepcopy(self.voxel)
        if isinstance(self.voxel, np.ndarray): 
            self.voxel = self.voxel.tolist() 

        for i in range(len(self.voxel)):
            for j in range(len(self.voxel[0])):
                for k in range(len(self.voxel[0][0])):
                    if self.voxel[i][j][k] and count < 1:
                        count += 1
                        self.dfs(i, j, k)
                    else:
                        break
        self.voxel = np.array(self.voxel, dtype=int)
        voxel_rec = (self.voxel != voxel_prev) * 1  
        vexel_rec = np.array(voxel_rec) 
        return voxel_rec