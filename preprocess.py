import os
import json
import h5py
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--raw_path', type=str, default='/mnt/ssd1/donny/test_pn/pointnet.pytorch/point_clouds/', help='path that contain {phase}/points/*.pts files')
parser.add_argument(
    '--proc_path', type=str, help='path that contain preprocessed training and test data splitted from FT and shifted voxilized point clouds, in .h5 format')
parser.add_argument(
    '-r', '--resolution', type=int, default=16, help='resolution of the voxels, i.e. if res=16, resulting voxels have dimensions of 16*16*16')
parser.add_argument(
    '-n', '--n_pts', type=int, default=3000, help='number of point clouds per structure, max now is 3000')
opt = parser.parse_args()


with open('struct_id.json') as fh:
    label_dict  = json.load(fh)
    
dirs = {phase: os.path.join(opt.raw_path, phase, 'points') for phase in label_dict}

data = []
data_labels = []
for phase in dirs:
    pts_file_list = os.listdir(dirs[phase])
    for i in tqdm(range(opt.n_pts), desc=f'Processing {phase} point clouds...'):
        pts_file = pts_file_list[i]
        pts_file = os.path.join(dirs[phase], pts_file)
        if os.path.isdir(pts_file):
            # skip directories
            continue
        try: 
            sim_box, _ = map_coord_from_pts(pts_file, opt.resolution)
            FT_shifted = FT_calc(sim_box)
            FT_shifted = normalize_pts(FT_shifted)
        except UnicodeDecodeError:
            pass
        data.append(FT_shifted)
        data_labels.append(label_dict[phase])
        
X_train, X_test, y_train, y_test = train_test_split(data, data_labels, test_size=0.2, random_state=99)

with h5py.File(opt.proc_path, "w") as hf:  
    hf.create_dataset('X_train', data=X_train)
    hf.create_dataset('y_train', data=y_train)
    hf.create_dataset('X_test', data=X_test)
    hf.create_dataset('y_test', data=y_test)