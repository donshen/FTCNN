import os
import json
import torch
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *
import matplotlib.pyplot as plt
from plotly.graph_objs import *
from scipy.special import softmax
from scipy.ndimage import zoom
from sklearn.metrics import confusion_matrix

class Inferer:
    def __init__(self, res=16):
        with open('struct_id.json') as fh:
            self.label_dict = json.load(fh)
        self.res = res
        
    def preproc(self, x, y, res):
        x = np.array(x)
        x = x.reshape(x.shape[0], 1, res, res, res)
        y = np.array(y)
        x_proc = torch.from_numpy(np.array(x)).float()
        y_proc = torch.from_numpy(np.array(y)).long()
        return x_proc, y_proc

    def prep_data_loader(self, x, y, res, shuffle=False):
        x, y = self.preproc(x, y, res)
        data = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(data, batch_size = len(data), shuffle=shuffle)
        return dataloader

    def make_inference_dataset_from_path(self, path, label):
        X = []
        y = []
        pts_file_list = os.listdir(path)
        pts_file_list= [file for file in pts_file_list if file[-3:] == 'pts']
        def get_key(file):
            return int(file.split('_')[-1][:-4])
        
        pts_file_list = sorted(pts_file_list, key=get_key)
        for pts_file in tqdm(pts_file_list):
            pts_file = os.path.join(path, pts_file)
            sim_box, _ = map_coord_from_pts(pts_file, self.res)
            FT_shifted = FT_calc(sim_box)
            FT_shifted = normalize_pts(FT_shifted)
            X.append(FT_shifted)
            y.append(label)
        return X, y

#     def infer(self, model, data_path, truth):
#         label = self.label_dict[truth]
#         X_infer, y_infer = self.make_inference_dataset_from_path(data_path, label)
#         infer_loader = self.prep_data_loader(X_infer, y_infer, self.res)    
#         for i, (X, y_true) in enumerate(infer_loader):      
#             infer_data = Variable(X.view(len(X), 1, self.res, self.res, self.res))
#             # Forward propagation
#             outputs = model.to('cpu')(infer_data)
#             # Get predictions from the maximum value
#             y_pred = torch.max(outputs.data, 1)[1]
#             acc = (y_pred == y_true).sum() / len(y_pred)
#             print(f'Ground Truth: {truth}; Prediction Accuracy: {acc.numpy()}')
#             conf_mat = confusion_matrix(y_true, y_pred)
#             scores = softmax(outputs.data, axis=1).numpy()
#         return scores, conf_mat


    def predict(self, infer_data_loader, model, device):
        y_true, y_pred = [], []
        acc = []
        scores = []
        for i, (X, y_true_batch) in enumerate(infer_data_loader):
            X = Variable(X.view(len(X), 1, self.res, self.res, self.res))
            output = model.to(device)(X)
            y_pred_batch = output.data.argmax(dim=1)
            acc_batch = (sum(y_pred_batch == y_true_batch) / len(y_pred_batch))
            print(f'Prediction Accuracy: {acc_batch}')
            scores.extend(F.softmax(output.data, dim=1).numpy())
        return np.mean(acc, 0), np.array(scores)

    def get_infer_data(self, scan_3d, l_window, step, label, out_res=16):
        infer_inputs = []
        infer_labels = []
        point_clouds = []
        voxels = []
        orig_res_x, orig_res_y, orig_res_z = scan_3d.shape
        resize_ratio = out_res / l_window
        nx, ny, nz = orig_res_x // step, orig_res_y // step, orig_res_z // step

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    region_3d = scan_3d[i * step: i * step + l_window,
                                       j * step: j * step + l_window,
                                       k * step: k * step + l_window]

                    ## Generate point clouds (minority region)
                    # coord = normalize_array(np.argwhere(scan_3d[:l_window,:l_window,:l_window] > 0))
                    # point_clouds.append(coord)

                    region_3d_resize = zoom(region_3d, (resize_ratio, resize_ratio, resize_ratio))
                    voxels.append(region_3d_resize)
                    infer_inputs.append(normalize_pts(FT_calc(region_3d_resize)))
                    infer_labels.append(label)

        return infer_inputs, infer_labels, voxels                
    
    def plot_softmax_scores(self, scores):    
        x = np.arange(1, len(scores) + 1) 
        fig, ax = plt.subplots(1, figsize=(6.5, 4))
        labels = {phase: val + 1 for phase, val in self.label_dict.items()}
        scores_plot = [scores[:, i] for i in range(len(self.label_dict))]
        ax.stackplot(x, *scores_plot, alpha = 0.88)
        ax.legend(labels,bbox_to_anchor=(1.31, 1.03), fontsize=16)
        ax.tick_params(direction='in', width=1, labelsize=16)
        ax.set_ylim([0, 1])
        ax.set_xlabel('Frame Index', fontsize=16)
        ax.set_ylabel('Softmax Probability', fontsize=16)
        plt.show()
