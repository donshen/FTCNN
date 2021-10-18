import os
import json
import torch
from tqdm import tqdm
from torch.autograd import Variable
from utils import *
import matplotlib.pyplot as plt
from plotly.graph_objs import *
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

class Inferer:
    def __init__(self, res=16):
        with open('struct_id.json') as fh:
            self.label_dict = json.load(fh)
        self.res = res
        
    def preproc(self, x, y, res):
        x = np.array(x).reshape(len(x), 1, res, res, res)
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

    def infer(self, model, data_path, truth):
        label = self.label_dict[truth]
        X_infer, y_infer = self.make_inference_dataset_from_path(data_path, label)
        infer_loader = self.prep_data_loader(X_infer, y_infer, self.res)    
        for i, (X, y_true) in enumerate(infer_loader):      
            infer_data = Variable(X.view(len(X), 1, self.res, self.res, self.res))
            # Forward propagation
            outputs = model.to('cpu')(infer_data)
            # Get predictions from the maximum value
            y_pred = torch.max(outputs.data, 1)[1]
            acc = (y_pred == y_true).sum() / len(y_pred)
            print(f'Ground Truth: {truth}; Prediction Accuracy: {acc.numpy()}')
            conf_mat = confusion_matrix(y_true, y_pred)
            scores = softmax(outputs.data, axis=1).numpy()
        return scores, conf_mat
    
    def plot_softmax_scores(self, scores):    
        x = np.arange(1, len(scores) + 1) 
        fig, ax = plt.subplots(1,figsize=(6.5, 4))
        labels = {phase: val+1 for phase, val in self.label_dict.items()}
        ax.stackplot(x, scores[:,0], scores[:,1], scores[:,2], 
                     scores[:,3], scores[:,4], scores[:,5], scores[:,6], scores[:,7], scores[:,8], alpha = 0.88)
        ax.legend(labels,bbox_to_anchor=(1.31, 1.03), fontsize = 16)
        ax.tick_params(direction='in',width = 1)
        ax.set_ylim([0, 1])
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Softmax Probability')
        plt.show()
