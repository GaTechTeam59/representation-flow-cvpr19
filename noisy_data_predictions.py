import os
import re
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torchvision
import flow_2p1d_resnets as model
import torch_hmdb_helper as helper
import pretrained_model


classes = ["cartwheel", "climb", "dive", "kick", "pullup", "run", "sit",
                      "situp", "somersault", "stand"]
class_labels = {classes[i]: i for i in range(len(classes))}

noise_types = ["fire", "fog", "grunge", "lights", "rain", "snow"]

meta_data = {
    "class": "",
    "class_id": 0,
    "noise-free": 0,
    "fire": 0,
    "fog": 0,
    "grunge": 0,
    "lights": 0,
    "rain": 0,
    "snow": 0
}

def parse_split_file(path):
    split_file_list = os.listdir(path)
    if ".DS_Store" in split_file_list:
        split_file_list.remove(".DS_Store")
    split_file_list.sort()
    file_info = dict()
    for split_file in split_file_list:
        match = re.search("^[a-zA-Z]*", split_file)
        class_label = match.group(0)
        with open(path + split_file, 'r') as f:
            for l in f.readlines():
                v, _ = l.strip().split(' ')
                file_info[v] = meta_data
                file_info[v]["class"] = class_label
                file_info[v]["class_id"] = class_labels[class_label]
    return file_info


def infer_actions(path_to_noisy_data_folders: str, net):
    data_folder_list = os.listdir(path_to_noisy_data_folders)
    if ".DS_Store" in data_folder_list:
        data_folder_list.remove(".DS_Store")
    for data_folder in data_folder_list:
        path_to_data = path_to_noisy_data_folders + data_folder
        confusion_matrix = infer_using_hmdb_utility(net, path_to_data)
        confusion_matrix_numpy = confusion_matrix.numpy()
        df = pd.DataFrame(confusion_matrix_numpy)  
        df.to_csv(f"{data_folder}.csv")
    return


def infer_using_hmdb_utility(net, root):
    length = 32
    size = 112
    annotation_path = "./data/hmdb/train_test_splits_subset"
    frames_per_clip = length
    step_between_clips = 10000*length
    fold = 1
    confusion_matrix = torch.zeros(10, 51)

    for train in [True, False]:
        data = helper.get_hmdb_data(
            size=size,
            root=root,
            annotation_path=annotation_path,
            frames_per_clip=frames_per_clip,
            step_between_clips=step_between_clips,
            fold=fold,
            train=train,
        )
        n_clips = data.__len__()

        for k in range(n_clips):
            sample = data.__getitem__(k)
            frames = sample[0].permute(3, 0, 1, 2).unsqueeze(0)
            net.eval()
            with torch.no_grad():
                predclass = torch.argmax(net(frames).squeeze())
                confusion_matrix[sample[2], predclass] += 1
    return confusion_matrix


if __name__ == '__main__':
    path_to_trained_weights = './hmdb-fof-model/hmdb_flow-of-flow_2p1d.pt'
    net = pretrained_model.load_model_weights(path_to_trained_weights)
    path_to_noisy_data_folders = '/users/jakeknigge/downloads/hmdb_overlay/'
    infer_actions(path_to_noisy_data_folders, net)
