from collections import OrderedDict
import numpy as np
import torch
from train_model_class import Model

class Args:
  def __init__(self, mode: str='rgb', exp_name: str="hmdb-test", batch_size: int=32, length: int=32, 
               learnable: str="[0,1,1,1,1]", niter: int=20, system: str="hmdb_subset", model: str="3d",
               resnet: int=18, pretrained: bool=False, pretrained_model=None, 
               dataset_root: str="./hmdb51_org_subset", split_path: str="./data/hmdb/train_test_splits_subset",
               num_epochs: int=2):
    self.mode = mode
    self.exp_name = exp_name
    self.batch_size = batch_size
    self.length = length
    self.learnable = learnable
    self.niter = niter
    self.system = system
    self.model = model
    self.resnet = resnet
    self.pretrained = pretrained
    self.pretrained_model = pretrained_model
    self.root = dataset_root
    self.annotation_path = split_path
    self.num_epochs = num_epochs


def make_pretrained_model(path_to_trained_weights):
    _pretrained_model = torch.load(path_to_trained_weights,
                                   map_location=torch.device(device)
                                  )
    clean_keys = [key.replace("module.", "",) for key in _pretrained_model.keys()]
    pretrained_model = OrderedDict(
       (clean_keys[idx], v) for idx, v in enumerate(_pretrained_model.values())
    )
    return pretrained_model

classes = ["cartwheel", "climb", "dive", "kick", "pullup",
           "run", "sit", "situp", "somersault", "stand"]

# args = Args(mode="rgb", exp_name="hmdb-test", batch_size=32, length=32, learnable="[0,1,1,1,1]", niter=20, system="hmdb", model="3d", resnet=18)
if __name__ == '__main__':
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using device: %s" % device)

    path_to_trained_weights = './hmdb-fof-model/hmdb_flow-of-flow_2p1d.pt'
    model_weights = make_pretrained_model(path_to_trained_weights)
    
    noise_types = [] # ['fire', 'fog', 'grunge', 'lights', 'rain', 'snow']

    # base, noiseless
    args = Args(resnet=34, pretrained=True, pretrained_model=model_weights)

    model = Model(device, args)

    model.train()
    
    # what a noisy trial would look like
    for noise in noise_types:
       args = Args(resnet=34, pretrained=True, pretrained_model=model_weights, dataset_root=f"./hmdb51_org_subset/{noise}")
       model = Model(device, args)
       
       model.train()
    
    
    
