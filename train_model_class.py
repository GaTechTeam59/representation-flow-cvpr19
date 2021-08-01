import os
import sys
import argparse
import inspect
import datetime
import json

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

#import models
import flow_2p1d_resnets
import testmodel
import testmodel_resnet

# device = torch.device('cuda')

##################
#
# Create model, dataset, and training setup
#
##################


class Model:
    def __init__(self, device, args):
        self.device = device
        if args.resnet == 18:
            #self.model = flow_2p1d_resnets.resnet18(pretrained=False, mode=args.mode, n_iter=args.niter,
            #                               learnable=eval(args.learnable), num_classes=400)
            self.model = flow_2p1d_resnets.resnet18(pretrained=args.pretrained, pretrained_model=args.pretrained_model, n_iter=args.niter,
                                           learnable=eval(args.learnable), num_classes=400)
        if args.resnet == 34:
            #self.model = flow_2p1d_resnets.resnet34(pretrained=False, mode=args.mode, n_iter=args.niter,
            #                               learnable=eval(args.learnable), num_classes=400)
            self.model = flow_2p1d_resnets.resnet34(pretrained=args.pretrained, 
                                                    pretrained_model=args.pretrained_model, # mode=args.mode,
                                                    n_iter=args.niter,
                                                    learnable=eval(args.learnable), 
                                                    num_classes=51)

        # if args.resnet == <something else>:
        #     self.model = test_model.ResNet(...)
        self.args = args

    def train(self):
        model = self.model
        args = self.args

        """torch.distributed.init_process_group(
            backend='BACKEND',
            init_method='env://'
        )
        model = nn.parallel.DistributedDataParallel(model).to(device)
        """
        model = nn.DataParallel(model).to(self.device)
        batch_size = args.batch_size
	
        """
        if args.system == 'hmdb':
            from hmdb_dataset import HMDB as DS
            dataseta = DS('data/hmdb/split0_train.txt', './ssd/hmdb/', model=args.model, mode=args.mode, length=args.length)
            dl = torch.utils.data.DataLoader(dataseta, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

            dataset = DS('data/hmdb/split0_test.txt', './ssd/hmdb/', model=args.model, mode=args.mode, length=args.length, c2i=dataseta.class_to_id)
            vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
            dataloader = {'train':dl, 'val':vdl}
        """
        
        if args.system == "hmdb_subset":
            from torch_hmdb_helper import get_hmdb_data
            length = args.length  # 32
            size = 112
            root = args.root # "./hmdb51_org_subset"
            annotation_path = args.annotation_path # "./data/hmdb/train_test_splits_subset"
            frames_per_clip = length*2
            step_between_clips = 1
            fold = 1
            
            # train set
            dataseta = get_hmdb_data(
                size=size, 
                root=root, 
                annotation_path=annotation_path, 
                frames_per_clip=frames_per_clip,
                step_between_clips=step_between_clips,
                fold=fold,
                train=True
            )
            dl = torch.utils.data.DataLoader(dataseta, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            
            # validation / test set
            dataset = get_hmdb_data(
                size=size, 
                root=root, 
                annotation_path=annotation_path, 
                frames_per_clip=frames_per_clip,
                step_between_clips=step_between_clips,
                fold=fold,
                train=False
            )
            vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            
            dataloader = {'train':dl, 'val':vdl}
        
        # scale lr for flow layer
        params = model.parameters()
        params = [p for p in params]


        #print([p for p in model.parameters() if (p == model.module.flow_layer.t).all()])
        #exit()

        lr = 0.01
        #solver = optim.SGD([{'params':params}, {'params':other, 'lr':0.01*lr}], lr=lr, weight_decay=1e-6, momentum=0.9)
        solver = optim.SGD([{'params':params}], lr=lr, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(solver, patience=7)


        #################
        #
        # Setup logs, store model code
        # hyper-parameters, etc...
        #
        #################
        #log_name = datetime.datetime.today().strftime('%m-%d-%H%M')+'-'+args.exp_name
        #os.mkdir('./logs')  # REF added
        #log_path = os.path.join('logs/',log_name)
        #os.mkdir(log_path)
        # os.system('cp * logs/'+log_name+'/')  # REF

        # deal with hyper-params...
        #with open(os.path.join(log_path,'params.json'), 'w') as out:
        #    hyper = vars(args)
        #    json.dump(hyper, out)
        log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[]}



        ###############
        #
        # Train the model and save everything
        #
        ###############
        num_epochs = args.num_epochs
        for epoch in range(num_epochs):

            for phase in ['train', 'val']:
                train = (phase=='train')
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                num_classes = 10 # REF TODO: this is hard-coded; is there a way to derive this?
                cm = np.zeros((num_classes, num_classes)).to(self.device)  # REF
                tloss = 0.
                acc = 0.
                tot = 0
                c = 0
                e=s=0

                with torch.set_grad_enabled(train):
                    # for vid, cls in dataloader[phase]:
                    for vid, _, cls in dataloader[phase]:
                        if c%200 == 0:
                            print('epoch',epoch,'iter',c)
                        #s=time.time()
                        #print('btw batch', (s-e)*1000)
                        vid = vid.to(self.device)
                        cls = cls.to(self.device)

                        outputs = model(vid)

                        pred = torch.max(outputs, dim=1)[1]
                        corr = torch.sum((pred == cls).int())

                        for i, j in zip(pred, cls):  # REF
                            cm[i, j] += 1
                        acc += corr.item()
                        tot += vid.size(0)
                        loss = F.cross_entropy(outputs, cls)
                        #print(loss)

                        if phase == 'train':
                            solver.zero_grad()
                            loss.backward()
                            solver.step()
                            log['iterations'].append(loss.item())

                        tloss += loss.item()
                        c += 1
                        #e=time.time()
                        #print('batch',batch_size,'time',(e-s)*1000)

                if phase == 'train':
                    log['epoch'].append(tloss/c)
                    log['train_acc'].append(acc/tot)
                    print('train loss',tloss/c, 'acc', acc/tot)
                    print("Confusion matrix")  # REF
                    print(f"{cm:.4f}")  # REF
                    print(f"train_acc: {np.trace(cm)/cm.sum():.4f}")
                else:
                    log['validation'].append(tloss/c)
                    log['val_acc'].append(acc/tot)
                    print('val loss', tloss/c, 'acc', acc/tot)
                    lr_sched.step(tloss/c)
                    print("Confusion matrix")  # REF
                    print(f"{cm:.4f}")  # REF
                    print(f"train_acc: {np.trace(cm)/cm.sum():.4f}")

        return cm
            #with open(os.path.join(log_path,'log.json'), 'w') as out:
            #    json.dump(log, out)
            #torch.save(model.state_dict(), os.path.join(log_path, 'hmdb_flow-of-flow_2p1d.pt'))


            #lr_sched.step()
