import torch
import torch.utils.data as data_utl

import numpy as np
import random

import os
import pandas as pd
import lintel


class HMDB(data_utl.Dataset):

    def __init__(self, split_file, root, mode='rgb', length=16, model='2d', random=False, c2i={}):
        self.class_to_id = c2i
        self.id_to_class = []
        for i in range(len(c2i.keys())):
            for k in c2i.keys():
                if c2i[k] == i:
                    self.id_to_class.append(k)
        cid = 0
        self.data = []
        self.model = model
        self.size = 112

        with open(split_file, 'r') as f:
            for l in f.readlines():
                if len(l) <= 5:
                    continue
                v,c = l.strip().split(' ')
                # v_out = v.split('.')[0]+'_0.mp4'
                # print(f"v: {v} | v_out: {v_out} | c: {c}")
                # if v_out not in os.listdir("./ssd/hmdb"):
                #     cmd = f"cd ./ssd/hmdb && ffmpeg -i {v} -c:v copy -c:a copy {v_out} && cd .."
                #     os.system(cmd)
                # v = v.split('.')[0]+'_0.mp4'
                if c not in self.class_to_id:
                    self.class_to_id[c] = cid
                    self.id_to_class.append(c)
                    cid += 1
                self.data.append([os.path.join(root, v), self.class_to_id[c]])

        self.split_file = split_file
        self.root = root
        self.mode = mode
        self.length = length
        self.random = random

    def __getitem__(self, index):
        vid, cls = self.data[index]
        # with open(vid, 'rb') as f:
        #     enc_vid = f.read()

        # # df, w, h, _ = lintel.loadvid(enc_vid, should_random_seek=self.random, num_frames=self.length*2)
        # df = np.frombuffer(df, dtype=np.uint8)

        # w=w//2
        # h=h//2
        # # print(f"w {w} | h {h} | df\n{df}")
        
        # # center crop
        # if not self.random:
        #     i = int(round((h-self.size)/2.))
        #     j = int(round((w-self.size)/2.))
        #     df = np.reshape(df, newshape=(self.length*2, h*2, w*2, 3))[::2,::2,::2,:][:, i:-i, j:-j, :]
        # else:
        #     th = self.size
        #     tw = self.size
        #     i = random.randint(0, h - th) if h!=th else 0
        #     j = random.randint(0, w - tw) if w!=tw else 0
        #     print(f"h: {h} | w: {w}")
        #     print(f"th: {th} | tw: {tw}")
        #     print(f"i: {i} | j: {j}")
        #     df = np.reshape(df, newshape=(self.length*2, h*2, w*2, 3))[::2,::2,::2,:][:, i:i+th, j:j+tw, :]
            
        # if self.mode == 'flow':
        #     #print(df[:,:,:,1:].mean())
        #     #exit()
        #     # only take the 2 channels corresponding to flow (x,y)
        #     df = df[:,:,:,1:]
        #     if self.model == '2d':
        #         # this should be redone...
        #         # stack 10 along channel axis
        #         df = np.asarray([df[:10],df[2:12],df[4:14]]) # gives 3x10xHxWx2
        #         df = df.transpose(0,1,4,2,3).reshape(3,20,self.size,self.size).transpose(0,2,3,1)
            
        # print(f"w {w} | h {h} | df\n{df}")        
        # df = 1-2*(df.astype(np.float32)/255)

        # if self.model == '2d':
        #     # 2d -> return TxCxHxW
        #     return df.transpose([0,3,1,2]), cls
        # # 3d -> return CxTxHxW
        # return df.transpose([3,0,1,2]), cls


    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    for k in range(4):
        dataseta = HMDB(f'data/hmdb/split{k}_test.txt', '/ssd/hmdb/', model='2d', mode='flow', length=16, c2i={})
        df = pd.DataFrame.from_dict(dataseta.class_to_id, orient="index")
        df.to_csv(f"split{k}_mapping.csv")
        print(f"mapping for split {k}")
        for key, value in dataseta.class_to_id.items():
            print(f"{key}: {value}")
        print(f"")
        keys = list(dataseta.class_to_id.keys())
        keys.sort()
        for idx, key in enumerate(keys):
            print(f"{key}: {idx}")
