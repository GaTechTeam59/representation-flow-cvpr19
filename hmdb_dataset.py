import torch
import torch.utils.data as data_utl

import torch
from torchvision import datasets
import torchvision.transforms as transforms

def center_crop(img: torch.Tensor, output_size: int):
    img = torch.transpose(torch.transpose(img, 2, 3), 1, 2)
    cropped_list = []
    for i in range(img.shape[0]):
        pil_img = transforms.functional.to_pil_image(img[i, :, :, :])
        cropped_pil = transforms.functional.center_crop(pil_img, output_size)
        cropped = transforms.functional.to_tensor(cropped_pil)
        cropped = torch.transpose(torch.transpose(cropped, 1, 0), 2, 1)
        cropped_list.append(cropped)
    stacked_cropped_list = torch.stack(cropped_list)
    return stacked_cropped_list


class HMDB(data_utl.Dataset):
    # test or train
    def __init__(self, stage):
        length = 32
        size = 112
        root = "/Users/jakeknigge/docs/github-clones/representation-flow-cvpr19/hmdb51_org_subset"
        annotation_path = "/Users/jakeknigge/downloads/testTrainMulti_7030_splits-test"
        frames_per_clip = length * 2
        step_between_clips = 1
        fold = 1
        self.data = datasets.HMDB51(
            root=root,
            annotation_path=annotation_path,
            frames_per_clip=frames_per_clip,
            step_between_clips=step_between_clips,
            fold=fold,
            train= stage,
            transform=transforms.Lambda(lambda frames: center_crop(frames, output_size=size))
        )



        ### Classes definition also required


    def __getitem__(self, index):
        vid, cls = self.data[index]
        with open(vid, 'rb') as f:
            enc_vid = f.read()

        df, w, h, _ = lintel.loadvid(enc_vid, should_random_seek=self.random, num_frames=self.length*2)
        df = np.frombuffer(df, dtype=np.uint8)

        w=w//2
        h=h//2
        print(f"w {w} | h {h} | df\n{df}")
        
        # center crop
        if not self.random:
            i = int(round((h-self.size)/2.))
            j = int(round((w-self.size)/2.))
            df = np.reshape(df, newshape=(self.length*2, h*2, w*2, 3))[::2,::2,::2,:][:, i:-i, j:-j, :]
        else:
            th = self.size
            tw = self.size
            i = random.randint(0, h - th) if h!=th else 0
            j = random.randint(0, w - tw) if w!=tw else 0
            print(f"h: {h} | w: {w}")
            print(f"th: {th} | tw: {tw}")
            print(f"i: {i} | j: {j}")
            df = np.reshape(df, newshape=(self.length*2, h*2, w*2, 3))[::2,::2,::2,:][:, i:i+th, j:j+tw, :]
            
        if self.mode == 'flow':
            #print(df[:,:,:,1:].mean())
            #exit()
            # only take the 2 channels corresponding to flow (x,y)
            df = df[:,:,:,1:]
            if self.model == '2d':
                # this should be redone...
                # stack 10 along channel axis
                df = np.asarray([df[:10],df[2:12],df[4:14]]) # gives 3x10xHxWx2
                df = df.transpose(0,1,4,2,3).reshape(3,20,self.size,self.size).transpose(0,2,3,1)
            
        print(f"w {w} | h {h} | df\n{df}")        
        df = 1-2*(df.astype(np.float32)/255)

        if self.model == '2d':
            # 2d -> return TxCxHxW
            return df.transpose([0,3,1,2]), cls
        # 3d -> return CxTxHxW
        return df.transpose([3,0,1,2]), cls


    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    DS = HMDB
    dataseta = DS('data/hmdb/split0_train.txt', './ssd/hmdb/', model='2d', mode='flow', length=16)
    dataset = DS('data/hmdb/split0_test.txt', './ssd/hmdb/', model='2d', mode='rgb', length=16, c2i=dataseta.class_to_id)

    for i in range(len(dataseta)):
        print(dataseta[i][0].shape)

