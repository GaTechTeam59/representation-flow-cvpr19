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
        classes = ["cartwheel", "climb", "dive", "kick", "pullup", "run", "sit",
                   "situp", "somersault", "stand"]
        class_labels = {i: classes[i] for i in range(len(classes))}

        pred_idx = [i + j for i in [0, 1, 2, 3] for j in [0, 1000, 2000, 3000, 4000]]
        pred_idx.sort()
        for k in pred_idx:
            sample = self.data.__getitem__(k)
            frames = sample[0].permute(3, 0, 1, 2).unsqueeze(0)

            """
            net.eval()
            with torch.no_grad():
                predclass = torch.argmax(net(frames).squeeze())
                print((f"sample: {k} | "
                       f"predicted class label: {predclass} | "
                    f"sample class label: {sample[2]} ({class_labels[sample[2]]})"))
                    
            """

        print(data)
        frames = data.__getitem__(0)[0]
        print(data.__getitem__(0))
        frames_ = frames.permute(3, 0, 1, 2).unsqueeze(0)
        print(frames.shape)
        print(frames_.shape)
        return frames_

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    DS = HMDB
    dataseta = DS('data/hmdb/split0_train.txt', '/ssd/hmdb/', model='2d', mode='flow', length=16)
    dataset = DS('data/hmdb/split0_test.txt', '/ssd/hmdb/', model='2d', mode='rgb', length=16, c2i=dataseta.class_to_id)

    for i in range(len(dataseta)):
        print(dataseta[i][0].shape)

