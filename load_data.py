import torch
from torchvision import datasets
import torchvision.transforms as transforms

length = 32
size = 112
root = "/Users/jakeknigge/docs/github-clones/representation-flow-cvpr19/hmdb51_org_subset"
annotation_path = "/Users/jakeknigge/downloads/testTrainMulti_7030_splits-test"
frames_per_clip = length*2
step_between_clips = 1
fold = 1
train = False

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


data = datasets.HMDB51(
    root=root,
    annotation_path=annotation_path,
    frames_per_clip=frames_per_clip,
    step_between_clips=step_between_clips,
    fold=fold,
    train=train,
    transform=transforms.Lambda(lambda frames: center_crop(frames, output_size=size))
)

print(data)

frames = data.__getitem__(0)[0]
