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


def get_hmdb_data(size, root, annotation_path, frames_per_clip,
                  step_between_clips, fold, train):
    data = datasets.HMDB51(
        root=root,
        annotation_path=annotation_path,
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        fold=fold,
        train=train,
        transform=transforms.Lambda(
            lambda frames: center_crop(frames, output_size=size)
        )
    )
    return data


if __name__ == '__main__':
    length = 30
    size = 112
    root = "./hmdb51_org_subset"
    annotation_path = "./data/hmdb/train_test_splits_subset"
    frames_per_clip = length*1
    step_between_clips = 1
    fold = 1
    train = False

    data = get_hmdb_data(
        size=size,
        root=root,
        annotation_path=annotation_path,
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        fold=fold,
        train=train
    )

    dl = torch.utils.data.DataLoader(data, batch_size=5, shuffle=True,
                                     num_workers=0, pin_memory=True)

    for idx, (batch_video_tensors, _, batch_class_labels) in enumerate(dl):
        print(f"batch {idx}")
        print(f"batch_video_tensors: {batch_video_tensors.shape}")
        print(f"batch_class_labels: {batch_class_labels}\n")
