import os
from collections import OrderedDict
import torch
import torchvision
import flow_2p1d_resnets as model
import torch_hmdb_helper as helper

os.environ['KMP_DUPLICATE_LIB_OK']='True'

classes = ["cartwheel", "climb", "dive", "kick", "pullup", "run", "sit",
                      "situp", "somersault", "stand"]
class_labels = {i: classes[i] for i in range(len(classes))}

def load_model_weights(path_to_trained_weights):
    _pretrained_model = torch.load(
        path_to_trained_weights, map_location=torch.device('cpu')
    )
    clean_keys = [
      key.replace("module.", "",) for key in _pretrained_model.keys()
    ]
    pretrained_model = OrderedDict(
        (clean_keys[idx], v) for idx, v in enumerate(_pretrained_model.values())
    )

    net = model.ResNet(model.BasicBlock, [3, 4, 6, 3], 51)
    net.load_state_dict(pretrained_model)
    return net

def _using_hmdb_utility(net):
    length = 32
    size = 112
    root = "./hmdb51_org_subset"
    annotation_path = "./data/hmdb/train_test_splits_subset"
    frames_per_clip = length*2
    step_between_clips = 1
    fold = 1
    train = False

    data = helper.get_hmdb_data(
        size=size,
        root=root,
        annotation_path=annotation_path,
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        fold=fold,
        train=train,
    )
    print(f"data size:\n{data}")

    pred_idx = [i + j for i in [0, 1, 2, 3]
                for j in [0, 1000, 2000, 3000, 4000]]
    pred_idx.sort()
    for k in pred_idx:
        sample = data.__getitem__(k)
        frames = sample[0].permute(3, 0, 1, 2).unsqueeze(0)
        net.eval()
        with torch.no_grad():
            predclass = torch.argmax(net(frames).squeeze())
            print((f"sample: {k} | "
                   f"predicted class label: {predclass} | "
                   f"sample class label: {sample[2]} "
                   f"({class_labels[sample[2]]})"))
    return


def _using_io_utility(net):
    filename = "./ssd/hmdb/Aerial_Cartwheel_Tutorial_By_Jujimufu_cartwheel_f_nm_np1_ri_med_0_0.mp4"
    av_in = torchvision.io.read_video(filename, start_pts=0, end_pts=2.0,
                                      pts_unit='sec')
    frames = helper.center_crop(av_in[0], 112)
    frames = frames.permute(3, 0, 1, 2).unsqueeze(0)
    net.eval()
    with torch.no_grad():
        predclass = torch.argmax(net(frames).squeeze())
        print(f"predicted class label: {predclass} | actual: cartwheel")
    return


if __name__ == '__main__':
    path_to_trained_weights = './hmdb-fof-model/hmdb_flow-of-flow_2p1d.pt'
    net = load_model_weights(path_to_trained_weights)
    _using_hmdb_utility(net)
    _using_io_utility(net)
