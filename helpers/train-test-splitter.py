import random

def _generate_train_test_split(ratio_train=0.8):
    random.seed(2)
    master = []
    label_dictionary = dict()

    with open("./data/hmdb/split0_master.txt", 'r') as f:
        for _idx, line in enumerate(f.readlines()):
            video, label = line.strip().split(' ')
            master.append([video, label])
            label_dictionary.setdefault(label, [0, [video]])
            label_dictionary[label][0] += 1
            if label_dictionary[label][0] > 1:
                label_dictionary[label][1].append(video)

    train_text = ""
    test_text = ""
    for key, value in label_dictionary.items():
        idx_train = int(ratio_train * value[0]) + 1
        random.shuffle(value[1])
        train = value[1][0:idx_train]
        test = value[1][idx_train::]
        for item in train:
            train_text += f"{item} {key}\n"
        for item in test: 
            test_text += f"{item} {key}\n"
    return train_text, test_text


def _generate_text(filename: str, text_data: str):
    with open(f"./data/hmdb/{filename}", "w") as file:
        file.write(text_data)


if __name__ == '__main__':
    train_text, test_text = _generate_train_test_split()
    _generate_text("split0_train.txt", train_text)
    _generate_text("split0_test.txt", test_text)
