import json
import numpy as np

def check_acc(path_pred, path_true):
    preds, labels = [], []
    with open(path_true, 'r',) as fr:
        for line in fr.readlines():
            temp_label = json.loads(line)['label']
            labels.append(int(temp_label))
    with open(path_pred, 'r',) as fr:
        for line in fr.readlines():
            preds.append(int(line.strip('\n')))
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()

if __name__ == "__main__":
    path_pred = 'bustm_output/bert/test_labels'
    path_true = '../datasets/bustm/test_public.json'
    acc = check_acc(path_pred, path_true)
    print(acc)
