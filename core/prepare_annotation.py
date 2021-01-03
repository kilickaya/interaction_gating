import scipy.io as sio
import pickle as pkl
import numpy as np

def pkl_dump(data, path, is_highest=True):
    with open(path, 'wb') as f:
        if not is_highest:
            pkl.dump(data, f)
        else:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

annot = sio.loadmat('/var/scratch/mkilicka/code/where-is-interaction/main/data/anno_hico.mat')
annot = annot['dataset']

list_tr = annot['train_names'][0][0]
list_te = annot['val_names'][0][0]

n_tr = len(list_tr)
n_te = len(list_te)

# prepare list of images
x_tr = np.array([annot['train_names'][0][0][i][0][0] for i in range(n_tr)])
x_te = np.array([annot['val_names'][0][0][i][0][0] for i in range(n_te)])

y_tr = np.array([annot['anno_split_train'][0][0][:, i] for i in range(n_tr)])
y_te = np.array([annot['anno_split_val'][0][0][:, i] for i in range(n_te)])

print(x_tr.shape)
print(x_te.shape)
print(y_tr.shape)
print(y_te.shape)


pkl_dump((x_tr, y_tr, x_te, y_te), '/var/scratch/mkilicka/Datasets/Hico/annotation/anno_hico.pkl')