import os
import torch
import pickle
from PIL import Image
from torch.utils.data import Dataset
import itertools
def _image_reader(path):
    return Image.open(path).convert('RGB')


class BaseDataSet(Dataset):

    image_cache = None

    def __init__(self, cfg, split):
        self.root_path = os.path.join(cfg.DATA.BASE_PATH, cfg.DATA.DATASET)#/home/data/datasets/FGdatas/FashionAI

        self.fnamelist = []
        filepath = os.path.join(self.root_path, cfg.DATA.PATH_FILE[split])# /home/data/datasets/FGdatas/FashionAI/filenames_train.txt(if split='TRAIN')
        assert os.path.exists(filepath), f"File {filepath} does not exist."
        with open(filepath, 'r') as f:
            for l in f:
                self.fnamelist.append(l.strip())#each image in filenames_train.txt append to fnamelist one by one(if split='TRAIN')
        
        self.image_loader = _image_reader
        
        if BaseDataSet.image_cache is None:
            with open(os.path.join(self.root_path, 'image_cache_darn_notest.pkl'), 'rb') as f:
                BaseDataSet.image_cache = pickle.load(f)

    def __len__(self):
        return self.fnamelist

    def __getitem__(self, index):
        # print(index)
        # path = os.path.join(self.root_path, self.fnamelist[index[0]])
        # assert os.path.exists(path), f"File {path} does not exist."

        # img = self.image_loader(path)

        path = self.fnamelist[index[0]]
        img = BaseDataSet.image_cache[path]

        return (img,) + index[1:] + (index[0],)

def tripletInfo_collate_fn(batch):
    xpn= batch
    n = len(xpn) // 3
    # avx : (a, vp)
    # avp : (a, vp)
    # avn : (a, vn)
    x, x_a, avx, x_id = zip(*xpn[:n])
    p, p_a, avp, p_id = zip(*xpn[n:2*n])
    n, n_a, avn, n_id = zip(*xpn[2*n:3*n])
    # avx : (a, v)
    # x和p的attr和value是一样的，neg和x的attr是一样的，value不同
    # print("anchor", avx)
    # print("pos", avp)
    # print("neg", avn)
    x_a = tuple([item[0] for item in avx])
    p_a = tuple([item[0] for item in avp])
    n_a = tuple([item[0] for item in avn])

    x_v = tuple([item[1] for item in avx])
    p_v = tuple([item[1] for item in avp])
    n_v = tuple([item[1] for item in avn])
    # print(x_a, x_v)
    av = avx + avp + avn

    flag = [torch.tensor(0) if x[0]==x[1] else torch.tensor(-10000.0) for x in itertools.product(av, av)]
    flag = torch.stack(flag,dim=0)
    
    de_flag = [torch.tensor(-10000.0) if x[0]==x[1] else torch.tensor(0) for x in itertools.product(av, av)]
    de_flag = torch.stack(de_flag,dim=0)

    flag = flag.reshape(len(av), -1)
    de_flag = de_flag.reshape(len(av), -1)

    return x, p, n, torch.LongTensor(x_a), flag, de_flag, x_a, p_a, n_a, x_v, p_v, n_v, x_id, p_id, n_id

def image_collate_fn(batch):
    # print(batch)
    x, a, v, x_id = zip(*batch)

    x_a = torch.LongTensor(a)
    x_v = torch.LongTensor(v)

    return x, x_a, x_v, a, v
