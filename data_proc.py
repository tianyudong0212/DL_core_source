import os
from typing import Tuple
import random
import torch
from torch.utils.data import Dataset

class FullGeneratedDataset(Dataset):
    def __init__(self, dir, split, tr_te_ratio=0.8) -> None:
        super().__init__()
        self.file_paths = os.listdir(dir)
        self.file_paths.sort()
        self.generated = []
        self.original = []
        for i in range(0, len(self.file_paths), 2):
            assert self.file_paths[i].endswith('generated.txt') \
                and self.file_paths[i+1].endswith('original.txt')
            gen_path = os.path.join(dir, self.file_paths[i])
            ori_path = os.path.join(dir, self.file_paths[i+1])
            
            with open(gen_path, 'r', encoding='utf-8') as fg:
                self.generated.append(fg.read())
            with open(ori_path, 'r', encoding='utf-8') as fo:
                self.original.append(fo.read())
        
        split_ind = int(len(self.generated) * tr_te_ratio)
        if split == 'train':
            self.generated = self.generated[: split_ind]
            self.original = self.original[: split_ind]
        elif split == 'test':
            self.generated = self.generated[split_ind: ]
            self.original = self.original[split_ind: ]

    def __getitem__(self, index) -> Tuple:
        return self.generated[index], self.original[index]
    
    def __len__(self):
        return len(self.generated)



class HybridDataset(Dataset):
    def __init__(self, dir, split, tr_te_ratio=0.8) -> None:
        super().__init__()
        self.file_paths = os.listdir(dir)
        self.file_paths.sort()
        self.generated = []
        self.original = []
        for i in range(0, len(self.file_paths), 2):
            assert self.file_paths[i].endswith('generatedAbstract.txt') \
                and self.file_paths[i+1].endswith('originalAbstract.txt')
            gen_path = os.path.join(dir, self.file_paths[i])
            ori_path = os.path.join(dir, self.file_paths[i+1])
            
            with open(gen_path, 'r', encoding='utf-8') as fg:
                self.generated.append(fg.read())
            with open(ori_path, 'r', encoding='utf-8') as fo:
                self.original.append(fo.read())
            
        split_ind = int(len(self.generated) * tr_te_ratio)
        if split == 'train':
            self.generated = self.generated[: split_ind]
            self.original = self.original[: split_ind]
        elif split == 'test':
            self.generated = self.generated[split_ind: ]
            self.original = self.original[split_ind: ]

    def __getitem__(self, index) -> Tuple:
        return self.generated[index], self.original[index]
    
    def __len__(self):
        return len(self.generated)


def contrastive_collate_fn(batch):
    generates = []
    originals = []
    ret = []
    for generated_sample, original_sample in batch:
        generates.append(generated_sample)
        originals.append(original_sample)
    for i, (generated_sample, original_sample) in enumerate(batch):
        temp_list = []
        temp_list.append(generated_sample)
        pos_smp = generates.pop(i)
        temp_list.append(pos_smp)
        generates.insert(i, pos_smp)
        temp_list.extend(originals)
        ret.append(temp_list)
    return ret


def generate_collate_fn(batch):
    pass





# tt = HybridDataset(
#     dir='GeneratedTextDetection-main/Dataset/Hybrid_AbstractDataset',
#     split='test'
# )
# print('done')