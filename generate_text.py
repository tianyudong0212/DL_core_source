from transformers import AutoTokenizer, T5EncoderModel, AutoModelForSeq2SeqLM
import torch
import random

from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import os
from tqdm import tqdm
from collections import deque
from transformers import AutoTokenizer, T5EncoderModel, T5ForConditionalGeneration
import pickle

from data_proc import FullGeneratedDataset, HybridDataset, contrastive_collate_fn
from utils_fn import contrastive_loss



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'

def perturbation(text):
    length_text = len(text)
    choice_id = random

fully_gene_datapath = 'GeneratedTextDetection-main/Dataset/FullyGenerated'
hybrid_gene_datapath = 'GeneratedTextDetection-main/Dataset/Hybrid_AbstractDataset'
model_path = 'flan-t5-small'


fully_gene_te_dataset = FullGeneratedDataset(split='test', dir=fully_gene_datapath)
fully_gene_te_dataloader = DataLoader(
    dataset=fully_gene_te_dataset,
    batch_size=4,
    # collate_fn=contrastive_collate_fn
)


flant5_small_tokenizer = AutoTokenizer.from_pretrained(model_path)
flant5_small_gen_model = T5ForConditionalGeneration.from_pretrained(model_path)

for i, data in enumerate(fully_gene_te_dataloader):
    generated, original = data
    input_ids = flant5_small_tokenizer(generated, return_tensors='pt',truncation=True, padding=True, max_length=512).input_ids
    outputs = flant5_small_gen_model.generate(torch.tensor(input_ids))
    print('done')

print('done')
