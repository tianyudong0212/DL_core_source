import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import os
from tqdm import tqdm
from collections import deque
from transformers import AutoTokenizer, T5EncoderModel
import pickle

from data_proc import FullGeneratedDataset, HybridDataset, contrastive_collate_fn
from utils_fn import contrastive_loss


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'

fully_gene_datapath = 'GeneratedTextDetection-main/Dataset/FullyGenerated'
hybrid_gene_datapath = 'GeneratedTextDetection-main/Dataset/Hybrid_AbstractDataset'
model_path = 'flan-t5-small'

fully_gene_tr_dataset = FullGeneratedDataset(split='train', dir=fully_gene_datapath)
fully_gene_te_dataset = FullGeneratedDataset(split='test', dir=fully_gene_datapath)
fully_gene_tr_dataloader = DataLoader(
    dataset=fully_gene_tr_dataset,
    batch_size=3,
    collate_fn=contrastive_collate_fn
)
fully_gene_te_dataloader = DataLoader(
    dataset=fully_gene_te_dataset,
    batch_size=3,
    collate_fn=contrastive_collate_fn
)


hybrid_gene_tr_dataset = HybridDataset(split='train', dir=hybrid_gene_datapath)
hybrid_gene_te_dataset = HybridDataset(split='test', dir=hybrid_gene_datapath)
hybrid_gene_tr_dataloader = DataLoader(
    dataset=hybrid_gene_tr_dataset,
    batch_size=3,
    collate_fn=contrastive_collate_fn
)
hybrid_gene_te_dataloader = DataLoader(
    dataset=hybrid_gene_te_dataset,
    batch_size=3,
    collate_fn=contrastive_collate_fn
)




flant5_small_tokenizer = AutoTokenizer.from_pretrained(model_path)
flant5_small_model =  T5EncoderModel.from_pretrained(model_path).to(DEVICE)
optimizer_fully = Adam(params=flant5_small_model.parameters(), lr=5e-4)
scheduler_fully = StepLR(
    optimizer=optimizer_fully,
    step_size=1,
    gamma=0.2,
    verbose=True
)
loss_fn = contrastive_loss

def contrastive_learn(train_ld, test_ld, model, tokenizer, num_epoch, optimizer, loss_fn, task:str):
    dp1 = nn.Dropout(p=0.2)
    dp2 = nn.Dropout(p=0.2)
    slide_losses = deque(maxlen=20)
    all_step_slide_losses = []
    all_step_current_losses = []

    for i_epoch in range(1, num_epoch+1):
        t = tqdm(train_ld)
        for i, data in enumerate(t):
            batch_loss = 0
            for mini_data in data:
                optimizer.zero_grad()
                inputs = tokenizer(mini_data, return_tensors='pt',truncation=True, padding=True, max_length=512).to(DEVICE)
                outputs = model(**inputs).last_hidden_state
                embs = outputs.mean(dim=1).squeeze()
                q = dp1(embs[0])
                k0 = dp2(embs[0])
                loss = loss_fn(q, k0, embs[1:])
                batch_loss += loss


            batch_loss.backward()
            optimizer.step()
            temp_loss = batch_loss.item()
            slide_losses.append(temp_loss)

            slide_avg_loss = sum(slide_losses) / len(slide_losses)
            all_step_slide_losses.append(slide_avg_loss)
            all_step_current_losses.append(temp_loss)
            t.set_description(f'epoch:{i_epoch}, temploss:{temp_loss:.03}, meanloss:{slide_avg_loss:.03}')
    
        # save model to disk
        if i_epoch == num_epoch:
            print(f'save trained models to disk...')
            model_name = task + f'meanloss_{slide_avg_loss:.03}.pt'
            model_path = os.path.join('temp_mdls', model_name)
            
            with open(model_path, 'wb') as f:
                torch.save(model, f)
    
    return all_step_slide_losses, all_step_current_losses

slide_loss_fully, current_loss_fully = contrastive_learn(fully_gene_tr_dataloader, 
                                                        fully_gene_te_dataloader,
                                                        flant5_small_model,
                                                        flant5_small_tokenizer,
                                                        6,
                                                        optimizer_fully,
                                                        loss_fn,
                                                        task='fully')

# with open('exp_files/slide_loss_fully.pkl', 'wb') as f1:
#     pickle.dump(slide_loss_fully, f1)

# with open('exp_files/current_loss_fully.pkl', 'wb') as f2:
#     pickle.dump(current_loss_fully, f2)


# slide_loss_hybrid, current_loss_hybrid = contrastive_learn(hybrid_gene_tr_dataloader, 
#                                                         hybrid_gene_te_dataloader,
#                                                         flant5_small_model,
#                                                         flant5_small_tokenizer,
#                                                         6,
#                                                         optimizer_fuly,
#                                                         loss_fn)

# with open('exp_files/slide_loss_hybrid.pkl', 'wb') as f3:
#     pickle.dump(slide_loss_hybrid, f3)

# with open('exp_files/current_loss_hybrid.pkl', 'wb') as f4:
#     pickle.dump(current_loss_hybrid, f4)
if __name__ == '__main__':
    print('done')