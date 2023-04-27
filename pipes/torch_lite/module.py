import math, json, torch, evaluate
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"../../data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = self.data[idx]['input_ids']
        labels = self.data[idx]['labels']
        return input_ids, labels



class PLModel(pl.LightningModule):
    def __init__(self, config, model, tokenizer=None):
        super(PLModel, self).__init__()

        self.lr = config.lr
        self.mode = config.mode
        self.pad_id = config.pad_id
        self.batch_size = config.batch_size
        
        self.model = model
        if self.mode == 'test':
            self.tokenizer = tokenizer
            self.metric_module = evaluate.load('bleu')


    def _collate_fn(self, batch):
        ids_batch, labels_batch = [], []
        for ids, labels in batch:
            ids_batch.append(torch.LongTensor(ids)) 
            labels_batch.append(torch.LongTensor(labels))
        
        ids_batch = pad_sequence(ids_batch, batch_first=True, padding_value=self.pad_id)
        mask_batch = (ids_batch==self.pad_id)
        labels_batch = pad_sequence(labels_batch, batch_first=True, padding_value=self.pad_id)
        
        return {'input_ids': ids_batch,
                'attention_mask': mask_batch,
                'labels': labels_batch}

    def load_dataloader(self, split):
        return DataLoader(Dataset(split), 
                            batch_size=self.batch_size,
                            collate_fn=self._collate_fn,
                            shuffle=True if split == 'train' else False, 
                            pin_memory=True,
                            num_workers=2)


    def train_dataloader(self):
        return self.load_dataloader('train')

    def validation_dataloader(self):
        return self.load_dataloader('valid')

    def test_dataloader(self):
        return self.load_dataloader('test')        


    def training_step(self, batch, batch_idx):
        device = self.model.device
        loss =  self.model(input_ids=batch['input_ids'].to(device), 
                           attention_mask=batch['attention_mask'].to(device),
                           labels=batch['labels'].to(device)).loss
        
        self.log('train loss', loss)
        self.log('train ppl', math.exp(loss))
        
        return loss

    def validataion_step(self, batch, batch_idx):
        device = self.model.device
        loss =  self.model(input_ids=batch['input_ids'].to(device), 
                           attention_mask=batch['attention_mask'].to(device),
                           labels=batch['labels'].to(device)).loss
        
        self.log('train loss', loss)
        self.log('train ppl', math.exp(loss))
        


    def test_step(self, batch, batch_idx):
        device = self.model.device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        greedy_pred = self.model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          use_cache=True)
        
        beam_pred = self.model.generate(input_ids=input_ids, 
                                        attention_mask=attention_mask, 
                                        use_cache=True)
        
        greedy_pred = self.tokenizer.batch_decode(greedy_pred)
        beam_pred = self.tokenizer.batch_decode(beam_pred)
        labels = self.tokenizer.batch_decode(labels)

        greedy_score = self.metric_module.compute(greedy_pred, labels)
        beam_score = self.metric_module.compute(beam_pred, labels)

        return {'greedy_score': greedy_score,
                'beam_score': beam_score}


    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), self.lr)
