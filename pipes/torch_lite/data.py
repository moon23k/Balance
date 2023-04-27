import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = self.data[idx]['input_ids']
        labels = self.data[idx]['labels']
        
        return input_ids, labels



class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.pad_id = config.pad_id
        self.num_workers = 2


    def prepare_data(self, split):
        return Dataset(split)


    def setup(self, stage=None):
        if stage == 'fit':
            return self.train_datalaoder, self.valid_dataloader
        elif stage == 'test':
            return self.test_datalaoder


    def collate_fn(self, batch):
        ids_batch, labels_batch = [], []
        for ids, labels in batch:
            ids_batch.append(ids) 
            labels_batch.append(labels)

        return {'input_ids': pad_sequence(ids_batch, batch_first=True, padding_value=self.pad_id),
                'labels': pad_sequence(labels_batch, batch_first=True, padding_value=self.pad_id)}

    
    def train_dataloader(self):
        return DataLoader()
    
    def val_dataloader(self):
        return DataLoader()
    
    def test_datalaoder(self):
        return DataLoader()


    def on_epoch_end(self):
        return