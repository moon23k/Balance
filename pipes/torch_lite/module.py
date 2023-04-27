import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn import functional as F




class PLModule(pl.LightningModule):
	def __init__(self, config):
		super(PLModule, self).__init__()
  
        self.lr = config.lr
        self.device = config.device
		self.model = T5ForConditionalGeneration.from_pretrained(config.mname).to(self.device)


    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device), 
        attention_mask = (input_ids == self.pad_id).to(self.device)
        labels = batch['labels'].to(self.device)      

        loss = self.model(input_ids=input_ids, 
                          attention_mask=attention_mask,
                          labels=labels).loss
        
        return loss

    def validataion_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device), 
        attention_mask = (input_ids == self.pad_id).to(self.device)
        labels = batch['labels'].to(self.device)      
        
        loss = self.model(input_ids=input_ids, 
                          attention_mask=attention_mask,
                          labels=labels).loss
        
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device), 
        attention_mask = (input_ids == self.pad_id).to(self.device)
        labels = batch['labels'].to(self.device)      
        
        loss = self.model(input_ids=input_ids, 
                          attention_mask=attention_mask,
                          labels=labels).loss
        
        return loss


    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), self.lr)                   