import os, argparse, torch
from module import PLModel
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import (
    set_seed, 
    T5Config, 
    T5TokenizerFast, 
    T5ForConditionalGeneration
)




class Config(object):
    def __init__(self, mode):    

        self.mode = mode
        self.mname = 't5-base'
        self.ckpt = "ckpt/torch_lite.pt"
        
        self.clip = 1
        self.lr = 5e-5
        self.n_epochs = 10
        self.batch_size = 32
        self.iters_to_accumulate = 4

        self.early_stop = True
        self.patience = 3

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def load_model(config):
    #Inner methods
    def count_params(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params
        
    def check_size(model):
        param_size, buffer_size = 0, 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb


    #Actual Process
    if config.mode == 'train':
        model = T5ForConditionalGeneration.from_pretrained(config.mname)
        print("Pretrained T5 Model has loaded")
    

    elif config.mode != 'train':
        assert os.path.exists(config.ckpt)
        
        model_config = T5Config.from_pretrained(config.mname)
        model = T5ForConditionalGeneration(model_config)
        print("Initialized T5 Model has loaded")
        
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Trained Model states has loaded from {config.ckpt}")


    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")
    
    return model.to(config.device)



def inference(model, tokenizer):
    model.eval()
    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #End Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has terminated! ---')
            break        

        #convert user input_seq into model input_ids
        encodings = tokenizer(input_seq)
        preds = model.generate(**encodings)

        preds = tokenizer.decode(preds, skip_special_tokens=True)

        #Search Output Sequence
        print(f"Model Out Sequence >> {preds}")       



def main(mode):
    set_seed(42)
    config = Config(mode)
    model = load_model(config)
    tokenizer = T5TokenizerFast.from_pretrained(config.mname)
    setattr(config, 'vocab_size', tokenizer.vocab_size)
    setattr(config, 'pad_id', tokenizer.pad_token_id)


    if mode == 'train':
        pl_model = PLModel(config, model)
        trainer = pl.Trainer(max_epochs=config.n_epochs)
        trainer.fit(pl_model)

    elif mode == 'test':
        pl_model = PLModel(config, model, tokenizer)
        trainer = pl.Trainer()
        trainer.test()

    elif model == 'inference':
        inference(config, model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']    

    main(args.mode)