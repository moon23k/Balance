import os, yaml, argparse, torch

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from module import (
    load_dataloader,
    load_model,
    Trainer,
    Tester,
    Generator
)



def set_seed(SEED=42):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config(object):
    def __init__(self, args):    

        #Get attributions from 'config.yaml' file
        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.task = args.task
        self.mode = args.mode
        self.balance = args.balance
        self.model_type = args.model
        self.search_method = args.search        

        self.ckpt = f"ckpt/{self.task}/{self.balance}_{self.model_type}_model.pt"
        self.tokenizer_path = f'data/{self.task}/tokenizer.json'

        if self.task == 'summarization':
            self.batch_size = self.batch_size // 4

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' \
                           if use_cuda and self.mode != 'inference' \
                           else 'cpu'
        self.device = torch.device(self.device_type)

        self.enc_n_layers = self.n_layers
        self.dec_n_layers = self.n_layers            
        self.enc_hidden_dim = self.hidden_dim
        self.dec_hidden_dim = self.hidden_dim
        self.enc_pff_dim = self.pff_dim
        self.dec_pff_dim = self.pff_dim

        hidden_dim = self.hidden_dim
        pff_dim = self.pff_dim
        n_layers = self.n_layers


        if self.balance == 'equal':
            if self.model_type == 'wide':
                self.enc_hidden_dim = hidden_dim * 2
                self.dec_hidden_dim = hidden_dim * 2
                self.enc_pff_dim = pff_dim * 2
                self.dec_pff_dim = pff_dim * 2            
            elif self.model_type == 'deep':
                self.enc_n_layers = n_layers * 2
                self.dec_n_layers = n_layers * 2

        elif self.balance == 'encoder':
            if self.model_type == 'wide':
                self.enc_hidden_dim = hidden_dim * 2
                self.dec_hidden_dim = hidden_dim
                self.enc_pff_dim = pff_dim * 2
                self.dec_pff_dim = pff_dim
            elif self.model_type == 'deep':
                self.enc_n_layers = n_layers * 2
                self.dec_n_layers = n_layers

        elif self.balance == 'decoder':
            if self.model_type == 'wide':
                self.enc_hidden_dim = hidden_dim
                self.dec_hidden_dim = hidden_dim * 2
                self.enc_pff_dim = pff_dim
                self.dec_pff_dim = pff_dim * 2
            elif self.model_type == 'deep':
                self.enc_n_layers = n_layers
                self.dec_n_layers = n_layers * 2


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def load_tokenizer(config):
    assert os.path.exists(config.tokenizer_path)

    tokenizer = Tokenizer.from_file(config.tokenizer_path)    
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[(config.bos_token, config.bos_id), 
                        (config.eos_token, config.eos_id)]
        )
    
    return tokenizer




def main(args):
    set_seed()
    config = Config(args)
    model = load_model(config)
    tokenizer = load_tokenizer(config)


    if config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()
    
    elif config.mode == 'inference':
        generator = Generator(config, model, tokenizer)
        generator.inference()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-balance', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.task.lower() in ['translation', 'dialogue', 'summarization']
    assert args.mode.lower() in ['train', 'test', 'inference']
    assert args.balance.lower() in ['equal', 'encoder', 'decoder']
    assert args.model.lower() in ['default', 'wide', 'deep']
    assert args.search.lower() in ['greedy', 'beam']

    if args.balance.lower() in ['encoder', 'decoder']:
        assert args.model in ['wide', 'deep']

    if args.mode == 'train':
        os.makedirs(f"ckpt/{args.task}", exist_ok=True)
    else:
        assert os.path.exists(f'ckpt/{args.task}/model.pt')

    main(args)