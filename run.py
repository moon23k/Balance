import os, argparse, torch
import torch.nn as nn
import torch.optim as optim
from transformers import (set_seed,
						  T5Config, 
						  T5TokenizerFast, 
						  T5ForConditionalGeneration)



class Config(object):
    def __init__(self, args):    
        self.task = args.task
        self.mode = args.mode
        self.ckpt = f"ckpt/{self.task}.pt"

        self.clip = 1
        self.n_epochs = 10
        self.batch_size = 128
        self.learning_rate = 5e-5
        self.iters_to_accumulate = 4

        self.early_stop = True
        self.patience = 3

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.device_type = 'cuda'
        else:
            self.device_type = 'cpu'

        if self.task == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if use_cuda else 'cpu')


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")


def inference(config):
    return


def main(args):
    set_seed()
    config = Config(args)
    model = load_model(config)
    

    if args.pipe == 'pytorch':
    	os.system(f'python3 pipes/torch/run.py -mode {args.mode}')	

    if args.pipe == 'pytorch_lite':
    	os.system(f'python3 pipes/torch_lite/run.py -mode {args.mode}')

    if args.pipe == 'huggingface':
    	os.system(f'python3 pipes/huggingface/run.py -mode {args.mode}')

    if args.pipe == 'fairseq':
    	os.system(f'python3 pipes/fairseq/run.py -mode {args.mode}')    	    	


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-pipe', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']
    assert args.pipe in ['torch', 'torch_lite', 'huggingface', 'fairseq']

    main(args)