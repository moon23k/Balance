import os, argparse, torch
from module.test import Tester
from module.train import Trainer
from module.data import load_dataloader
from transformers import (set_seed, 
	 					  T5Config,
	 					  T5ForConditionalGeneration)




class Config(object):
    def __init__(self, mode):    

        self.mode = mode
        self.mname = 't5-base'
        self.ckpt = f"ckpt/{self.pipe}.pt"
        
        self.clip = 1
        self.lr = 5e-5
        self.max_len = 300
        self.n_epochs = 10
        self.batch_size = 128
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
        model = T5ForConditionalGeneration.from_pretrained(config.m_name)
        print("Pretrained T5 Model has loaded")
    

    elif config.mode != 'train':
        assert os.path.exists(config.ckpt)
        
        model_config = T5Config.from_pretrained(config.m_name)
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

        if isinstance(model, EncoderDecoderModel):
            preds = model.generate(**encodings, use_cache=True)
        else:
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
		train_dataloader = load_dataloader(config, 'train')
		valid_dataloader = load_dataloader(config, 'valid')
		trainer = Trainer(config, model, train_dataloader, valid_dataloader)
		trainer.trian()


	elif mode == 'test':
		test_dataloader = load_dataloader(config, 'test')
		tester = Tester(config, model, tokenizer, test_dataloader)
		tester.test()

	elif model == 'inference':
		inference(config, model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']    

    main(args.mode)