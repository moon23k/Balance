import json, torch
from datasets import Dataset
from datasets import load_dataset
from transformers import (Trainer, 
						  TrainingArguments, 
						  DataCollatorForSeq2Seq,
						  T5TokenizerFast,
						  T5ForConditionalGeneration) 



def load_dataset(split):
	with open(f"data/{split}.json", 'r') as f:
		data = json.load(f)

	dataset = Dataset.from_list(train)
	return dataset


def main(mode):
	tokenizer = T5TokenizerFast.from_pretrained('t5-small', model_max_length=512)
	model = T5ForConditionalGeneration.from_pretrained('t5-small')

	train_ds = load_dataset('train')
	valid_ds = load_dataset('valid')
	test_ds = load_dataset('test')

	data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)


	default_args = {
	    "output_dir": "tmp",
	    "evaluation_strategy": "epoch",
	    "num_train_epochs": 1,
	    "log_level": "error",
	    "report_to": "none",
	    "logging_strategy": 'epoch'
	}


	training_args = TrainingArguments(
	    per_device_train_batch_size=32,
	    gradient_accumulation_steps=4,
	    gradient_checkpointing=True,
	    fp16=True,
	    optim="adafactor",
	    group_by_length=True,
	    **default_args,
	)

	trainer = Trainer(model=model, 
	                  args=training_args, 
	                  train_dataset=train_ds, 
	                  eval_dataset=valid_ds,
	                  data_collator=data_collator)

	result = trainer.train()
	print_summary(result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    
    args = parser.parse_args()
    assert args.mode in ['train', 'test', 'inference']

    main(args.mode)