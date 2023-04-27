import os, json 
from datasets import load_dataset
from transformers import T5TokenizerFast




def process_data(orig_data, tokenizer, volumn=12000):
    min_len = 10 
    max_len = 300
    max_diff = 50

    volumn_cnt = 0
    processed = []
    
    for elem in orig_data:
        src, trg = elem['en'].lower(), elem['de'].lower()
        src_len, trg_len = len(src), len(trg)

        #define filtering conditions
        min_condition = (src_len >= min_len) & (trg_len >= min_len)
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        dif_condition = abs(src_len - trg_len) < max_diff

        if max_condition & min_condition & dif_condition:            
            processed.append({'input_ids': tokenizer(src).input_ids,
                              'labels': tokenizer(trg).input_ids})
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == volumn:
                break

    return processed


def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')
    


def main():
    #Load Original Data
    orig = load_dataset('wmt14', 'de-en', split='train')['translation']
    tokenizer = T5TokenizerFast.from_pretrained('t5-small', model_max_length=512)

    #PreProcess Data
    processed = process_data(orig, tokenizer)

    #Save Data
    save_data(processed)



if __name__ == '__main__':
    main()