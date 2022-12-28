import torch, json



def process_data(orig_data, tokenizer, volumn=12000):
    min_len = 10 
    max_len = 300
    max_diff = 50
    prefix = 'translate English to German: '

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
            src_tokenized = tokenizer(prefix + src, max_length=512, truncation=True, padding=True)
            trg_tokenized = tokenizer(trg, max_length=512, truncation=True, padding=True)

            processed.append({'input_ids': src_tokenized['input_ids'],
                              'attention_mask': src_tokenized['attention_mask'],
                              'labels': trg_tokenized['input_ids']})
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == volumn:
                break

    return processed


def split_data(data_obj):
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]    
    return train, valid, test


    