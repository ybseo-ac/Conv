import os





def ready_dataset(config, dataset, tokenizer):

    # EOS = tokenizer.eos_token_id
    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    if 'llada' in config.backbone: 
        PAD = tokenizer.encode("<|mdm_mask|>")[0]   # '<|mdm_mask|>'
    else:
        PAD = tokenizer.pad_token_id
    print(PAD)
    print(tokenizer.decode([PAD]))
    wrap = True
    def preprocess_and_tokenize(example):
        text = example['instruction']
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'
        text0=[]
        # for t in text:
        #     a = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{t}\n\n### Response:\n"""
        for t in text:
            a = f"""Below is an instruction that describes a task. Think through the problem carefully, explaining in detail before arriving at the final answer.\n\n### Instruction:\n{t}\n\n### Response:\n"""
            text0.append(a)
        text = text0
        tokens = tokenizer(text,
                            add_special_tokens=False,
                            return_attention_mask=False,
                            return_token_type_ids=False
                            )
        ar_list = []   
        df_list = []
        for t in tokens['input_ids']:
            a = [EOS] + t + [PAD] *(config.model_max_length - len(t)-1)
            a = a[:config.model_max_length]
            df_list.append(a) 
            
        tokens['xT'] = df_list
        del tokens['input_ids']
        return tokens

    test_dataset = dataset.map(preprocess_and_tokenize, batched=True,       num_proc=len(os.sched_getaffinity(0)),
        load_from_cache_file=True, desc='Tokenizing')

    test_dataset = test_dataset.with_format("torch")
    return test_dataset



def generation_eos_cut(generated, tokenizer):   # generated : [{}, {},,,]
    temp_new = []
    for data in generated:
        x = data['output']
        token0 = tokenizer.encode(x, return_tensors='pt')
        token_indices = (token0 == tokenizer.eos_token_id).nonzero(as_tuple=True)[1].tolist()[1:]
        answer_len = token_indices[0]
        short_x = tokenizer.decode(token0[0,1:answer_len])
        data['output'] = short_x
        temp_new.append(data)
    return temp_new



def generation_len_cut(generated, tokenizer, len0=500):   # generated : [{}, {},,,]
    temp_new = []
    for data in generated:
        x = data['output']
        token0 = tokenizer.encode(x, return_tensors='pt')
        short_x = tokenizer.decode(token0[0, : len0])
        data['output'] = short_x
        temp_new.append(data)
    return temp_new