import datasets
import json
import re
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--eval_data', type=str, default='')
args = parser.parse_args()

ds = datasets.load_dataset("openai/gsm8k", 'main')
pattern = r"#### (\-?[0-9\.,]+)"


dic = {}
for data in ds['test']:
    match = re.search(pattern, data['answer'])
    if match: 
        label = match.group(1)
        dic[data['question']] ={'answer': data['answer'], 'label': label}
    else:
        raise Exception("!!! strange !!!")
        

with open(f"answer_generation/generated/gsm8k_large/{args.eval_data}/eos_cut.json", 'r') as f:
    results = json.load(f)
    
pattern = r"\$?[\d]+"

correct_list = []
for data in results:
    match = re.findall(pattern, data['output'].replace(',',''))
    if match ==[]:
        y='a'
    else:
        y = match[-1]
    
    correct = dic[data['instruction']]['label'] ==y
    correct_list.append(correct)
    
print(np.array(correct_list).mean())