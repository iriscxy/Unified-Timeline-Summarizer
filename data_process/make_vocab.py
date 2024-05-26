import json
from tqdm import tqdm
import pdb
from nltk.tokenize import word_tokenize

f=open('train.json')
fw=open('vocab','w')
lines=f.readlines()
counter = {}

for line in tqdm(lines[:]):
    content=json.loads(line)
    src=content['src']
    tgt=content['tgt']
    src=src+' '+tgt
    src=word_tokenize(src)
    for word in src:
        if word not in counter:
            counter[word] = 1
        else:
            counter[word] += 1
new_dict=sorted(counter.items(), key=lambda x: x[1], reverse=True)
for x in new_dict:
    fw.write(x[0]+' '+str(x[1])+'\n')
