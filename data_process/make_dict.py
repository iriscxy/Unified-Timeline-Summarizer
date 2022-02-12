import sys

import collections
import json
from tqdm import tqdm

if len(sys.argv) < 4:
    print('usage: file_in file_out VOCAB_SIZE keys....')
    exit()
keys = sys.argv[4:]
print('processing data, keys: [%s]' % ','.join(keys))
fi = open(sys.argv[1], encoding='utf8')
fo = open(sys.argv[2], 'w', encoding='utf8')
vocab_counter = collections.Counter()
counter = 0
err_counter = 0
try:
    for l in fi:
        jdata = json.loads(l)
        for k in keys:
            if k not in jdata or jdata[k] is None:
                err_counter += 1
                continue
            # all_words = jdata[k].split() if isinstance(jdata[k], str) else ' '.join(jdata[k]).split()
            all_words = list(jdata[k]) if isinstance(jdata[k], str) else list(' '.join(jdata[k]))
            # all_words = [w if '@' not in w else w[:3] for w in all_words if w != '<S>' and w != '</S>']
            vocab_counter.update(all_words)
        counter += 1
        sys.stdout.write('counter %d err %d \r' % (counter, err_counter))
        sys.stdout.flush()
except KeyboardInterrupt:
    print('Interrupted start writing vocab....')

print("Writing vocab file...")
for word, count in tqdm(vocab_counter.most_common(int(sys.argv[3])), desc="Writing vocab"):
    fo.write(word + ' ' + str(count) + '\n')
fo.close()