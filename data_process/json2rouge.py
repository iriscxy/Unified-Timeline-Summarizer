import os, sys, json

json_path = sys.argv[1]
out_path = sys.argv[2]

if not os.path.exists(os.path.join(out_path, 'decode')): os.makedirs(os.path.join(out_path, 'decode'))
if not os.path.exists(os.path.join(out_path, 'reference')): os.makedirs(os.path.join(out_path, 'reference'))

for i, l in enumerate(open(json_path, encoding='utf8')):
    sys.stdout.write('%d \r' % i)
    sys.stdout.flush()
    data = json.loads(l)

    with open(os.path.join(out_path, 'decode', "%06d_decoded.txt" % i), 'w', encoding='utf8') as f:
        summ = data['summary']
        if isinstance(data['summary'], list):
            summ = ' '.join(summ)
        f.write(summ + '\n')

    with open(os.path.join(out_path, 'reference', "%06d_reference.txt" % i), 'w', encoding='utf8') as f:
        content = data['content']
        if isinstance(data['content'], list):
            content = ' '.join(content)
        f.write(content + '\n')
