# coding: utf-8
# @Author: Li Mingzhe
# @Time: 2020/11/2 0002 下午 07:18

import json
import re
import numpy as np

file = open('input_file', 'r')
target = open('output_file', 'w')


def text_to_sentences(t):
    pattern = r'[。？！.?!]'
    sentences = []
    while True:
        g = re.search(pattern, t)
        if g is None:
            if t != "":
                sentences.append(t)
            break
        position = g.span()[1]
        sentences.append(t[:position])
        t = t[position:]
    return sentences


def BOW_scorer(list1, list2):
    w1 = []
    for sent in list1:
        w1.extend(sent.split())
    w2 = []
    for sent in list2:
        w2.extend(sent.split())
    s1, s2 = set(w1), set(w2)
    if len(s2) == 0:
        return 0
    return float(len(s1 & s2)) / float(len(s2))


num = 0
for line in file:
    data = json.loads(line)
    summary = data['summ'][:4]
    article = data['document'][:8]
    sents = []
    for event in article:
        sentences = text_to_sentences(event)[:3]
        while len(sentences) < 3:
            sentences.append('')
        sents.extend(sentences)
    while len(sents) < 24:
        sents.append('')

    summ_sens = []
    for event in summary:
        sentences = text_to_sentences(event)
        summ_sens.extend(sentences)

    data['sentences'] = list(sents)

    label = []
    sum = []
    for sum_sen in summ_sens:
        scores = []
        for art_sen in sents:
            scores.append(BOW_scorer([art_sen], [sum_sen]))
        id = scores.index(max(scores))
        label.append(id)
        sents[id] = ''

    label.sort()

    data['extract_label'] = label
    data['summ'] = summ_sens
    json.dump(data, target)
    target.write('\n')
    num += 1
    if num % 1000 == 0:
        print(num)
