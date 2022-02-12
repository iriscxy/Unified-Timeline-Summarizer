import os
import re

from bleu.calculatebleu import calcu_bleu


def sys_bleu(result_txt, evaluated_list, reference_list):
    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')
    evaluates = './tmp/' + result_txt + '_e.txt'
    references = './tmp/' + result_txt + '_r.txt'
    with open(evaluates, 'w', encoding='utf8') as f1:
        for item in evaluated_list:
            f1.writelines(item + '\n')
    with open(references, 'w', encoding='utf8') as f2:
        for item in reference_list:
            f2.writelines(item + '\n')
    ret = sys_bleu_file(evaluates, references)
    os.remove(evaluates)
    os.remove(references)
    return ret


def sys_bleu_file(evaluated_file, reference_file):
    ScoreBleu_path = 'bleu/zgen_bleu/ScoreBleu.sh'
    command = ' '.join(["sh", ScoreBleu_path, "-t", evaluated_file, "-r", reference_file])
    result = os.popen(command).readline()
    sys_bleu_scores = re.findall(r"\= (.+?) \(", result)
    sys_bleu_score = float(sys_bleu_scores[0])
    return sys_bleu_score


def sys_bleu_perl(result_txt, evaluated_list, reference_list):
    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')
    evaluates = './tmp/' + result_txt + '_e.txt'
    references = './tmp/' + result_txt + '_r.txt'
    with open(evaluates, 'w', encoding='utf8') as f1:
        for item in evaluated_list:
            f1.writelines(item + '\n')
    with open(references, 'w', encoding='utf8') as f2:
        for item in reference_list:
            f2.writelines(item + '\n')
    ret = sys_bleu_perl_file(evaluates, references)
    os.remove(evaluates)
    os.remove(references)
    return ret


def sys_bleu_perl_file(evaluated_file, reference_file):
    command = 'bleu/multi-bleu-yiping.perl %s < %s' % (reference_file, evaluated_file)
    result = os.popen(command).readline()
    return result.strip()


def sys_bleu_yiping(result_txt, evaluated_list, reference_list):
    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')
    evaluates = './tmp/' + result_txt + '_e.txt'
    references = './tmp/' + result_txt + '_r.txt'
    with open(evaluates, 'w', encoding='utf8') as f1:
        for item in evaluated_list:
            f1.writelines(item + '\n')
    with open(references, 'w', encoding='utf8') as f2:
        for item in reference_list:
            f2.writelines(item + '\n')

    bleu, bleu1, bleu2, bleu3, bleu4 = calcu_bleu(evaluates, references)

    os.remove(evaluates)
    os.remove(references)
    return bleu, bleu1, bleu2, bleu3, bleu4


def sys_bleu_yiping_file(evaluated_file, reference_file):
    return calcu_bleu(evaluated_file, reference_file)


def get_list(results):
    lines = open(results, 'r').readlines()
    evaluates = []
    references = []
    for i in range(0, len(lines), 7):
        if len(lines[i + 1].strip().split(' ')) > 5000:
            continue
        else:
            references.append(lines[i + 3].strip().replace('__', ''))
            evaluates.append(lines[i + 5].strip().replace('__', ''))

    return evaluates, references
