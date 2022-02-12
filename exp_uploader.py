import requests
import time
import os
import getpass

host = ''

class Exp:
    def __init__(self, proj_name, exp_name, command):
        self.proj_name = proj_name
        self.exp_name = exp_name
        self.command = command
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.ip = get_host_ip()
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.username = getpass.getuser()
        self.gpu = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else ''

    def to_json(self):
        self.json = {
            'project': self.proj_name,
            'exp_name': self.exp_name,
            'command': self.command,
            'start_time': self.start_time,
            'ip': self.ip,
            'comments': '',
            'path': self.path,
            'results': {},
            'username': self.username,
            'pid': os.getpid(),
            'result_notification': False,
            'gpu': self.gpu
        }
        return self.json


def init_exp(exp: Exp):
    try:
        requests.post('%s/update_exp' % host, json=exp.to_json(), timeout=10)
    except Exception:
        print('uploading experiment info error!')


def append_comments(exp: Exp, comments: str):
    try:
        requests.post('%s/%s/%s/append_comments' % (host, exp.proj_name, exp.exp_name), data={'comments': comments}, timeout=10)
    except Exception:
        print('uploading experiment info error!')


def append_results(exp: Exp, key: str, value: str):
    try:
        requests.post('%s/%s/%s/update_results' % (host, exp.proj_name, exp.exp_name), data={'key': key, 'value': value}, timeout=10)
    except Exception:
        print('uploading experiment info error!')


def upload_test_progress(exp: Exp, progress: float, tested_sample_number: int, total_sample: int, step: int, dir: str):
    try:
        requests.post('%s/%s/%s/update_test_progress' % (host, exp.proj_name, exp.exp_name), json={
            'progress': progress,
            'tested_sample_number': tested_sample_number,
            'total_sample': total_sample,
            'pid': os.getpid(),
            'step': step,
            'dir': dir,
        }, timeout=10)
    except Exception:
        print('uploading experiment info error!')


def heart_beat(exp: Exp, loss=None, global_step=None):
    try:
        requests.post('%s/%s/%s/heart_beat' % (host, exp.proj_name, exp.exp_name), json={'loss': loss, 'step': global_step}, timeout=10)
    except Exception as e:
        print('uploading experiment info error! %s' % e)

def get_host_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

if __name__ == '__main__':
    e = Exp('test', 'legal_debugging', 'run_summarization.py --mode=train --data_path=../legal_train.json.seg --eval_path=../legal_valid.json.seg --vocab_path=../legal_vocab --dropout=0.7 --exp_name=legal_debugging')
    # init_exp(e)
    append_results(e, 'asda', 'dadadas')
