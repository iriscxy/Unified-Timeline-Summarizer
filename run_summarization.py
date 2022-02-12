import codecs
import random
from datetime import datetime
import glob
import json
import sys
import time
import os
import zipfile

import re
import shutil
import stat
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab
from model import SummarizationModel
from decode import BeamSearchDecoder
import util
from tensorflow.python import debug as tf_debug
from gpu_cluster import get_available_gpu, show_gpu_status, get_free_gpu
from util import bcolors, get_input_with_timeout
from exp_uploader import append_results, Exp, init_exp, heart_beat
import subprocess

FLAGS = tf.flags.FLAGS

# Where to find data
tf.flags.DEFINE_string('data_path', None, 'Path expression to train/decode data file.')
tf.flags.DEFINE_string('vocab_path', None, 'Path expression to text vocabulary file.')
tf.flags.DEFINE_string('eval_path', None, 'Path expression to eval data file.')
tf.flags.DEFINE_string('test_path', None, '')
tf.flags.DEFINE_string('pretrain_emb_pkl', None, '')
tf.flags.DEFINE_enum('data_type', 'json', ['bin', 'json'], 'file type of data file')
tf.flags.DEFINE_enum('lang', 'en', ['en', 'zh'], '')
tf.flags.DEFINE_string('json_input_key', 'content', '')
tf.flags.DEFINE_string('json_target_key', 'summary', '')
tf.flags.DEFINE_string('json_sent_key', 'sentences', '')
tf.flags.DEFINE_string('preprocessor', 'timeline', '')
tf.flags.DEFINE_string('string_split', 'space', 'string_split.')

# Important settings
tf.flags.DEFINE_enum('mode', 'train', ['train', 'decode', 'eval', 'auto_decode'], '')
tf.flags.DEFINE_boolean('single_pass', False,
                            'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
tf.flags.DEFINE_string('proj_name', 'seq2seqV2', 'name of project.')
tf.flags.DEFINE_string('log_root', 'logs', 'Root directory for all logging.')
tf.flags.DEFINE_string('exp_name', None, 'Name for experiment. ')

# Hyperparameters
tf.flags.DEFINE_enum('encoder', 'rnn', ['mix', 'rnn', 'transformer'], '')
tf.flags.DEFINE_integer('hidden_dim', 128, 'dimension of RNN hidden states')
tf.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.flags.DEFINE_integer('batch_size', 64, 'minibatch size')
tf.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.flags.DEFINE_integer('min_dec_steps', 20, 'Minimum sequence length of generated summary.')
tf.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary.')
tf.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
tf.flags.DEFINE_integer('dataset_size', None, 'dataset_size')
tf.flags.DEFINE_enum('optimizer', 'adagrad', ['adagrad', 'adam'], 'decoder model type')

# hierarchical encoder settings
tf.flags.DEFINE_boolean('seq_clip', True, "clip the sequence length by HRED")
tf.flags.DEFINE_integer('max_art_lens', 8, 'max number of event')
tf.flags.DEFINE_integer('max_hredsent_lens', 50, 'max length of event')
tf.flags.DEFINE_integer('max_sen_len', 20, 'max number of event')
tf.flags.DEFINE_integer('max_ext_steps', 4, 'max timesteps of decoder (max summary tokens)')
tf.flags.DEFINE_integer('kernel_size', 25, 'max number of event')
tf.flags.DEFINE_integer('memory_layer', 3, 'memory layer num')

# Transformer hyperparameters
tf.flags.DEFINE_integer('num_layers', 4, 'The number of layers')
tf.flags.DEFINE_integer('num_units', 128, 'The number of hidden units')
tf.flags.DEFINE_integer('num_heads', 8, 'The number of heads in the multi-head attention')
tf.flags.DEFINE_integer('ffn_inner_dim', 512, 'The number of units of the inner linear transformation in the feed forward layer.')
tf.flags.DEFINE_float('dropout', 0.2, 'The probability to drop units from the outputs')
tf.flags.DEFINE_float('attention_dropout', 0.2, 'The probability to drop units from the attention')
tf.flags.DEFINE_float('relu_dropout', 0.2, 'The probability to drop units from the ReLU activation in the feed forward layer.')

# Pointer-generator or baseline model
tf.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')
tf.flags.DEFINE_boolean('decode_rouge', True, '')
tf.flags.DEFINE_boolean('decode_bleu', True, '')
tf.flags.DEFINE_boolean('eval_when_epoch_finish', False, '')
tf.flags.DEFINE_integer('eval_every_step', 10000, '')
tf.flags.DEFINE_boolean('decode_rouge_server', False, '')

# Coverage hyperparameters
tf.flags.DEFINE_boolean('coverage', False,
                        'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.flags.DEFINE_float('cov_loss_wt', 1.0,
                          'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.flags.DEFINE_boolean('convert_to_coverage_model', False,
                        'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")
tf.flags.DEFINE_integer('device', None, '')
tf.flags.DEFINE_integer('auto_decode_epoch_num', None, '')
tf.flags.DEFINE_boolean('plot_gradients', False, "plot the gradients on tensorboard")
tf.flags.DEFINE_string('current_source_code_zip', None, "current_source_code_zip")

tf.flags.mark_flag_as_required("data_path")
tf.flags.mark_flag_as_required("eval_path")
tf.flags.mark_flag_as_required("exp_name")
tf.flags.mark_flag_as_required("dataset_size")

tf.flags.register_validator('test_path',
                         lambda value: value is not None if FLAGS.mode == 'auto_decode' else True,
                         message='auto_decode test_path should not be empty')
tf.flags.register_validator('test_path',
                         lambda value: value is not None if FLAGS.eval_when_epoch_finish else True,
                         message='eval_when_epoch_finish test_path should not be empty ')

from text_batcher import Batcher


def convert_to_coverage_model():
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    tf.logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=util.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = util.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver()  # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()


def start_auto_decode_proc(epoch_num=None):
    def run_command(command, stdout=None):
        if stdout is None:
            with open(os.devnull, 'w') as devnull:
                child = subprocess.Popen(command, shell=True, stdout=devnull)
                return child
        else:
            child = subprocess.Popen(command, shell=True, stdout=stdout)
            return child
    flag_str = ''
    except_key = ['mode', 'data_path', 'log_root', 'h', 'help', 'helpfull', 'helpshort', 'device', 'vocab_path',
                  'test_path', 'eval_path']
    for key, val in FLAGS.__flags.items():
        val = val._value
        if key not in except_key and val is not None:
            flag_str += '--%s=%s ' % (key, val)
        elif key == 'mode':
            flag_str += '--mode=auto_decode '
        elif key == 'data_path':
            flag_str += '--data_path=%s ' % os.path.abspath(FLAGS.test_path)
        elif key == 'test_path':
            flag_str += '--test_path=%s ' % os.path.abspath(FLAGS.test_path)
        elif key == 'vocab_path':
            flag_str += '--vocab_path=%s ' % os.path.abspath(FLAGS.vocab_path)
        elif key == 'eval_path':
            flag_str += '--eval_path=%s ' % os.path.abspath(FLAGS.eval_path)
        elif key == 'log_root':
            flag_str += '--log_root=%s ' % os.path.abspath(os.path.join(FLAGS.log_root, '../'))
        elif key == 'device':
            flag_str += '--device=%d ' % get_free_gpu()
    if epoch_num is not None:
        flag_str += '--auto_decode_epoch_num=%d ' % epoch_num

    source_code_path = os.path.join(os.path.abspath(os.path.dirname(FLAGS.current_source_code_zip)), 'train_code')
    if os.path.exists(source_code_path):
        shutil.rmtree(source_code_path)
    zip_ref = zipfile.ZipFile(FLAGS.current_source_code_zip, 'r')
    zip_ref.extractall(source_code_path)
    zip_ref.close()
    # child = run_command('unzip %s -d %s' % (FLAGS.current_source_code_zip, source_code_path))
    # child.wait()
    tf.logging.info('unzip source code finish!')

    src_bleu = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'bleu')
    dst_bleu = os.path.join(source_code_path, 'bleu')
    os.symlink(src_bleu, dst_bleu)
    tf.logging.info('making bleu symlink success!')

    run_file_path = os.path.join(source_code_path, 'run_summarization.py')
    tf.logging.debug(' '.join([sys.executable, run_file_path, flag_str]))
    child = run_command(' '.join([sys.executable, run_file_path, flag_str]))
    sys.stderr.write(' '.join([sys.executable, run_file_path, flag_str]) + '\n')
    sys.stderr.flush()


def setup_training(model, batcher, eval_batcher):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    init_exp(Exp(FLAGS.proj_name, FLAGS.exp_name, ' '.join(sys.argv)))

    epoch_ckpt_dir = os.path.join(FLAGS.log_root, "epoch_ckpt")
    if not os.path.exists(epoch_ckpt_dir): os.makedirs(epoch_ckpt_dir)

    model.build_graph()  # build the graph
    if FLAGS.convert_to_coverage_model:
        assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
        convert_to_coverage_model()
    epoch_saver = tf.train.Saver(max_to_keep=99)  # keep 3 checkpoints at a time

    summary_writer = tf.summary.FileWriter(train_dir)
    # summary_hook = tf.train.SummarySaverHook(save_secs=60, output_dir=train_dir,
    #                                          summary_writer=summary_writer, summary_op=model._summaries)
    monitoredSession = tf.train.MonitoredTrainingSession(checkpoint_dir=train_dir,
                                      save_checkpoint_secs=600, hooks=[], config=util.get_config())
    tf.logging.info("Created session.")
    try:
        run_training(model, batcher, monitoredSession, summary_writer, eval_batcher, epoch_saver)
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        batcher.close()
        eval_batcher.close()
        # monitoredSession.close()


def run_training(model, batcher, sess, summary_writer, eval_batcher, epoch_saver):
    """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
    tf.logging.info("starting run_training")

    def get_session(sess):
        session = sess
        while type(session).__name__ != 'Session':
            # pylint: disable=W0212
            session = session._sess
        return session

    with sess:
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        train_step = None
        while not sess.should_stop():
            batch = batcher.next_batch()
            summary_flag = False
            if train_step is not None and train_step % 20 == 0:
                summary_flag = True
            t0 = time.time()
            results = model.run_train_step(sess, batch, summary_flag)
            t1 = time.time()

            train_step = results['global_step']
            train_epoch = results['global_epoch']

            if 'summaries' in results:
                summaries = results['summaries']
                summary_writer.add_summary(summaries, train_step)
                # summary_writer.flush()

            if train_step * FLAGS.batch_size > train_epoch * FLAGS.dataset_size:
                epoch_num = sess.run(model.add_epoch_op, model._make_feed_dict(batch))
                epoch_ckpt_dir = os.path.join(FLAGS.log_root, "epoch_ckpt")
                epoch_saver.save(get_session(sess), os.path.join(epoch_ckpt_dir, 'ep{}'.format(epoch_num)))
                # append_results(Exp(FLAGS.proj_name, FLAGS.exp_name, ' '.join(sys.argv)), 'epoch', str(epoch_num))
                if FLAGS.eval_when_epoch_finish:
                    start_auto_decode_proc(epoch_num)

            if train_step % 20 == 0:
                loss = results['loss']
                ext_loss = results['ext_loss']
                con_loss = results['con_loss']
                acc, precision, recall = results['metrics']
                model.run_metrics(sess, batch)
                tf.logging.info('epoch: %d | step: %d | loss: %.3f | ext_loss: %.3f | con_loss: %.3f | acc: %.3f | time: %.3f', train_epoch, train_step, loss, ext_loss, con_loss, acc, t1 - t0)
                if not np.isfinite(loss):
                    raise Exception("Loss is not finite. Stopping.")
                if FLAGS.coverage:
                    coverage_loss = results['coverage_loss']
                    tf.logging.info("coverage_loss: %f", coverage_loss)

            if FLAGS.eval_every_step is not None and train_step % FLAGS.eval_every_step == 0 and train_step > 1:
                start_auto_decode_proc()


def get_max_epoch_num(log_root):
    """
    :param log_root:
    :return:
    """
    f = open(os.path.join(log_root, 'epoch_ckpt', 'checkpoint'))
    line = f.readline()
    f.close()
    pattern = re.compile('model_checkpoint_path: "ep(\d*)"')
    match = pattern.findall(line)
    if match:
        return int(match[0])


def select_decode_mode(placeholder_session, decode_model_hps, vocab, batcher, hps):
    epoch_ckpt_path = os.path.join(FLAGS.log_root, 'epoch_ckpt')
    if os.path.exists(epoch_ckpt_path):
        decode_type = get_input_with_timeout('1. only decode newest checkpoint；2. decode all epoches ；3. decode selected epoch', 15, '1')
    else:
        decode_type = '1'

    if decode_type == '1':  # decode newest ckpt
        model = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab)
        placeholder_session.close()
        try:
            final_metrics = decoder.decode()
            auto_decoding(decoder, final_metrics)
        except KeyboardInterrupt:
            tf.logging.info('stop decoding!')
    elif decode_type == '2':
        metrics_file = open(os.path.join(FLAGS.log_root, 'epoch-metric.txt'), 'a', encoding='utf8')
        model = SummarizationModel(decode_model_hps, vocab)
        saver, session = None, None
        placeholder_session.close()
        for epoch_num in range(2, 99):
            decoder = BeamSearchDecoder(model, batcher, vocab, 'epoch_ckpt', 'ep{}'.format(epoch_num), saver,
                                        session)
            saver = decoder._saver
            session = decoder._sess
            try:
                final_metrics = decoder.decode()
                auto_decoding(decoder, final_metrics)
                if FLAGS.decode_bleu:
                    metrics_file.write('epoch ' + str(epoch_num) + ': ' + final_metrics['sys_bleu_perl'] + '\n')
                    metrics_file.flush()
                batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)
            except KeyboardInterrupt:
                tf.logging.info('stop decoding!')
                break
    elif decode_type == '3':
        model = SummarizationModel(decode_model_hps, vocab)
        max_epoch_num = get_max_epoch_num(FLAGS.log_root)
        placeholder_session.close()
        while True:
            epoch_num = int(input('input epoch num (2~{})'.format(max_epoch_num)))
            if epoch_num >= 2 or epoch_num <= max_epoch_num:
                break
        decoder = BeamSearchDecoder(model, batcher, vocab, 'epoch_ckpt', 'ep{}'.format(epoch_num))
        try:
            final_metrics = decoder.decode()
            auto_decoding(decoder, final_metrics)
            tf.logging.info(bcolors.OKGREEN + json.dumps(final_metrics) + bcolors.ENDC)
        except KeyboardInterrupt:
            tf.logging.info('stop decoding!')
    else:
        print('unknown decode type %s' % decode_type)


def auto_decoding(decoder, final_metrics):
    if FLAGS.decode_bleu:
        try:
            append_results(Exp(FLAGS.proj_name, FLAGS.exp_name, ' '.join(sys.argv)),
                           'BLEU' + decoder._decode_dir_name, final_metrics['sys_bleu_perl'])
            tf.logging.info(bcolors.OKGREEN + final_metrics['sys_bleu_perl'] + bcolors.ENDC)
            with open(os.path.join(FLAGS.log_root, 'epoch-metric.txt'), 'a', encoding='utf8') as f:
                f.write('%s BLEU' % decoder._decode_dir_name + '\n')
                f.write(final_metrics['sys_bleu_perl'] + '\n')
        except Exception as e:
            tf.logging.error('decode_bleu error %s', e)
    if FLAGS.decode_rouge:
        try:
            keys = ['rouge-1-f', 'rouge-2-f', 'rouge-L-f', 'rouge-1-r', 'rouge-2-r', 'rouge-L-r']
            append_results(Exp(FLAGS.proj_name, FLAGS.exp_name, ' '.join(sys.argv)),
                           'LROUGE' + decoder._decode_dir_name, ' '.join(['%.5f' % final_metrics[k] for k in keys]))
            with open(os.path.join(FLAGS.log_root, 'epoch-metric.txt'), 'a', encoding='utf8') as f:
                f.write('%s ROUGE' % decoder._decode_dir_name + '\n')
                f.write(json.dumps(final_metrics['rouge-dict']) + '\n')

            if FLAGS.lang == 'zh':
                keys = ['rouge-1-f-num', 'rouge-2-f-num', 'rouge-L-f-num', 'rouge-1-r-num', 'rouge-2-r-num',
                        'rouge-L-r-num']
                append_results(Exp(FLAGS.proj_name, FLAGS.exp_name, ' '.join(sys.argv)),
                               'LROUGE-num' + decoder._decode_dir_name, ' '.join(['%.5f' % final_metrics[k] for k in keys]))
                with open(os.path.join(FLAGS.log_root, 'epoch-metric.txt'), 'a', encoding='utf8') as f:
                    f.write('%s ROUGE-num' % decoder._decode_dir_name + '\n')
                    f.write(json.dumps(final_metrics['rouge-dict-num']) + '\n')
        except Exception as e:
            tf.logging.error('decode_rouge error %s', e)


def main(unused_argv):
    np.random.seed(111)
    random.seed(111)
    tf.set_random_seed(111)  # a seed value for randomness

    # GPU tricks
    if FLAGS.device is None:
        index_of_gpu = get_available_gpu()
        if index_of_gpu < 0:
            index_of_gpu = ''
        FLAGS.device = index_of_gpu
        tf.logging.info(bcolors.OKGREEN + 'using {}'.format(FLAGS.device) + bcolors.ENDC)
    else:
        index_of_gpu = FLAGS.device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index_of_gpu)
    placeholder_session = None

    if FLAGS.mode == 'auto_decode':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import logging
        log = logging.getLogger('tensorflow')
        log.setLevel(logging.FATAL)
        for h in log.handlers:
            log.removeHandler(h)
        log.addHandler(logging.NullHandler())
    else:
        from tensorflow.python.platform import tf_logging
        import logging
        tf_logging._logger.handlers[0].setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        tf.logging.info('try to occupy GPU memory!')
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        placeholder_session = tf.Session(config=config)
        limit = placeholder_session.run(tf.contrib.memory_stats.BytesLimit()) / 1073741824
        tf.logging.info('occupy GPU memory %f GB', limit)

    gpu_info = show_gpu_status()
    gpu = [gpu_info[k] for k in gpu_info.keys() if gpu_info[k]['index'] == index_of_gpu][0]
    tf.logging.info('\napps on GPU {}\n'.format(index_of_gpu) + '\n'.join(
        [gpu['apps'][pid]['user'] + '\t' + str(gpu['apps'][pid]['memory']) for pid in gpu['apps'].keys()]))

    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
    tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == "train":
            os.makedirs(FLAGS.log_root)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)  # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if 'decode' in FLAGS.mode:
        FLAGS.batch_size = FLAGS.beam_size
        FLAGS.single_pass = True
        FLAGS.dataset_size = -1

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and 'decode' not in FLAGS.mode:
        raise Exception("The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt',
                   'max_art_lens', 'max_hredsent_lens', 'pointer_gen', 'max_ext_steps']
    hps_dict = {}
    export_json = {}
    for key, val in FLAGS.__flags.items():
        val = val._value
        export_json[key] = val
        if key in hparam_list:
            hps_dict[key] = val
            tf.logging.info('{} {}'.format(key, val))
    for val in FLAGS:  # for each flag // New modification for TF 1.5
        if val in hparam_list:  # if it's in the list
            hps_dict[val] = FLAGS[val].value  # add it to the dict // New modification for TF 1.5
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    ######################
    # save parameters and python script
    ######################
    # save parameters
    tf.logging.info('saving parameters')
    current_time_str = datetime.now().strftime('%m-%d-%H-%M')
    json_para_file = open(os.path.join(FLAGS.log_root, 'flags-' + current_time_str + '-' + FLAGS.mode + '.json'), 'w')
    json_para_file.write(json.dumps(export_json, indent=4) + '\n')
    json_para_file.close()
    # save python source code
    FLAGS.current_source_code_zip = os.path.abspath(os.path.join(FLAGS.log_root, 'source_code_bak-' + current_time_str + '-' + FLAGS.mode + '.zip'))
    tf.logging.info('saving source code: %s', FLAGS.current_source_code_zip)
    python_list = glob.glob('./*.py')
    zip_file = zipfile.ZipFile(FLAGS.current_source_code_zip, 'w')
    for d in python_list:
        zip_file.write(d)
    for d in glob.glob('data_process/*.py'):
        zip_file.write(d)
    for d in glob.glob('opennmt/*.py'):
        zip_file.write(d)
    for d in glob.glob('opennmt/*/*.py'):
        zip_file.write(d)
    zip_file.close()

    # Create a batcher object that will create minibatches of data
    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    if FLAGS.mode == 'train':
        tf.logging.info("creating model...")
        model = SummarizationModel(hps, vocab)
        eval_batcher = Batcher(FLAGS.eval_path, vocab, hps, False)
        placeholder_session.close()
        setup_training(model, batcher, eval_batcher)
    elif FLAGS.mode == 'decode':
        st = os.stat("bleu/multi-bleu-yiping.perl")
        os.chmod("bleu/multi-bleu-yiping.perl", st.st_mode | stat.S_IRGRP | stat.S_IRUSR | stat.S_IROTH |
                 stat.S_IXGRP | stat.S_IXOTH | stat.S_IXUSR)
        decode_model_hps = hps._replace(max_dec_steps=1, max_ext_steps=1)
        select_decode_mode(placeholder_session, decode_model_hps, vocab, batcher, hps)
    elif FLAGS.mode == 'auto_decode':
        tf.logging.info(bcolors.REDBACK + "start auto decode! " + os.environ["CUDA_VISIBLE_DEVICES"] + bcolors.ENDC)
        st = os.stat("bleu/multi-bleu-yiping.perl")
        os.chmod("bleu/multi-bleu-yiping.perl", st.st_mode | stat.S_IRGRP | stat.S_IRUSR | stat.S_IROTH |
                 stat.S_IXGRP | stat.S_IXOTH | stat.S_IXUSR)
        decode_model_hps = hps._replace(max_dec_steps=1, max_ext_steps=1)
        model = SummarizationModel(decode_model_hps, vocab)
        if FLAGS.auto_decode_epoch_num is None:
            decoder = BeamSearchDecoder(model, batcher, vocab)
        else:
            decoder = BeamSearchDecoder(model, batcher, vocab, 'epoch_ckpt', 'ep{}'.format(FLAGS.auto_decode_epoch_num))
        try:
            final_metrics = decoder.decode()
            auto_decoding(decoder, final_metrics)
        except KeyboardInterrupt:
            tf.logging.info('stop decoding!')
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode/auto_decode")


if __name__ == '__main__':
    tf.app.run()
