# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to run beam search decoding, including running ROUGE evaluation and producing JSON datafiles for the in-browser attention visualizer, which can be found here https://github.com/abisee/attn_vis"""
import sys

import os
import time
import re

import shutil
import tensorflow as tf
import beam_search
import data
import json
import pyrouge
import util
import logging
from bleu.calculatebleu import calcu_bleu
from bleu.score_bleu import sys_bleu_perl_file, sys_bleu_file
from text_batcher import Batcher
from util import bcolors
from exp_uploader import append_results, Exp, upload_test_progress

FLAGS = tf.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint


class BeamSearchDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, batcher: Batcher, vocab, ckpt_dir=None, ckpt_name=None, saver=None, session=None):
        """Initialize decoder.

        Args:
          model: a Seq2SeqAttentionModel object.
          batcher: a Batcher object.
          vocab: Vocabulary object
        """
        self._model = model
        self._model.build_graph()
        self._batcher = batcher
        self._vocab = vocab
        self._saver = tf.train.Saver()  # we use this to load checkpoints for decoding
        self._sess = tf.Session(config=util.get_config())

        # Load an initial checkpoint to use for decoding
        if ckpt_dir and ckpt_name:
            ckpt_path = util.load_specific_ckpt(self._saver, self._sess, ckpt_dir, ckpt_name)
        else:
            ckpt_path = util.load_ckpt(self._saver, self._sess)
        if ckpt_path is not None:
            tf.logging.info("ckpt path --> %s", ckpt_path)

        if FLAGS.single_pass:
            # Make a descriptive decode directory name
            if ckpt_path.startswith('ep'):
                ckpt_name = 'epoch-' + ckpt_path.replace('ep', '')
                self.step = -1
            else:
                ckpt_name = "ckpt-" + ckpt_path.split('-')[-1]  # this is something of the form "ckpt-123456"
                self.step = int(ckpt_path.split('-')[-1])
            self._decode_dir_name = get_decode_dir_name(ckpt_name)
            self._decode_dir = os.path.join(FLAGS.log_root, self._decode_dir_name)
            tf.logging.info("decode path --> %s", self._decode_dir)
            if os.path.exists(self._decode_dir):
                shutil.rmtree(self._decode_dir)
                tf.logging.info('removed %s', self._decode_dir)

        else:  # Generic decode dir name
            self._decode_dir = os.path.join(FLAGS.log_root, "decode")

        # Make the decode dir if necessary
        if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

        if FLAGS.single_pass:
            # Make the dirs to contain output written in the correct format for pyrouge
            self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
            if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
            self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
            if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)
            self._rouge_ext_dir = os.path.join(self._decode_dir, "extracted")
            if not os.path.exists(self._rouge_ext_dir): os.mkdir(self._rouge_ext_dir)
            self._bleu_dec_dir = os.path.join(self._decode_dir, "bleu")
            if not os.path.exists(self._bleu_dec_dir): os.mkdir(self._bleu_dec_dir)
            self._vis_dir = os.path.join(self._decode_dir, "vis")
            if not os.path.exists(self._vis_dir): os.mkdir(self._vis_dir)
            if FLAGS.lang == 'zh':
                self._rouge_num_dec_dir = os.path.join(self._decode_dir, "num_decoded")
                if not os.path.exists(self._rouge_num_dec_dir): os.mkdir(self._rouge_num_dec_dir)

                self._rouge_num_ref_dir = os.path.join(self._decode_dir, "num_reference")
                if not os.path.exists(self._rouge_num_ref_dir): os.mkdir(self._rouge_num_ref_dir)

    def decode(self):
        """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
        final_metrics = {}
        t0 = time.time()
        counter = 0
        num_dict = {'counter': 0, 'w2i': {}}

        while True:
            batch = self._batcher.next_batch()  # 1 example repeated across batch
            if batch is None:  # finished decoding dataset in single_pass mode
                assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
                upload_test_progress(Exp(FLAGS.proj_name, FLAGS.exp_name, ' '.join(sys.argv)),
                                     1.0, counter, self._batcher._total_lines, self.step, self._decode_dir)
                if FLAGS.decode_rouge:
                    tf.logging.info("Output has been saved in %s and %s. Now starting ROUGE eval...",
                                    self._rouge_ref_dir,
                                    self._rouge_dec_dir)
                    try:
                        t0 = time.time()
                        if FLAGS.lang == 'zh':
                            num_results_dict = rouge_eval(self._rouge_num_ref_dir, self._rouge_num_dec_dir)
                            final_metrics['rouge-1-f'] = num_results_dict['rouge_1_f_score']
                            final_metrics['rouge-2-f'] = num_results_dict['rouge_2_f_score']
                            final_metrics['rouge-L-f'] = num_results_dict['rouge_l_f_score']
                            final_metrics['rouge-1-r'] = num_results_dict['rouge_1_recall']
                            final_metrics['rouge-2-r'] = num_results_dict['rouge_2_recall']
                            final_metrics['rouge-L-r'] = num_results_dict['rouge_l_recall']
                            final_metrics['rouge-dict'] = num_results_dict
                        else:
                            results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
                            final_metrics['rouge-1-f'] = results_dict['rouge_1_f_score']
                            final_metrics['rouge-2-f'] = results_dict['rouge_2_f_score']
                            final_metrics['rouge-L-f'] = results_dict['rouge_l_f_score']
                            final_metrics['rouge-1-r'] = results_dict['rouge_1_recall']
                            final_metrics['rouge-2-r'] = results_dict['rouge_2_recall']
                            final_metrics['rouge-L-r'] = results_dict['rouge_l_recall']
                            final_metrics['rouge-dict'] = results_dict
                        writer = open(os.path.join(self._decode_dir, 'num_rouge_dict.txt'), 'w', encoding='utf8')
                        for word, count in num_dict['w2i'].items():
                            writer.write(word + ' ' + str(count) + '\n')
                        writer.close()
                        t1 = time.time()
                        tf.logging.info('calculate Rouge score cost %d seconds', t1 - t0)
                        tf.logging.info(bcolors.HEADER + '-----------ROUGE SCORE-----------' + bcolors.ENDC)
                        tf.logging.info(
                            bcolors.OKGREEN + 'R1F %.5f R2F %.5f RLF %.5f R1R %.5f R2R %.5f RLR %.5f' + bcolors.ENDC,
                            final_metrics['rouge-1-f'], final_metrics['rouge-2-f'], final_metrics['rouge-L-f'],
                            final_metrics['rouge-1-r'], final_metrics['rouge-2-r'], final_metrics['rouge-L-r'])
                        tf.logging.info(bcolors.HEADER + '-----------ROUGE SCORE-----------' + bcolors.ENDC)
                    except Exception as e:
                        sys.stderr.write('calculate rouge error %s \n' % e)
                        sys.stderr.flush()
                if FLAGS.decode_bleu:
                    try:
                        ref_file = os.path.join(self._bleu_dec_dir, "reference.txt")
                        decoded_file = os.path.join(self._bleu_dec_dir, "decoded.txt")

                        t0 = time.time()
                        # bleu, bleu1, bleu2, bleu3, bleu4 = calcu_bleu(decoded_file, ref_file)
                        # sys_bleu = sys_bleu_file(decoded_file, ref_file)
                        sys_bleu_perl = sys_bleu_perl_file(decoded_file, ref_file)
                        t1 = time.time()
                        tf.logging.info(self._bleu_dec_dir)
                        tf.logging.info(bcolors.HEADER + '-----------BLEU SCORE-----------' + bcolors.ENDC)
                        # append_results(Exp(FLAGS.proj_name, FLAGS.exp_name, ''), "BLEU "+self._decode_dir, '%f \t %f \t %f \t %f \t %f' % (bleu, bleu1, bleu2, bleu3, bleu4))
                        # tf.logging.info(
                        #     bcolors.OKGREEN + '%f \t %f \t %f \t %f \t %f' + bcolors.ENDC, bleu, bleu1, bleu2, bleu3, bleu4)
                        # tf.logging.info(bcolors.OKGREEN + 'sys_bleu %f' + bcolors.ENDC, sys_bleu)
                        tf.logging.info(bcolors.OKGREEN + 'sys_bleu_perl %s' + bcolors.ENDC, sys_bleu_perl)
                        tf.logging.info(bcolors.OKGREEN + 'Table Format sys_bleu_perl %s' + bcolors.ENDC,
                                        sys_bleu_perl.split(' (')[0].replace(', ', '\t'))
                        tf.logging.info(bcolors.OKGREEN + 'Markdown Format sys_bleu_perl %s' + bcolors.ENDC,
                                        sys_bleu_perl.split(' (')[0].replace(', ', '|'))
                        # append_results(Exp(FLAGS.proj_name, FLAGS.exp_name, ''), "BLEU "+self._decode_dir, sys_bleu_perl.split(' (')[0])
                        tf.logging.info(bcolors.HEADER + '-----------BLEU SCORE-----------' + bcolors.ENDC)
                        tf.logging.info('calculate BLEU score cost %d seconds', t1 - t0)
                        # final_metrics['sys_bleu'] = sys_bleu
                        # final_metrics['calcu_bleu'] = [bleu, bleu1, bleu2, bleu3, bleu4]
                        final_metrics['sys_bleu_perl'] = sys_bleu_perl
                    except Exception as e:
                        sys.stderr.write('calculate bleu error %s \n' % e)
                        sys.stderr.flush()
                break

            original_article = batch.original_articles[0]  # string
            original_abstract = batch.original_abstracts[0]  # string
            original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

            article_withunks = data.show_art_oovs(original_article, self._vocab, FLAGS.string_split)  # string
            abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab,
                                                   (batch.art_oovs[0] if FLAGS.pointer_gen else None), FLAGS.string_split)  # string

            # Run beam search to get best Hypothesis
            best_hyp, ext_ids = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_hyp.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self._vocab,
                                                 (batch.art_oovs[0] if FLAGS.pointer_gen else None))

            ext_sens = [batch.original_sentences[0][i] for i in ext_ids]

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            decoded_output = ' '.join(decoded_words)  # single string

            print_results(article_withunks, abstract_withunks, decoded_output, ext_sens, counter)

            if counter % 50 == 0:
                upload_test_progress(Exp(FLAGS.proj_name, FLAGS.exp_name, ' '.join(sys.argv)),
                                     counter / self._batcher._total_lines,
                                     counter, self._batcher._total_lines, self.step, self._decode_dir)

            if FLAGS.single_pass:
                if FLAGS.decode_rouge:
                    self.write_for_rouge(original_abstract_sents, decoded_words, ext_sens, counter, num_dict)
                if FLAGS.decode_bleu:
                    self.write_for_bleu(original_abstract_sents, decoded_words)
                counter += 1  # this is how many examples we've decoded
            else:
                self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists,
                                       best_hyp.p_gens)  # write info to .json file for visualization tool
                # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
                t1 = time.time()
                if t1 - t0 > SECS_UNTIL_NEW_CKPT:
                    tf.logging.info(
                        'We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint',
                        t1 - t0)
                    _ = util.load_ckpt(self._saver, self._sess)
                    t0 = time.time()
        return final_metrics

    def write_for_rouge(self, reference_sents, decoded_words, ext_sens, ex_index, num_dict=None):
        """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

        Args:
          reference_sents: list of strings
          decoded_words: list of strings
          ex_index: int, the index with which to label the files
        """
        # First, divide decoded output into sentences
        decoded_sents = [s.strip() for s in re.split('[.;?!。？！]', ' '.join(decoded_words))]
        decoded_sents = [s for s in decoded_sents if s != '']

        reference_sents = [s.strip() for s in re.split('[.;?!。？！]', ' '.join(reference_sents))]
        reference_sents = [s for s in reference_sents if s != '']

        extracted_sents = [s for s in ext_sens if s != '']

        # pyrouge calls a perl script that puts the data into HTML files.
        # Therefore we need to make our output HTML safe.
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        reference_sents = [make_html_safe(w) for w in reference_sents]
        # extracted_sents = [make_html_safe(w) for w in extracted_sents]

        # Write to file
        ref_file = os.path.join(self._rouge_ref_dir, "%06d_reference.txt" % ex_index)
        decoded_file = os.path.join(self._rouge_dec_dir, "%06d_decoded.txt" % ex_index)
        extracted_file = os.path.join(self._rouge_ext_dir, "%06d_decoded.txt" % ex_index)
        try:
            with open(ref_file, "w", encoding='utf8') as f:
                for idx, sent in enumerate(reference_sents):
                    f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
            with open(decoded_file, "w", encoding='utf8') as f:
                for idx, sent in enumerate(decoded_sents):
                    f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")
            with open(extracted_file, "w", encoding='utf8') as f:
                for idx, sent in enumerate(extracted_sents):
                    f.write(sent) if idx == len(extracted_sents) - 1 else f.write(sent + "\n")
        except Exception as e:
            print('decode output error %d %s', (ex_index, e))
        if FLAGS.lang == 'zh':
            # Write to file
            num_ref_file = os.path.join(self._rouge_num_ref_dir, "%06d_reference.txt" % ex_index)
            num_decoded_file = os.path.join(self._rouge_num_dec_dir, "%06d_decoded.txt" % ex_index)
            dec = []
            ref = []
            with open(num_ref_file, "w") as f:
                for idx, sent in enumerate(reference_sents):
                    sent, num_dict = self.convert_to_num_sentence(sent, num_dict)
                    ref.append(sent)
                    f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
            with open(num_decoded_file, "w") as f:
                for idx, sent in enumerate(decoded_sents):
                    sent, num_dict = self.convert_to_num_sentence(sent, num_dict)
                    dec.append(sent)
                    f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

    def convert_to_num_sentence(self, sent: str, num_dict: dict):
        num_sent = []
        for w in sent.split(' '):
            if w in num_dict['w2i']:
                num_sent.append(str(num_dict['w2i'][w]))
            else:
                i = num_dict['counter'] + 1
                num_dict['counter'] = i
                num_dict['w2i'][w] = i
                num_sent.append(str(i))
        return str(' '.join(num_sent)), num_dict

    def write_for_bleu(self, reference_sents, decoded_words):
        """Write output to file in correct format for eval with bleu. This is called in single_pass mode.

        Args:
          reference_sents: list of strings
          decoded_words: list of strings
          ex_index: int, the index with which to label the files
        """
        # First, divide decoded output into sentences
        reference_sentence = ' '.join(reference_sents).replace('\n', '')
        decoded_sentence = ' '.join(decoded_words).replace('\n', '')
        # Write to file
        ref_file = os.path.join(self._bleu_dec_dir, "reference.txt")
        decoded_file = os.path.join(self._bleu_dec_dir, "decoded.txt")

        with open(ref_file, "a", encoding="utf8") as f:
            f.write(reference_sentence + "\n")
        with open(decoded_file, "a", encoding="utf8") as f:
            f.write(decoded_sentence + "\n")

    def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
        """Write some data to json file, which can be read into the in-browser attention visualizer tool:
          https://github.com/abisee/attn_vis

        Args:
          article: The original article string.
          abstract: The human (correct) abstract string.
          attn_dists: List of arrays; the attention distributions.
          decoded_words: List of strings; the words of the generated summary.
          p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
        """
        article_lst = article.split()  # list of words
        decoded_lst = decoded_words  # list of decoded words
        to_write = {
            'article_lst': [make_html_safe(t) for t in article_lst],
            'decoded_lst': [make_html_safe(t) for t in decoded_lst],
            'abstract_str': make_html_safe(abstract),
            'attn_dists': attn_dists
        }
        if FLAGS.pointer_gen:
            to_write['p_gens'] = p_gens
        output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
        with open(output_fname, 'w') as output_file:
            json.dump(to_write, output_file)
        tf.logging.info('Wrote visualization data to %s', output_fname)


def print_results(article, abstract, decoded_output, ext_sens, counter=None):
    """Prints the article, the reference summmary and the decoded summary to screen"""
    tf.logging.info('ARTICLE  : %s', article)
    tf.logging.info('REFERENCE: %s', abstract)
    tf.logging.info('GENERATED: %s', decoded_output)
    tf.logging.info('EXTRACTED: %s', ext_sens)
    if counter is not None:
        tf.logging.info("------------%d------------", counter)
    else:
        tf.logging.info("------------------------")


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    """Log ROUGE results to screen and write to file.

    Args:
      results_dict: the dictionary returned by pyrouge
      dir_to_write: the directory where we will write the results to"""
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    tf.logging.info(log_str)  # log to screen
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    tf.logging.info("Writing final ROUGE results to %s...", results_file)
    with open(results_file, "a") as f:
        f.write('=====================')
        f.write(log_str)


def get_decode_dir_name(ckpt_name):
    """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

    if "train" in FLAGS.data_path:
        dataset = "train"
    elif "val" in FLAGS.data_path:
        dataset = "val"
    elif "test" in FLAGS.data_path:
        dataset = "test"
    else:
        raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
    dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (
    dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
    if ckpt_name is not None:
        dirname += "_%s" % ckpt_name
    return dirname
