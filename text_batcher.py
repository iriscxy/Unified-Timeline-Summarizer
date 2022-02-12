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

"""This file contains code to process data into batches"""

import importlib
import queue as Queue
import time
from multiprocessing import Process
from random import shuffle
from subprocess import check_output
from threading import Thread
from glob import glob
from typing import List, Optional
from multiprocessing import Queue as mQueue

import numpy as np
import tensorflow as tf

import data

FLAGS = tf.app.flags.FLAGS


def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])


def split_str_to_word(input_str: str) -> List[str]:
    if FLAGS.string_split == 'char':
        words = list(input_str)
    else:
        words = input_str.split(' ' if FLAGS.string_split == 'space' else FLAGS.string_split)
    return words


class Example(object):
    """Class representing a train/val/test example for text summarization."""

    def __init__(self, article: str, abstract_sentences: List[str], sentences, extract_label, vocab: data.Vocab, max_enc_steps, max_dec_steps):
        """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        Args:
          article: source text; a string. each token is separated by a single space.
          abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
          vocab: Vocabulary object
          hps: hyperparameters

        """
        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        # Process the article
        article_words = split_str_to_word(' '.join(article))
        if len(article_words) > max_enc_steps:
            article_words = article_words[:max_enc_steps]
        self.enc_len = len(article_words)  # store the length after truncation but before padding
        # self.enc_input = [vocab.word2id(w) if '@' not in w else vocab.word2id(w[:3]) for w in
        #                   article_words if w != '']  # list of word ids; OOVs are represented by the id for UNK token
        self.enc_input = [vocab.word2id(w) for w in article_words if w != '']

        # Process the event
        article = article[:FLAGS.max_art_lens]
        for i in range(len(article)):
            article[i] = ' '.join(article[i].split()[:FLAGS.max_hredsent_lens])
        self.hred_art_len = len(article)
        self.hred_enc_lens, self.hred_enc_inputs = [], []
        self.hred_enc_inputs_org = []
        for sent in article:
            sent = sent.split()
            self.hred_enc_inputs_org.append(sent)
            self.hred_enc_inputs.append([vocab.word2id(w) for w in sent])
            self.hred_enc_lens.append(len(sent))


        # Process the sentences
        self.sentences = []
        self.sentences_len = []
        self.sentence_padding_mask = []
        self.sent_id_mask = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]
        for i, sen in enumerate(sentences):
            sen = sen.split()[:FLAGS.max_sen_len]
            self.sentences.append([vocab.word2id(w) for w in sen])
            self.sentences_len.append(len(sen))
            if len(sen) > 0:
                self.sentence_padding_mask.append(1)
            else:
                self.sentence_padding_mask.append(0)
                self.sent_id_mask[i] = -1

        xy_label = np.zeros([4, 24], dtype=np.float32)
        j = 1
        for i in range(len(extract_label) - 1):
            xy_label[j][extract_label[i]] = 1
            j += 1

        self.ext_input = xy_label

        self.ext_target = extract_label


        # Process the abstract
        abstract = ' '.join(abstract_sentences)  # string
        abstract_words = split_str_to_word(abstract)
        abs_ids = [vocab.word2id(w) for w in abstract_words if w != '']

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, max_dec_steps, start_decoding,
                                                                 stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if FLAGS.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)
            # self.hred_enc_input_extend_vocab, self.hred_article_oovs = data.hred_article2ids(self.hred_enc_inputs_org, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, max_dec_steps, start_decoding,
                                                        stop_decoding)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_sentences = sentences
        if FLAGS.string_split == 'char':
            self.original_abstract_sents = abstract_sentences[0]
        else:
            self.original_abstract_sents = abstract_sentences

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

        Args:
          sequence: List of ids (integers)
          max_len: integer
          start_id: integer
          stop_id: integer

        Returns:
          inp: sequence length <=max_len starting with start_id
          target: sequence same length as input, ending with stop_id only if there was no truncation
        """
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if FLAGS.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)

    def pad_sen_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        for i in range(len(self.sentences)):
            sen_len = len(self.sentences[i])
            if sen_len < max_len:
                self.sentences[i] += [pad_id] * (max_len-sen_len)

    def pad_hred_input(self, max_len, max_art_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        for i in range(len(self.hred_enc_lens)):
            hred_sent_len = len(self.hred_enc_inputs[i])
            if hred_sent_len < max_len:
                self.hred_enc_inputs[i] += [pad_id] * (max_len-hred_sent_len)
                # if FLAGS.pointer_gen:
                #     self.hred_enc_input_extend_vocab[i] += [pad_id] * (max_len-hred_sent_len)

        if len(self.hred_enc_inputs) < max_art_len:
            org_len = len(self.hred_enc_inputs)
            for _ in range(max_art_len-org_len):
                self.hred_enc_inputs.append([pad_id] * max_len)
                self.hred_enc_lens.append(0)
                # if FLAGS.pointer_gen:
                #     self.hred_enc_input_extend_vocab.append([pad_id] * max_len)



class Batch(object):
    """Class representing a minibatch of train/val/test examples for text summarization."""

    def __init__(self, example_list, hps, vocab):
        """Turns the example_list into a Batch object.

        Args:
           example_list: List of Example objects
           hps: hyperparameters
           vocab: Vocabulary object
        """
        self.pad_id = vocab.word2id(data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list, hps)  # initialize the input to the encoder
        self.init_hred_seq(example_list, hps)
        self.init_ext_seq(example_list, hps)
        self.init_decoder_seq(example_list, hps)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list: List[Example], hps):
        """Initializes the following:
            self.enc_batch:
              numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
            self.enc_lens:
              numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
            self.enc_padding_mask:
              numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.

          If hps.pointer_gen, additionally initializes the following:
            self.max_art_oovs:
              maximum number of in-article OOVs in the batch
            self.art_oovs:
              list of list of in-article OOVs (strings), for each example in the batch
            self.enc_batch_extend_vocab:
              Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
        """
        # Determine the maximum length of the encoder input sequence in this batch
        # max_enc_seq_len = max([ex.enc_len for ex in example_list])
        max_enc_seq_len = FLAGS.max_enc_steps

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1



        # For pointer-generator mode, need to store some extra info
        if hps.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]


    def init_hred_seq(self, example_list, hps):
        for ex in example_list:
            ex.pad_hred_input(hps.max_hredsent_lens, hps.max_art_lens, self.pad_id)

        self.hred_batch = np.zeros((hps.batch_size, hps.max_art_lens, hps.max_hredsent_lens), dtype=np.int32)
        self.hred_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.art_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.hred_padding_mask = np.zeros((hps.batch_size, hps.max_art_lens * hps.max_hredsent_lens), dtype=np.float32)
        self.hred_con_padding_mask = np.zeros((hps.batch_size, hps.max_art_lens), dtype=np.float32)

        for i, ex in enumerate(example_list):
            for j, hred_sent in enumerate(ex.hred_enc_inputs):
                self.hred_batch[i, j, :] = hred_sent[:]

            self.hred_lens[i] = FLAGS.max_art_lens * FLAGS.max_hredsent_lens
            self.art_lens[i] = ex.hred_art_len
            for j, hred_len in enumerate(ex.hred_enc_lens):
                for k in range(hred_len):
                    self.hred_padding_mask[i][j * hps.max_art_lens + k] = 1
            for j in range(ex.hred_art_len):
                self.hred_con_padding_mask[i][j] = 1

        # if hps.pointer_gen:
        #     # Determine the max number of in-article OOVs in this batch
        #     self.hred_max_art_oovs = max([len(ex.hred_article_oovs) for ex in example_list])
        #     # Store the in-article OOVs themselves
        #     self.hred_art_oovs = [ex.hred_article_oovs for ex in example_list]
        #     # Store the version of the enc_batch that uses the article OOV ids
        #     self.hred_batch_extend_vocab = np.zeros((hps.batch_size, hps.max_art_lens, hps.max_hredsent_lens),
        #                                             dtype=np.int32)
        #     for i, ex in enumerate(example_list):
        #         for j, hred_enc in enumerate(ex.hred_enc_input_extend_vocab):
        #             self.hred_batch_extend_vocab[i, j, :] = hred_enc[:]

    def init_ext_seq(self, example_list, hps):
        for ex in example_list:
            ex.pad_sen_input(FLAGS.max_sen_len, self.pad_id)

        self.sen_batch = np.zeros((hps.batch_size, 24, FLAGS.max_sen_len), dtype=np.int32)
        self.sen_lens = np.zeros((hps.batch_size, 24), dtype=np.int32)
        self.sen_padding_mask = np.zeros((hps.batch_size, 24), dtype=np.float32)

        self.ext_input = np.zeros((hps.batch_size, FLAGS.max_ext_steps, 24), dtype=np.float32)
        self.ext_target_batch = np.zeros((hps.batch_size, FLAGS.max_ext_steps), dtype=np.int32)

        self.sent_id_mask = np.zeros((hps.batch_size, 24), dtype=np.int32)

        for i, ex in enumerate(example_list):
            self.sen_batch[i, :, :] = ex.sentences
            self.sen_lens[i, :] = ex.sentences_len[:]
            self.sen_padding_mask[i, :] = ex.sentence_padding_mask

            self.ext_input[i, :, :] = ex.ext_input
            self.ext_target_batch[i, :] = ex.ext_target[:]

            self.sent_id_mask[i, :] = ex.sent_id_mask



    def init_decoder_seq(self, example_list: List[Example], hps):
        """Initializes the following:
            self.dec_batch:
              numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
            self.target_batch:
              numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
            self.dec_padding_mask:
              numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
            """
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
        self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list: List[Example]):
        """Store the original article and abstract strings in the Batch object"""
        self.original_articles = [ex.original_article for ex in example_list]  # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]  # list of list of lists
        self.original_sentences = [ex.original_sentences for ex in example_list]

class Batcher(object):
    """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, hps, single_pass):
        """Initialize the batcher. Start threads that process the data into batches.

        Args:
          data_path: tf.Example filepattern.
          vocab: Vocabulary object
          hps: hyperparameters
          single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
        """
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass

        if FLAGS.dataset_size is None or FLAGS.dataset_size < 0:
            tf.logging.info('counting file lines')
            lines = 0
            for f in glob(data_path):
                lines += wc(f)
            self._total_lines = lines
            FLAGS.dataset_size = lines
        else:
            tf.logging.info('using FLAGS.dataset_size as _total_lines')
            self._total_lines = FLAGS.dataset_size
        self._batch_num = 0

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = mQueue(self.BATCH_QUEUE_MAX * self._hps.batch_size)
        self._json_queue = mQueue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 16  # num threads to fill example queue
            self._num_batch_q_threads = 4  # num threads to fill batch queue
            self._bucketing_cache_size = 100  # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self.fill_json_thread = Thread(target=self.fill_json_queue)
        self.fill_json_thread.daemon = True
        self.fill_json_thread.start()
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Process(
                target=self.fill_example_queue, args=(self._hps.max_enc_steps, self._hps.max_dec_steps, self._json_queue)))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()
        self.kill_watch_thread = False
        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def close(self):
        self.kill_watch_thread = True
        for idx, t in enumerate(self._example_q_threads):
            t.terminate()
            t.join()

    def next_batch(self) -> Optional[Batch]:
        """Return a Batch from the batch queue.

        If mode='decode' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.

        Returns:
          batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
        """
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.logging.warning(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch

        self._batch_num += 1
        if self._batch_num * self._hps.batch_size > self._total_lines:
            self._batch_num = 0

        return batch

    def progress(self):
        return self._batch_num * self._hps.batch_size / self._total_lines

    def total_data(self):
        return self._total_lines

    def fill_example_queue(self, max_enc_steps, max_dec_steps, json_queue):
        """Reads data from file and processes into Examples which are then placed into the example queue."""
        while True:
            (article, abstract, sentences, extract_label) = json_queue.get()
            example = Example(article, [abstract], sentences, extract_label, self._vocab, max_enc_steps, max_dec_steps)  # Process into an Example.
            self._example_queue.put(example)  # place the Example in the example queue.

    def fill_batch_queue(self):
        """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

        In decode mode, makes batches that each contain a single example repeated.
        """
        while True:
            if 'decode' not in FLAGS.mode:
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self._hps.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                # inputs = sorted(inputs, key=lambda inp: inp.enc_len)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self._hps.batch_size):
                    batches.append(inputs[i:i + self._hps.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))

            else:  # beam search decode mode
                ex = self._example_queue.get()
                b = [ex for _ in range(self._hps.batch_size)]
                self._batch_queue.put(Batch(b, self._hps, self._vocab))

    def fill_json_queue(self):
        input_gen = self.text_generator(data.json_generator(self._data_path, self._single_pass))

        while True:
            try:
                (article, abstract, sentences, extract_label) = next(input_gen)
            except StopIteration:  # if there are no more examples:
                tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    tf.logging.info(
                        "single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")
            # if article.strip() == '' or abstract == '':
            #     tf.logging.warning('article or abstract is missing')
            #     continue
            # article = article.lower()
            # abstract = abstract.lower()
            self._json_queue.put((article, abstract, sentences, extract_label))

    def watch_threads(self):
        """Watch example queue and batch queue threads and restart if dead."""
        while True:
            time.sleep(10)
            if self.kill_watch_thread:
                tf.logging.info('Watching thread exit!')
                break
            if not self.fill_json_thread.is_alive():
                tf.logging.error('Found fill json queue thread dead. Restarting.')
                self.fill_json_thread = Thread(target=self.fill_json_queue)
                self.fill_json_thread.daemon = True
                self.fill_json_thread.start()
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Process(target=self.fill_example_queue,
                                    args=(self._hps.max_enc_steps, self._hps.max_dec_steps, self._json_queue))
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def text_generator(self, example_generator):
        """Generates article and abstract text from tf.Example.

        Args:
          example_generator: a generator of tf.Examples from file. See data.example_generator"""
        while True:
            e = next(example_generator)
            try:
                preprocessors = importlib.import_module('data_process.batcher_preprocessor')
                preprocessor = getattr(preprocessors, FLAGS.preprocessor)
                article_text, abstract_text, sentences, extract_label = preprocessor(e, FLAGS)
            except (ValueError, KeyError):
                tf.logging.error('Failed to get article or abstract from example')
                continue
            if len(article_text) == 0 or len(abstract_text) == 0 or len(sentences) == 0:  # See https://github.com/abisee/pointer-generator/issues/1
                continue
                # tf.logging.warning('Found an example with empty article text. Skipping it.')
            elif len(extract_label) < 4:
                continue
            else:
                yield (article_text, abstract_text, sentences, extract_label)
