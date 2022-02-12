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

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import pickle
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from ext_attention_decoder import ext_attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import variable_scope

from opennmt.encoders import *
from attention_gru_cell import AttentionGRUCell

FLAGS = tf.flags.FLAGS


class SummarizationModel(object):
    """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        hps = self._hps

        # encoder part
        self._enc_batch = tf.placeholder(tf.int32, [FLAGS.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [FLAGS.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, None], name='enc_padding_mask')

        if FLAGS.pointer_gen:
            self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [FLAGS.batch_size, None],
                                                          name='enc_batch_extend_vocab')
            self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

        # hierarchical encoder part
        self._hred_enc_batch = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.max_art_lens, FLAGS.max_hredsent_lens],
                                              name='hred_enc_batch')
        self._hred_enc_lens = tf.placeholder(tf.int32, [FLAGS.batch_size], name='hred_enc_lens')
        self._hred_art_lens = tf.placeholder(tf.int32, [FLAGS.batch_size], name='hred_art_lens')
        self._hred_enc_padding_mask = tf.placeholder(tf.float32,
                                                     [FLAGS.batch_size, FLAGS.max_art_lens * FLAGS.max_hredsent_lens],
                                                     name='hred_enc_padding_mask')
        self._hred_con_padding_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.max_art_lens],
                                                     name='hred_con_padding_mask')

        # extractive part
        self._sen_batch = tf.placeholder(tf.int32, [FLAGS.batch_size, 24, FLAGS.max_sen_len], name='sen_batch')
        self._sen_lens = tf.placeholder(tf.int32, [FLAGS.batch_size, 24], name='sen_lens')
        self._sen_padding_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, 24], name='sen_padding_mask')

        self._ext_input = tf.placeholder(tf.float32, [FLAGS.batch_size, hps.max_ext_steps, 24], name='ext_dec_batch')
        self._ext_target_batch = tf.placeholder(tf.int32, [FLAGS.batch_size, FLAGS.max_ext_steps],
                                                name='ext_target_batch')

        # inconsistent part
        self._sent_id_mask = tf.placeholder(tf.int32, [FLAGS.batch_size, 24], name='sen_id_mask')

        # decoder part
        self._dec_batch = tf.placeholder(tf.int32, [FLAGS.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [FLAGS.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, hps.max_dec_steps],
                                                name='dec_padding_mask')

        if "decode" in FLAGS.mode and FLAGS.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [FLAGS.batch_size, None], name='prev_coverage')

    def _make_feed_dict(self, batch, just_enc=False):
        """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

        Args:
          batch: Batch object
          just_enc: Boolean. If True, only feed the parts needed for the encoder.
        """
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
        if FLAGS.pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self._max_art_oovs] = batch.max_art_oovs

        feed_dict[self._hred_enc_batch] = batch.hred_batch  #
        feed_dict[self._hred_enc_lens] = batch.hred_lens  #
        feed_dict[self._hred_art_lens] = batch.art_lens  #
        feed_dict[self._hred_enc_padding_mask] = batch.hred_padding_mask  #
        feed_dict[self._hred_con_padding_mask] = batch.hred_con_padding_mask  #

        feed_dict[self._sen_batch] = batch.sen_batch
        feed_dict[self._sen_lens] = batch.sen_lens
        feed_dict[self._sen_padding_mask] = batch.sen_padding_mask

        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask

            feed_dict[self._ext_input] = batch.ext_input
            feed_dict[self._sent_id_mask] = batch.sent_id_mask
            feed_dict[self._ext_target_batch] = batch.ext_target_batch
        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
          fw_state, bw_state:
            Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        with tf.variable_scope('encoder'):
            if FLAGS.encoder in ['mix', 'rnn']:
                cell_fw = tf.contrib.rnn.LSTMCell(FLAGS.hidden_dim, initializer=self.rand_unif_init,
                                                  state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(FLAGS.hidden_dim, initializer=self.rand_unif_init,
                                                  state_is_tuple=True)
                (rnn_encoder_outputs, (rnn_fw_st, rnn_bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                encoder_inputs,
                                                                                                dtype=tf.float32,
                                                                                                sequence_length=seq_len,
                                                                                                swap_memory=True)
                rnn_encoder_outputs = tf.concat(axis=2,
                                                values=rnn_encoder_outputs)  # concatenate the forwards and backwards states
            if FLAGS.encoder in ['mix', 'transformer']:
                import opennmt as onmt
                fusion_self_encoder = onmt.encoders.SelfAttentionEncoder(2, num_units=2 * FLAGS.hidden_dim, num_heads=8)
                transformer_encoder_outputs, final_state, _ = fusion_self_encoder.encode(encoder_inputs, seq_len)
                transformer_fw_st = tf.contrib.rnn.LSTMStateTuple(final_state[0], final_state[0])
                transformer_bw_st = tf.contrib.rnn.LSTMStateTuple(final_state[1], final_state[1])
            if FLAGS.encoder == 'mix':
                encoder_outputs = tf.concat([rnn_encoder_outputs, transformer_encoder_outputs], axis=2)
                fw_st = rnn_fw_st
                bw_st = rnn_bw_st
            elif FLAGS.encoder == 'transformer':
                encoder_outputs = transformer_encoder_outputs
                transformer_encoder_outputs.set_shape([FLAGS.batch_size, None, 2 * FLAGS.hidden_dim])
                fw_st = transformer_fw_st
                bw_st = transformer_bw_st
            else:
                encoder_outputs = rnn_encoder_outputs
                fw_st = rnn_fw_st
                bw_st = rnn_bw_st
        return encoder_outputs, fw_st, bw_st

    def _reduce_states(self, fw_st, bw_st):
        """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

        Args:
          fw_st: LSTMStateTuple with hidden_dim units.
          bw_st: LSTMStateTuple with hidden_dim units.

        Returns:
          state: LSTMStateTuple with hidden_dim units.
        """
        hidden_dim = FLAGS.hidden_dim
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and reduce the state
            input_dim = fw_st.c.get_shape()[-1]
            w_reduce_c = tf.get_variable('w_reduce_c', [input_dim * 2, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [input_dim * 2, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=reuse):
            features = [
                fact_vec * prev_memory,
                fact_vec,
                prev_memory]

            feature_vec = tf.concat(features, 1)
            attention = tf.contrib.layers.fully_connected(feature_vec,
                                                          FLAGS.emb_dim,
                                                          activation_fn=tf.nn.tanh,
                                                          reuse=reuse, scope="fc1")
            attention = tf.contrib.layers.fully_connected(attention,
                                                          1,
                                                          activation_fn=None,
                                                          reuse=reuse, scope="fc2")
        return attention

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

        attentions = [tf.squeeze(self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
                      for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        reuse = True if hop_index > 0 else False

        # concatenate fact vectors and attentions for input into attGRU
        gru_inputs = tf.concat([fact_vecs, attentions], 2)

        with tf.variable_scope('attention_gru', reuse=reuse):
            episode, episode_1 = tf.nn.dynamic_rnn(AttentionGRUCell(self._hps.hidden_dim * 2), gru_inputs,
                                                   dtype=np.float32)

        return episode, episode_1

    def _memory_module(self, enc_states, content):
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            # generate n_hops episodes
            prev_memory = content

            for i in range(FLAGS.memory_layer):
                # get a new episode
                episode, episode_1 = self.generate_episode(prev_memory, content, enc_states, i)

                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode_1, content], 1),
                                                  self._hps.hidden_dim * 2, activation=tf.nn.relu)

            return episode

    def _add_decoder(self, inputs):
        """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

        Args:
          inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

        Returns:
          outputs: List of tensors; the outputs of the decoder
          out_state: The final state of the decoder
          attn_dists: A list of tensors; the attention distributions
          p_gens: A list of scalar tensors; the generation probabilities
          coverage: A tensor, the current coverage vector
        """
        cell = tf.contrib.rnn.LSTMCell(FLAGS.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

        prev_coverage = self.prev_coverage if "decode" in FLAGS.mode and FLAGS.coverage else None  # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time

        outputs, out_state, attn_dists, event_dists, p_gens, coverage = attention_decoder(inputs, self._dec_in_state,
                                                                                          self._enc_states,
                                                                                          self._enc_padding_mask,
                                                                                          self.event_local_states,
                                                                                          self.event_global_states,
                                                                                          self._hred_con_padding_mask,
                                                                                          FLAGS.max_hredsent_lens,
                                                                                          self.time_key,
                                                                                          cell,
                                                                                          initial_state_attention=(
                                                                                                  "decode" in FLAGS.mode),
                                                                                          pointer_gen=FLAGS.pointer_gen,
                                                                                          use_coverage=FLAGS.coverage,
                                                                                          prev_coverage=prev_coverage)

        return outputs, out_state, attn_dists, event_dists, p_gens, coverage

    def _add_ext_attention_decoder(self, inputs):
        cell = tf.contrib.rnn.LSTMCell(FLAGS.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

        prev_coverage = self.prev_coverage if "decode" in FLAGS.mode and FLAGS.coverage else None  # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time

        out_state, attn_dists = ext_attention_decoder(inputs, self._dec_in_state, self.sen_states,
                                                      self._sen_padding_mask, cell, initial_state_attention=(
                    "decode" in FLAGS.mode), pointer_gen=FLAGS.pointer_gen)

        return out_state, attn_dists

    def _calc_final_dist(self, vocab_dists, attn_dists):
        """Calculate the final distribution, for the pointer-generator model

        Args:
          vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
          attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

        Returns:
          final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
        """
        with tf.variable_scope('final_distribution'):
            # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
            vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]
            attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]

            # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
            extended_vsize = self._vocab.size() + self._max_art_oovs  # the maximum (over the batch) size of the extended vocabulary
            extra_zeros = tf.zeros((FLAGS.batch_size, self._max_art_oovs))
            vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in
                                    vocab_dists]  # list length max_dec_steps of shape (batch_size, extended_vsize)

            # Project the values in the attention distributions onto the appropriate entries in the final distributions
            # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
            # This is done for each decoder timestep.
            # This is fiddly; we use tf.scatter_nd to do the projection
            batch_nums = tf.range(0, limit=FLAGS.batch_size)  # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
            attn_len = tf.shape(self._enc_batch_extend_vocab)[1]  # number of states we attend over
            batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
            indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
            shape = [FLAGS.batch_size, extended_vsize]
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in
                                    attn_dists]  # list length max_dec_steps (batch_size, extended_vsize)

            # Add the vocab distributions and the copy distributions together to get the final distributions
            # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
            # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
            final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                           zip(vocab_dists_extended, attn_dists_projected)]

            return final_dists

    def _add_emb_vis(self, embedding_var):
        """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
        https://www.tensorflow.org/get_started/embedding_viz
        Make the vocab metadata file, then make the projector config file pointing to it."""
        train_dir = os.path.join(FLAGS.log_root, "train")
        vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
        self._vocab.write_metadata(vocab_metadata_path)  # write metadata file
        summary_writer = tf.summary.FileWriter(train_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = vocab_metadata_path
        projector.visualize_embeddings(summary_writer, config)

    def _add_inconsistent_loss(self, sent_dist, event_dist):
        hps = self._hps
        event_dist = tf.split(event_dist, FLAGS.max_ext_steps, 1)

        batch_nums = tf.expand_dims(tf.range(0, limit=hps.batch_size), 1)  # shape (batch_size, 1)
        indices = tf.stack((tf.tile(batch_nums, [1, 24]), self._sent_id_mask),
                           axis=2)  # shape (batch_size, enc_len, 2)

        losses = []
        inconsistent_topk = 4
        batch_nums_tilek = tf.tile(batch_nums, [1, inconsistent_topk])  # shape (batch_size, k)
        # To compute inconsistent loss = -log(sent_prob)*(word_attn)
        for s_attn_dist, e_attn_dist in zip(sent_dist, event_dist):
            # All pad tokens will get probability of 0.0 since the sentence id is -1 (gather_nd will produce 0.0 for invalid indices)
            selector_probs_projected = tf.gather_nd(tf.reshape(e_attn_dist, [FLAGS.batch_size, 24]),
                                                    indices)  # shape (batch_size, enc_len)

            topk_w, topk_w_id = tf.nn.top_k(s_attn_dist, inconsistent_topk)  # shape (batch_size, topk)
            topk_w_indices = tf.stack((batch_nums_tilek, topk_w_id), axis=2)  # shape (batch_size, topk, 2)
            topk_s = tf.gather_nd(selector_probs_projected, topk_w_indices)  # shape (batch_size, topk)
            # mean first than log
            loss_one_step = tf.reduce_mean(topk_w * topk_s, 1)  # shape (batch_size,)
            loss_one_step = -tf.log(loss_one_step + 1e-10)  # shape (batch_size,)

            # loss_one_step *= self._rewriter._dec_padding_mask[:, dec_step]  # shape (batch_size,)
            losses.append(loss_one_step)
        loss = tf.reduce_mean(sum(losses) / FLAGS.max_ext_steps)
        return loss

    def _add_seq2seq(self):
        """Add the whole sequence-to-sequence model to the graph."""
        vsize = self._vocab.size()  # size of the vocabulary

        with tf.variable_scope('seq2seq'):
            tf.logging.info('Building embeddings...')
            self.rand_unif_init = tf.random_uniform_initializer(-FLAGS.rand_unif_init_mag, FLAGS.rand_unif_init_mag,
                                                                seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=FLAGS.trunc_norm_init_std)

            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable('embedding', [vsize, FLAGS.emb_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
                if FLAGS.pretrain_emb_pkl is not None:
                    with open(FLAGS.pretrain_emb_pkl, 'rb') as f:
                        embeddings = pickle.load(f)
                    pretrained_word_embeddings = np.array(embeddings)
                    embedding = embedding.assign(pretrained_word_embeddings[:FLAGS.vocab_size])
                if FLAGS.mode == "train": self._add_emb_vis(embedding)  # add to tensorboard

                # input embedding
                emb_event_inputs = tf.nn.embedding_lookup(embedding,
                                                          self._hred_enc_batch)
                emb_enc_inputs = tf.nn.embedding_lookup(embedding,
                                                        self._enc_batch)  # tensor with shape (batch_size, max_enc_steps, emb_size)
                emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch,
                                                                                           axis=1)]  # list length max_dec_steps containing shape (batch_size, emb_size)

                emb_sen_inputs = tf.nn.embedding_lookup(embedding, self._sen_batch)

                # time embedding
                time_embedding = tf.get_variable('time_embedding', [self._hps.max_art_lens, FLAGS.emb_dim],
                                                 dtype=tf.float32, initializer=self.trunc_norm_init)
                # now I want to a [art_len, emb_size] ==> [batch_size, art_len, sent_len, emb_size]
                time_position = tf.expand_dims(tf.expand_dims(time_embedding, 1), 0)
                time_position = tf.tile(time_position, [self._hps.batch_size, 1, self._hps.max_hredsent_lens, 1])

                self.time_key = time_embedding

                emb_event_inputs = tf.concat(values=[emb_event_inputs, time_position], axis=-1)
                emb_event_inputs = tf.layers.dense(emb_event_inputs, self._hps.emb_dim)

            tf.logging.info('Building main encoders...')
            with tf.variable_scope('s2s_encoder'):
                enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
                self._enc_states = enc_outputs
                self._dec_in_state = self._reduce_states(fw_st, bw_st)

            with tf.variable_scope('hred_encoder'):
                seq_len = FLAGS.max_art_lens * FLAGS.max_hredsent_lens
                event_outputs, ev_fw_st, ev_bw_st = self._add_encoder(
                    tf.reshape(emb_event_inputs, [self._hps.batch_size, seq_len, self._hps.emb_dim]),
                    self._hred_enc_lens)
            # word-level [batch_size, art_len*sen_len, hidden_dim*2]

            event_states = tf.reduce_mean(tf.reshape(event_outputs,
                                                     [FLAGS.batch_size, FLAGS.max_art_lens, FLAGS.max_hredsent_lens,
                                                      FLAGS.hidden_dim * 2]), 2)
            # event-level [batch_size, art_len, hidden_dim*2]

            self.event_local_states = event_states

            with tf.variable_scope('sen_encoder'):
                _, sen_fw_st, sen_bw_st = self._add_encoder(
                    tf.reshape(emb_sen_inputs, [self._hps.batch_size * 24, -1, self._hps.emb_dim]),
                    tf.reshape(self._sen_lens, [self._hps.batch_size * 24]))

                sen_states = tf.layers.dense(tf.concat([sen_fw_st.c, sen_fw_st.h, sen_bw_st.c, sen_bw_st.h], -1),
                                             FLAGS.hidden_dim * 2)
                sen_states = tf.reshape(sen_states, [FLAGS.batch_size, 24, FLAGS.hidden_dim * 2])
            with tf.variable_scope('sen_rnn_encoder'):
                sen_states, _, _ = self._add_encoder(sen_states, tf.constant([24] * FLAGS.batch_size))

            with tf.variable_scope('memory'):
                q_vec = tf.concat([self._dec_in_state[0], self._dec_in_state[1]], -1)
                self.sen_states = self._memory_module(sen_states, q_vec)

            emb_ext_inputs = tf.split(tf.matmul(self._ext_input, self.sen_states), self._ext_input.get_shape()[1].value,
                                      1)

            with tf.variable_scope('graph_encoder'):
                relation_matrix = []
                for first_doc_index in range(FLAGS.max_art_lens):
                    row_in_relation_matrix = []
                    for second_doc_index in range(FLAGS.max_art_lens):
                        with tf.variable_scope('relation%d' % first_doc_index, reuse=tf.AUTO_REUSE):
                            first2second = tf.layers.dense(
                                tf.concat([event_states[:, first_doc_index, :], event_states[:, second_doc_index, :]],
                                          1),
                                FLAGS.hidden_dim / FLAGS.num_heads)
                            row_in_relation_matrix.append(first2second)
                    row_in_relation_matrix = tf.stack(row_in_relation_matrix, 1)
                    relation_matrix.append(row_in_relation_matrix)
                relation_matrix = tf.stack(relation_matrix, 1)

                self._relation_matrix = relation_matrix

                self.graph_encoder = SelfAttentionEncoder_with_relation(num_layers=FLAGS.num_layers,
                                                                        num_units=FLAGS.hidden_dim,
                                                                        num_heads=FLAGS.num_heads,
                                                                        ffn_inner_dim=FLAGS.ffn_inner_dim,
                                                                        dropout=FLAGS.dropout,
                                                                        attention_dropout=FLAGS.attention_dropout,
                                                                        relu_dropout=FLAGS.relu_dropout)

                event_repre, _, _ = self.graph_encoder.encode(event_states, relation_matrix,
                                                              sequence_length=self._hred_art_lens,
                                                              mode=FLAGS.mode)

                self.event_global_states = event_repre

            with tf.variable_scope('gen_decoder'):
                decoder_outputs, self._dec_out_state, self.attn_dists, self.event_dists, self.p_gens, self.coverage = self._add_decoder(
                    emb_dec_inputs)
            with tf.variable_scope('ext_decoder'):
                self._ext_out_state, ext_attn_dists = self._add_ext_attention_decoder(emb_ext_inputs)
                self.ext_attn_dists = ext_attn_dists

            tf.logging.info('Building output projections...')
            with tf.variable_scope('output_projection'):
                w = tf.get_variable('w', [FLAGS.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                vocab_scores = []
                for i, output in enumerate(decoder_outputs):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    vocab_scores.append(tf.nn.xw_plus_b(output, w, v))  # apply the linear layer

                vocab_dists = [tf.nn.softmax(s) for s in
                               vocab_scores]  # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.

            tf.logging.info('Calculating final word distributions...')
            if FLAGS.pointer_gen:
                final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
            else:  # final distribution is just vocabulary distribution
                final_dists = vocab_dists

            tf.logging.info('Calculating loss function...')
            if FLAGS.mode in ['train', 'eval']:
                # Calculate the loss
                with tf.variable_scope('consistent_loss'):
                    event_dists = tf.expand_dims(tf.transpose(self.event_dists, [1, 0, 2]), -1)
                    sent_dists = self.ext_attn_dists

                    filter_kernel = variable_scope.get_variable("f_k", [FLAGS.kernel_size, FLAGS.max_art_lens, 1, 1])
                    event_dists = tf.squeeze(
                        tf.nn.conv2d(event_dists, filter_kernel, strides=[1, FLAGS.kernel_size, 1, 1], padding='SAME'))
                    event_dists = tf.reshape(tf.tile(tf.expand_dims(event_dists, 2), [1, 1, 3, 1]),
                                             [FLAGS.batch_size, FLAGS.max_ext_steps, 24])
                    event_dists = tf.nn.softmax(event_dists, -1)
                    self.consistent_loss = self._add_inconsistent_loss(sent_dists, event_dists)

                with tf.variable_scope('ext_loss'):
                    ext_loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
                    batch_nums = tf.range(0, limit=FLAGS.batch_size)  # shape (batch_size)
                    for dec_step, dist in enumerate(ext_attn_dists):
                        targets = self._ext_target_batch[:, dec_step]
                        indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
                        gold_probs = tf.gather_nd(dist,
                                                  indices)  # shape (batch_size). prob of correct words on this step
                        losses = -tf.log(gold_probs + 1e-10)
                        ext_loss_per_step.append(losses)
                    self._loss_ext = tf.reduce_mean(ext_loss_per_step)

                    # self._loss_ext = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    # logits=tf.reshape(ext_attn_dists, [FLAGS.batch_size, FLAGS.max_ext_steps, 24]),
                    # labels=tf.one_hot(self._ext_target_batch, 24)))
                    # self._loss_ext = pairwise_hinge_loss(logits=tf.reshape(ext_attn_dists, [FLAGS.batch_size * FLAGS.max_ext_steps, 24]),
                    #                                      labels=tf.reshape(tf.one_hot(self._ext_target_batch, 24), [FLAGS.batch_size * FLAGS.max_ext_steps, 24]))

                with tf.variable_scope('loss'):
                    if FLAGS.pointer_gen:
                        # Calculate the loss per step
                        # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
                        loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
                        batch_nums = tf.range(0, limit=FLAGS.batch_size)  # shape (batch_size)
                        for dec_step, dist in enumerate(final_dists):
                            targets = self._target_batch[:, dec_step]
                            indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
                            gold_probs = tf.gather_nd(dist,
                                                      indices)  # shape (batch_size). prob of correct words on this step
                            losses = -tf.log(gold_probs + 1e-10)
                            loss_per_step.append(losses)
                            if dec_step % 10 == 0:
                                tf.logging.info('Calculating loss function for decode step %d...', dec_step)

                        # Apply dec_padding_mask and get loss
                        self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

                        metric_prediction = tf.argmax(tf.stack(final_dists, 1), axis=-1)
                        metric_labels = tf.cast(self._target_batch, tf.int64)
                        self.accuracy_metric, self.accuracy_update = tf.metrics.accuracy(metric_labels,
                                                                                         metric_prediction,
                                                                                         weights=self._dec_padding_mask,
                                                                                         name="my_metric")
                        self.recall_at_top_k_metric, self.recall_at_top_k_update = tf.metrics.recall(metric_labels,
                                                                                                     metric_prediction,
                                                                                                     weights=self._dec_padding_mask,
                                                                                                     name="my_metric")
                        self.precision_at_k_metric, self.precision_at_k_update = tf.metrics.precision(metric_labels,
                                                                                                      metric_prediction,
                                                                                                      weights=self._dec_padding_mask,
                                                                                                      name="my_metric")
                        # self.f1_metric, self.f1_update = tf.contrib.metrics.f1_score(metric_labels, metric_prediction, name="my_metric")
                        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
                        self.metric_initializer = tf.variables_initializer(var_list=running_vars)
                    else:  # baseline model
                        self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1),
                                                                      self._target_batch,
                                                                      self._dec_padding_mask)  # this applies softmax internally
                        metric_prediction = tf.argmax(tf.stack(final_dists, 1), axis=-1)
                        metric_labels = tf.cast(self._target_batch, tf.int64)
                        self.accuracy_metric, self.accuracy_update = tf.metrics.accuracy(metric_labels,
                                                                                         metric_prediction,
                                                                                         weights=self._dec_padding_mask,
                                                                                         name="my_metric")
                        self.recall_at_top_k_metric, self.recall_at_top_k_update = tf.metrics.recall(metric_labels,
                                                                                                     metric_prediction,
                                                                                                     weights=self._dec_padding_mask,
                                                                                                     name="my_metric")
                        self.precision_at_k_metric, self.precision_at_k_update = tf.metrics.precision(metric_labels,
                                                                                                      metric_prediction,
                                                                                                      weights=self._dec_padding_mask,
                                                                                                      name="my_metric")
                        # self.f1_metric, self.f1_update = tf.contrib.metrics.f1_score(metric_labels, metric_prediction, name="my_metric")
                        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
                        self.metric_initializer = tf.variables_initializer(var_list=running_vars)

                    # Calculate coverage loss from the attention distributions
                    if FLAGS.coverage:
                        with tf.variable_scope('coverage_loss'):
                            self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
                            tf.summary.scalar('coverage_loss', self._coverage_loss)
                        self._total_loss = self._loss + FLAGS.cov_loss_wt * self._coverage_loss

        if "decode" in FLAGS.mode:
            # We run decode beam search mode one decoder step at a time
            print(final_dists)
            assert len(
                final_dists) == 1  # final_dists is a singleton list containing shape (batch_size, extended_vsize)
            final_dists = final_dists[0]
            topk_probs, self._topk_ids = tf.nn.top_k(final_dists,
                                                     FLAGS.batch_size * 2)  # take the k largest probs. note batch_size=beam_size in decode mode
            self._topk_log_probs = tf.log(topk_probs)

            assert len(ext_attn_dists) == 1
            ext_attn_dists = ext_attn_dists[0]
            self.ext_ids = tf.argmax(ext_attn_dists, -1)

    def _add_train_op(self):
        """Sets self._train_op, the op to run for training."""
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        loss_to_minimize = self._total_loss if FLAGS.coverage else self._loss + self.consistent_loss
        ext_loss_to_minimize = self._loss_ext
        tvars = tf.trainable_variables()
        tf.logging.info('Building train ops for %d weight matrix with %d parameters...', len(tvars),
                        np.sum([np.prod(v.get_shape().as_list()) for v in tvars]))
        tf.summary.scalar('loss/minimize_loss', loss_to_minimize)
        tf.summary.scalar('loss/consistent_loss', self.consistent_loss)
        tf.summary.scalar('loss/perplexity', tf.exp(self._loss))
        tf.summary.scalar('loss/accuracy', self.accuracy_metric)
        tf.summary.scalar('loss/precision_at_k', self.precision_at_k_metric)
        tf.summary.scalar('loss/recall_at_top_k', self.recall_at_top_k_metric)
        tf.logging.info('Calculating gradients...')
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        ext_gradients = tf.gradients(ext_loss_to_minimize, tvars,
                                     aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        if FLAGS.plot_gradients:
            for grad, var in zip(gradients, tvars):
                tf.summary.histogram(var.name + '/gradient', grad)
            for grad, var in zip(ext_gradients, tvars):
                tf.summary.histogram(var.name + '/ext_gradient', grad)

        # Clip the gradients
        tf.logging.info('Clipping gradients...')
        with tf.device("/gpu:%d" % FLAGS.device if FLAGS.device != '' and FLAGS.device >= 0 else "/cpu:0"):
            grads, global_norm = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
            tf.summary.scalar('loss/global_norm', global_norm)

            ext_grads, ext_global_norm = tf.clip_by_global_norm(ext_gradients, FLAGS.max_grad_norm)
            tf.summary.scalar('loss/ext_global_norm', ext_global_norm)

        learning_rate = tf.train.polynomial_decay(FLAGS.lr, self.global_step,
                                                  FLAGS.dataset_size / FLAGS.batch_size * 5,
                                                  FLAGS.lr / 10)
        tf.summary.scalar('loss/learning_rate', learning_rate)
        # Apply adagrad optimizer
        if FLAGS.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=FLAGS.adagrad_init_acc)
            ext_optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=FLAGS.adagrad_init_acc)

        elif FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
            ext_optimizer = tf.train.AdamOptimizer(learning_rate)

        else:
            raise NotImplementedError()

        tf.logging.info('Applying gradients...')
        with tf.device("/gpu:%d" % FLAGS.device if FLAGS.device != '' and FLAGS.device >= 0 else "/cpu:0"):
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
            self._ext_train_op = ext_optimizer.apply_gradients(zip(ext_grads, tvars))

    def build_graph(self):
        """Add the placeholders, model, global step, train_op and summaries to the graph"""
        tf.logging.info('Building graph...')
        t0 = time.time()

        self._add_placeholders()
        self.global_epoch = tf.get_variable('epoch_num', [], initializer=tf.constant_initializer(1, tf.int32),
                                            trainable=False, dtype=tf.int32)
        self.add_epoch_op = tf.assign_add(self.global_epoch, 1)
        with tf.device("/gpu:%d" % FLAGS.device if FLAGS.device != '' and FLAGS.device >= 0 else "/cpu:0"):
            self._add_seq2seq()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._hps.mode == 'train':
            self._add_train_op()
        self._summaries = tf.summary.merge_all()

        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_train_step(self, sess, batch, summary=False):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'ext_train_op': self._ext_train_op,
            'loss': self._loss,
            'ext_loss': self._loss_ext,
            'con_loss': self.consistent_loss,
            'global_step': self.global_step,
            'global_epoch': self.global_epoch,
            'metric_update': [self.accuracy_update, self.precision_at_k_update, self.recall_at_top_k_update],
            'metrics': [self.accuracy_metric, self.precision_at_k_metric, self.recall_at_top_k_metric],
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        if summary:
            to_return['summaries'] = self._summaries

        result = sess.run(to_return, feed_dict)
        return result

    def run_metrics(self, sess, batch):
        to_return = {
            'reset': self.metric_initializer,
        }
        result = sess.run(to_return, self._make_feed_dict(batch))
        return result

    def run_eval_step(self, sess, batch):
        """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_encoder(self, sess, batch):
        """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

        Args:
          sess: Tensorflow session.
          batch: Batch object that is the same example repeated across the batch (for beam search)

        Returns:
          enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
          dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
        """
        feed_dict = self._make_feed_dict(batch, just_enc=True)  # feed the batch into the placeholders
        (enc_states, dec_in_state, global_step, time_key, local_states, global_states, sen_states) = \
            sess.run([self._enc_states, self._dec_in_state, self.global_step, self.time_key, self.event_local_states,
                      self.event_global_states, self.sen_states],
                     feed_dict)  # run the encoder

        # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        return enc_states, dec_in_state, time_key, local_states, global_states, sen_states

    def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, time_key, local_states,
                       global_states, prev_coverage):
        """For beam search decoding. Run the decoder for one step.

        Args:
          sess: Tensorflow session.
          batch: Batch object containing single example repeated across the batch
          latest_tokens: Tokens to be fed as input into the decoder for this timestep
          enc_states: The encoder states.
          dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
          prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

        Returns:
          ids: top 2k ids. shape [beam_size, 2*beam_size]
          probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
          new_states: new states of the decoder. a list length beam_size containing
            LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
          attn_dists: List length beam_size containing lists length attn_length.
          p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
          new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
        """

        beam_size = len(dec_init_states)

        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._hred_con_padding_mask: batch.hred_con_padding_mask,
            self._hred_art_lens: batch.art_lens,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens])),
            self.time_key: time_key,
            self.event_local_states: local_states,
            self.event_global_states: global_states,
        }

        to_return = {
            "ids": self._topk_ids,
            "probs": self._topk_log_probs,
            "states": self._dec_out_state,
            "attn_dists": self.attn_dists
        }
        #
        # if FLAGS.pointer_gen:
        #     feed[self._hred_enc_batch_extend_vocab] = batch.hred_batch_extend_vocab  #
        #     feed[self._hred_max_art_oovs] = batch.hred_max_art_oovs  #
        #     to_return['p_gens'] = self.p_gens
        if FLAGS.pointer_gen:
            feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed[self._max_art_oovs] = batch.max_art_oovs
            to_return['p_gens'] = self.p_gens

        if self._hps.coverage:
            feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            to_return['coverage'] = self.coverage

        results = sess.run(to_return, feed_dict=feed)  # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in
                      range(beam_size)]

        # Convert singleton list containing a tensor to a list of k arrays
        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()

        if FLAGS.pointer_gen:
            # Convert singleton list containing a tensor to a list of k arrays
            assert len(results['p_gens']) == 1
            p_gens = results['p_gens'][0].tolist()
        else:
            p_gens = [None for _ in range(beam_size)]

        # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
        if FLAGS.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in range(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage

    def extract_sentence(self, sess, batch, sen_states, ext_input, dec_state):

        # cells = [np.expand_dims(state.c, axis=0) for state in dec_state]
        # hiddens = [np.expand_dims(state.h, axis=0) for state in dec_state]
        # new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        # new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
        # new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed = {
            self._ext_input: ext_input,
            self._dec_in_state: dec_state,
            self.sen_states: sen_states,
            self._sen_padding_mask: batch.sen_padding_mask,
        }

        to_return = {
            "ext_ids": self.ext_ids,
            "state": self._ext_out_state,
            "sen_attn_dists": self.ext_attn_dists,
        }

        results = sess.run(to_return, feed_dict=feed)

        new_state = tf.contrib.rnn.LSTMStateTuple(results['state'].c, results['state'].h)

        return results['ext_ids'], new_state, results['sen_attn_dists']


def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

    Returns:
      a scalar
    """

    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex)  # overall average


def _coverage_loss(attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.

    Args:
      attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
      padding_mask: shape (batch_size, max_dec_steps).

    Returns:
      coverage_loss: scalar
    """
    coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
    covlosses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss
