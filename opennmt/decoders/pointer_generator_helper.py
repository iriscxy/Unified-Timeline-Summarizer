import tensorflow as tf
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _compute_attention
from opennmt.decoders import decoder
from opennmt.decoders.rnn_decoder import build_cell, align_in_time, _build_attention_mechanism

FLAGS = tf.flags.FLAGS


class PointerGeneratorGreedyEmbeddingHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
    def __init__(self, embedding, start_tokens, end_token):
        self.vocab_size = tf.shape(embedding)[-1]
        super(PointerGeneratorGreedyEmbeddingHelper, self).__init__(embedding, start_tokens, end_token)

    def sample(self, time, outputs, state, name=None):
        """sample for PointerGeneratorGreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        sample_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = tf.equal(sample_ids, self._end_token)
        all_finished = tf.reduce_all(finished)

        # since we have OOV words, we need change these words to UNK
        condition = tf.less(sample_ids, self.vocab_size)
        sample_ids = tf.where(condition, sample_ids, tf.ones_like(sample_ids) * 0)

        next_inputs = tf.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)


class PointerGeneratorDecoder(tf.contrib.seq2seq.BasicDecoder):
    """Pointer Generator sampling decoder."""

    def __init__(self, source_extend_tokens, source_oov_words, coverage, cell, helper, initial_state,
                 output_layer=None):
        self.source_oov_words = source_oov_words
        self.source_extend_tokens = source_extend_tokens
        self.coverage = coverage
        super(PointerGeneratorDecoder, self).__init__(cell, helper, initial_state, output_layer)

    @property
    def output_size(self):
        # Return the cell output and the id
        return tf.contrib.seq2seq.BasicDecoderOutput(
            rnn_output=self._rnn_output_size() + self.source_oov_words,
            sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return tf.contrib.seq2seq.BasicDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size() + self.source_oov_words),
            self._helper.sample_ids_dtype)

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.
        Args:
        time: scalar `int32` tensor.
        inputs: A (structure of) input tensors.
        state: A (structure of) state tensors and TensorArrays.
        name: Name scope for any created operations.
        Returns:
        `(outputs, next_state, next_inputs, finished)`.
        """
        with ops.name_scope(name, "PGDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            # the first cell state contains attention, which is context
            attention = cell_state.attention
            att_cell_state = cell_state.cell_state
            alignments = cell_state.alignments

            with tf.variable_scope('calculate_pgen'):
                # tf.logging.info('attention %s', attention)
                # tf.logging.info('inputs %s', inputs)
                # tf.logging.info('att_cell_state %s', att_cell_state)
                p_gen_input = [attention, inputs]
                for lstm_tuple in att_cell_state:
                    # p_gen_input.append(lstm_tuple.c)
                    p_gen_input.append(lstm_tuple.h)
                p_gen = linear(p_gen_input, 1, True)
                p_gen = tf.sigmoid(p_gen)

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            vocab_dist = tf.nn.softmax(cell_outputs) * p_gen

            # z = tf.reduce_sum(alignments,axis=1)
            # z = tf.reduce_sum(tf.cast(tf.less_equal(alignments, 0),tf.int32))
            alignments = alignments * (1 - p_gen)

            # x = tf.reduce_sum(tf.cast(tf.less_equal((1-p_gen), 0),tf.int32))
            # y = tf.reduce_sum(tf.cast(tf.less_equal(alignments[3], 0),tf.int32))

            # this is only for debug
            # alignments2 =  tf.Print(alignments2,[tf.shape(inputs),x,y,alignments[2][9:12]],message="zeros in vocab dist and alignments")

            # since we have OOV words, we need expand the vocab dist
            vocab_size = tf.shape(vocab_dist)[-1]
            extended_vsize = vocab_size + self.source_oov_words
            batch_size = tf.shape(vocab_dist)[0]
            extra_zeros = tf.zeros((batch_size, self.source_oov_words))
            # batch * extend vocab size
            vocab_dists_extended = tf.concat(axis=-1, values=[vocab_dist, extra_zeros])
            # vocab_dists_extended = tf.Print(vocab_dists_extended,[tf.shape(vocab_dists_extended),self.source_oov_words],message='vocab_dists_extended size')

            batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
            attn_len = tf.shape(self.source_extend_tokens)[1]  # number of states we attend over
            batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
            indices = tf.stack((batch_nums, self.source_extend_tokens), axis=2)  # shape (batch_size, enc_t, 2)
            shape = [batch_size, extended_vsize]
            attn_dists_projected = tf.scatter_nd(indices, alignments, shape)  # [batch_size, extended_vsize]

            final_dists = attn_dists_projected + vocab_dists_extended
            # final_dists = tf.Print(final_dists,[tf.reduce_sum(tf.cast(tf.less_equal(final_dists[0],0),tf.int32))],message='final dist')
            # note: sample_ids will contains OOV words
            sample_ids = self._helper.sample(
                time=time, outputs=final_dists, state=cell_state)

            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids)

            outputs = tf.contrib.seq2seq.BasicDecoderOutput(final_dists, sample_ids)
            return outputs, next_state, next_inputs, finished


class PointerGeneratorAttentionWrapper(tf.contrib.seq2seq.AttentionWrapper):
    def __init__(self, cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None,
                 coverage=False):
        super(PointerGeneratorAttentionWrapper, self).__init__(
            cell,
            attention_mechanism,
            attention_layer_size,
            alignment_history,
            cell_input_fn,
            output_attention,
            initial_cell_state,
            name)
        self.coverage = coverage

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.
        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.
        Args:
        batch_size: `0D` integer tensor: the batch size.
        dtype: The internal state data type.
        Returns:
        An `AttentionWrapperState` tuple containing zeroed out tensors and,
        possibly, empty `TensorArray` objects.
        Raises:
        ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                    "When calling zero_state of AttentionWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and the requested batch size.  Are you using "
                    "the BeamSearchDecoder?  If so, make sure your encoder output has "
                    "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                    "the batch_size= argument passed to zero_state is "
                    "batch_size * beam_width.")
            with tf.control_dependencies(
                    self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: tf.identity(s, name="checked_cell_state"),
                    cell_state)
            return tf.contrib.seq2seq.AttentionWrapperState(
                cell_state=cell_state,
                time=tf.zeros([], dtype=tf.int32),
                attention=_zero_state_tensors(self._attention_layer_size, batch_size, dtype),
                alignments=self._item_or_tuple(
                    attention_mechanism.initial_alignments(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms),
                # since we need to read the alignment history several times, so we need set clear_after_read to False
                alignment_history=self._item_or_tuple(
                    tf.TensorArray(dtype=dtype, size=0, clear_after_read=False, dynamic_size=True)
                    if self._alignment_history else ()
                    for _ in self._attention_mechanisms),
                attention_state=self._item_or_tuple(
                    attention_mechanism.initial_state(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms), )

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
            `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
            `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
            alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
            and context through the attention layer (a linear layer with
            `attention_layer_size` outputs).
        Args:
            inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
            A tuple `(attention_or_cell_output, next_state)`, where:
            - `attention_or_cell_output` depending on `output_attention`.
            - `next_state` is an instance of `AttentionWrapperState`
                containing the state calculated at this time step.
        Raises:
            TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, tf.contrib.seq2seq.AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
                cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).  Are you using "
                "the BeamSearchDecoder?  You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with tf.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = tf.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_alignments = state.alignments
            previous_alignment_history = state.alignment_history
        else:
            previous_alignments = [state.alignments]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        all_histories = []

        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            if self.coverage:
                # if we use coverage mode, previous alignments is coverage vector
                # alignment history stack has shape:  decoder time * batch * atten_len 
                # convert it to coverage vector
                previous_alignments[i] = tf.cond(
                    previous_alignment_history[i].size() > 0,
                    lambda: tf.reduce_sum(tf.transpose(previous_alignment_history[i].stack(), [1, 2, 0]), axis=2),
                    lambda: tf.zeros_like(previous_alignments[i]))
            # debug
            # previous_alignments[i] = tf.Print(previous_alignments[i],[previous_alignment_history[i].size(), tf.shape(previous_alignments[i]),previous_alignments[i]],message="atten wrapper:")
            attention, alignments, next_attention_state = _compute_attention(
                attention_mechanism, cell_output, previous_alignments[i],
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_alignments.append(alignments)
            all_histories.append(alignment_history)
            all_attentions.append(attention)
            all_attention_states.append(next_attention_state)

        attention = tf.concat(all_attentions, 1)
        next_state = tf.contrib.seq2seq.AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(all_histories),
            attention_state=self._item_or_tuple(all_attention_states))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state


def _pg_bahdanau_score(processed_query, keys, coverage, coverage_vector):
    """Implements Bahdanau-style (additive) scoring function.
    Args:
        processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
        coverage: Whether to use coverage mode.
        coverage_vector: only used when coverage is true
    Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or tf.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)
    v = tf.get_variable(
        "attention_v", [num_units], dtype=dtype)
    b = tf.get_variable(
        "attention_b", [num_units], dtype=dtype,
        initializer=tf.zeros_initializer())
    if coverage:
        w_c = tf.get_variable(
            "coverage_w", [num_units], dtype=dtype)
        # debug
        # coverage_vector = tf.Print(coverage_vector,[coverage_vector],message="score")
        coverage_vector = tf.expand_dims(coverage_vector, -1)
        return tf.reduce_sum(v * tf.tanh(keys + processed_query + coverage_vector * w_c + b), [2])
    else:
        return tf.reduce_sum(v * tf.tanh(keys + processed_query + b), [2])


class PointerGeneratorBahdanauAttention(tf.contrib.seq2seq.BahdanauAttention):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=float("-inf"),
                 name="PointerGeneratorBahdanauAttention",
                 coverage=False):
        """Construct the Attention mechanism.
        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
            normalize: Python boolean.  Whether to normalize the energy term.
            probability_fn: (optional) A `callable`.  Converts the score to
            probabilities.  The default is @{tf.nn.softmax}. Other options include
            @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
            Its signature should be: `probabilities = probability_fn(score)`.
            score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
            name: Name to use when creating ops.
            coverage: whether use coverage mode
        """
        super(PointerGeneratorBahdanauAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=normalize,
            probability_fn=probability_fn,
            score_mask_value=score_mask_value,
            name=name)
        self.coverage = coverage

    def __call__(self, query, state):
        """Score the query based on the keys and values.
        Args:
            query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
            previous_alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with tf.variable_scope(None, "pointer_generator_bahdanau_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _pg_bahdanau_score(processed_query, self._keys, self.coverage, state)
        # Note: previous_alignments is not used in probability_fn in Bahda attention, so I use it as coverage vector in coverage mode
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state


class PointerRNNDecoder(decoder.Decoder):
    """A basic RNN decoder."""

    def __init__(self,
                 num_layers,
                 num_units,
                 embeddings,
                 bridge=None,
                 cell_class=None,
                 dropout=0.3,
                 residual_connections=False,
                 coverage=False):
        """Initializes the decoder parameters.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      bridge: A :class:`opennmt.layers.bridge.Bridge` to pass the encoder state
        to the decoder.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell. Defaults to a LSTM cell.
      dropout: The probability to drop units in each layer output.
      residual_connections: If ``True``, each layer input will be added to its
        output.
    """
        self.num_layers = num_layers
        self.num_units = num_units
        self.bridge = bridge
        self.cell_class = cell_class
        self.dropout = dropout
        self.residual_connections = residual_connections
        self.embeddings = embeddings
        self.coverage = coverage

    @property
    def output_size(self):
        """Returns the decoder output size."""
        return self.num_units

    def _init_state(self, zero_state, initial_state=None):
        if initial_state is None:
            return zero_state
        elif self.bridge is None:
            raise ValueError("A bridge must be configured when passing encoder state")
        else:
            return self.bridge(initial_state, zero_state)

    def _get_attention(self, state, step=None):  # pylint: disable=unused-argument
        return None

    def _build_cell(self,
                    mode,
                    batch_size,
                    initial_state=None,
                    memory=None,
                    memory_sequence_length=None,
                    dtype=None):
        _ = memory_sequence_length

        if memory is None and dtype is None:
            raise ValueError("dtype argument is required when memory is not set")

        cell = build_cell(
            self.num_layers,
            self.num_units,
            mode,
            dropout=self.dropout,
            residual_connections=self.residual_connections,
            cell_class=self.cell_class)

        initial_state = self._init_state(
            cell.zero_state(batch_size, dtype or memory.dtype), initial_state=initial_state)

        return cell, initial_state

    def decode(self,
               inputs,
               sequence_length,
               enc_batch_extend_vocab,
               max_art_oovs,
               vocab_size=None,
               initial_state=None,
               sampling_probability=None,
               embedding=None,
               output_layer=None,
               mode=tf.estimator.ModeKeys.TRAIN,
               memory=None,
               memory_sequence_length=None,
               return_alignment_history=False):
        _ = memory
        _ = memory_sequence_length

        batch_size = tf.shape(inputs)[0]

        if (sampling_probability is not None
                and (tf.contrib.framework.is_tensor(sampling_probability)
                     or sampling_probability > 0.0)):
            if embedding is None:
                raise ValueError("embedding argument must be set when using scheduled sampling")

            tf.summary.scalar("sampling_probability", sampling_probability)
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs,
                sequence_length,
                embedding,
                sampling_probability)
            fused_projection = False
        else:
            helper = PointerGeneratorGreedyEmbeddingHelper(self.embeddings, tf.ones([batch_size], tf.int32) * 2, 3)
            fused_projection = True  # With TrainingHelper, project all timesteps at once.

        cell, initial_state = self._build_cell(
            mode,
            batch_size,
            initial_state=initial_state,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            dtype=inputs.dtype)

        if output_layer is None:
            output_layer = decoder.build_output_layer(
                self.output_size, vocab_size, dtype=inputs.dtype)

        basic_decoder = PointerGeneratorDecoder(
            enc_batch_extend_vocab,
            max_art_oovs,
            self.coverage,
            cell,
            helper,
            initial_state,
            output_layer=output_layer)

        outputs, state, length = tf.contrib.seq2seq.dynamic_decode(basic_decoder, maximum_iterations=FLAGS.max_dec_steps)

        # if fused_projection and output_layer is not None:
        #     logits = output_layer(outputs.rnn_output)
        # else:
        logits = outputs.rnn_output
        # Make sure outputs have the same time_dim as inputs
        inputs_len = tf.shape(inputs)[1]
        logits = align_in_time(logits, inputs_len)

        if return_alignment_history:
            alignment_history = self._get_attention(state)
            if alignment_history is not None:
                alignment_history = align_in_time(alignment_history, inputs_len)
            return (logits, state, length, alignment_history)
        return (logits, state, length)

    def step_fn(self,
                mode,
                batch_size,
                initial_state=None,
                memory=None,
                memory_sequence_length=None,
                dtype=tf.float32):
        cell, initial_state = self._build_cell(
            mode,
            batch_size,
            initial_state=initial_state,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            dtype=dtype)

        def _fn(step, inputs, state, mode):
            _ = mode
            # This scope is defined by tf.contrib.seq2seq.dynamic_decode during the
            # training.
            with tf.variable_scope("decoder"):
                outputs, state = cell(inputs, state)
                if self.support_alignment_history:
                    return outputs, state, self._get_attention(state, step=step)
                return outputs, state

        return _fn, initial_state


class PointerAttentionalRNNDecoder(PointerRNNDecoder):
    """A RNN decoder with attention.

  It simple overrides the cell construction to add an attention wrapper.
  """

    def __init__(self,
                 num_layers,
                 num_units,
                 embeddings,
                 bridge=None,
                 attention_mechanism_class=PointerGeneratorBahdanauAttention,
                 output_is_attention=True,
                 cell_class=None,
                 dropout=0.3,
                 residual_connections=False):
        """Initializes the decoder parameters.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      bridge: A :class:`opennmt.layers.bridge.Bridge` to pass the encoder state
        to the decoder.
      attention_mechanism_class: A class inheriting from
        ``tf.contrib.seq2seq.AttentionMechanism`` or a callable that takes
        ``(num_units, memory, memory_sequence_length)`` as arguments and returns
        a ``tf.contrib.seq2seq.AttentionMechanism``. Defaults to
        ``tf.contrib.seq2seq.LuongAttention``.
      output_is_attention: If ``True``, the final decoder output (before logits)
        is the output of the attention layer. In all cases, the output of the
        attention layer is passed to the next step.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell.
      dropout: The probability to drop units in each layer output.
      residual_connections: If ``True``, each layer input will be added to its
        output.
    """
        if attention_mechanism_class is None:
            attention_mechanism_class = PointerGeneratorBahdanauAttention
        super(PointerAttentionalRNNDecoder, self).__init__(
            num_layers,
            num_units,
            embeddings,
            bridge=bridge,
            cell_class=cell_class,
            dropout=dropout,
            residual_connections=residual_connections)
        self.attention_mechanism_class = attention_mechanism_class
        self.output_is_attention = output_is_attention

    @property
    def support_alignment_history(self):
        return True

    def _get_attention(self, state, step=None):
        alignment_history = state.alignment_history
        if step is not None:
            return alignment_history.read(step)
        return tf.transpose(alignment_history.stack(), perm=[1, 0, 2])

    def _build_cell(self,
                    mode,
                    batch_size,
                    initial_state=None,
                    memory=None,
                    memory_sequence_length=None,
                    dtype=None):
        attention_mechanism = _build_attention_mechanism(
            self.attention_mechanism_class,
            self.num_units,
            memory,
            memory_sequence_length=memory_sequence_length)

        cell, initial_cell_state = PointerRNNDecoder._build_cell(
            self,
            mode,
            batch_size,
            initial_state=initial_state,
            dtype=memory.dtype)

        cell = PointerGeneratorAttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=self.num_units,
            alignment_history=True,
            output_attention=self.output_is_attention,
            initial_cell_state=initial_cell_state)

        if mode == tf.estimator.ModeKeys.TRAIN and self.dropout > 0.0:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=1.0 - self.dropout)

        initial_state = cell.zero_state(batch_size, memory.dtype)

        return cell, initial_state


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term
