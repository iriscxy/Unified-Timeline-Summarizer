"""Base class and functions for dynamic decoders."""

import abc
import six

import tensorflow as tf

from opennmt.utils import beam_search


def logits_to_cum_log_probs(logits, sequence_length):
    """Returns the cumulated log probabilities of sequences.

    Args:
      logits: The sequence of logits of shape :math:`[B, T, ...]`.
      sequence_length: The length of each sequence of shape :math:`[B]`.

    Returns:
      The cumulated log probability of each sequence.
    """
    mask = tf.sequence_mask(
        sequence_length, maxlen=tf.shape(logits)[1], dtype=logits.dtype)
    mask = tf.expand_dims(mask, -1)

    log_probs = tf.nn.log_softmax(logits)
    log_probs = log_probs * mask
    log_probs = tf.reduce_max(log_probs, axis=-1)
    log_probs = tf.reduce_sum(log_probs, axis=1)

    return log_probs


def get_embedding_fn(embedding):
    """Returns the embedding function.

    Args:
      embedding: The embedding tensor or a callable that takes word ids.

    Returns:
      A callable that takes word ids.
    """
    if callable(embedding):
        return embedding
    else:
        return lambda ids: tf.nn.embedding_lookup(embedding, ids)


def build_output_layer(num_units, vocab_size, dtype=None):
    """Builds the output projection layer.

    Args:
      num_units: The layer input depth.
      vocab_size: The layer output depth.
      dtype: The layer dtype.

    Returns:
      A ``tf.layers.Dense`` instance.

    Raises:
      ValueError: if :obj:`vocab_size` is ``None``.
    """
    if vocab_size is None:
        raise ValueError("vocab_size must be set to build the output layer")

    layer = tf.layers.Dense(vocab_size, use_bias=True, dtype=dtype)
    layer.build([None, num_units])
    return layer


def get_sampling_probability(global_step,
                             read_probability=None,
                             schedule_type=None,
                             k=None):
    """Returns the sampling probability as described in
    https://arxiv.org/abs/1506.03099.

    Args:
      global_step: The training step.
      read_probability: The probability to read from the inputs.
      schedule_type: The type of schedule.
      k: The convergence constant.

    Returns:
      The probability to sample from the output ids as a 0D ``tf.Tensor`` or
      ``None`` if scheduled sampling is not configured.

    Raises:
      ValueError: if :obj:`schedule_type` is set but not :obj:`k` or if
       :obj:`schedule_type` is ``linear`` but an initial :obj:`read_probability`
       is not set.
      TypeError: if :obj:`schedule_type` is invalid.
    """
    if read_probability is None and schedule_type is None:
        return None

    if schedule_type is not None and schedule_type != "constant":
        if k is None:
            raise ValueError("scheduled_sampling_k is required when scheduled_sampling_type is set")

        step = tf.cast(global_step, tf.float32)
        k = tf.constant(k, tf.float32)

        if schedule_type == "linear":
            if read_probability is None:
                raise ValueError("Linear schedule requires an initial read probability")
            read_probability = min(read_probability, 1.0)
            read_probability = tf.maximum(read_probability - k * step, 0.0)
        elif schedule_type == "exponential":
            read_probability = tf.pow(k, step)
        elif schedule_type == "inverse_sigmoid":
            read_probability = k / (k + tf.exp(step / k))
        else:
            raise TypeError("Unknown scheduled sampling type: {}".format(schedule_type))

    return 1.0 - read_probability


@six.add_metaclass(abc.ABCMeta)
class Decoder(object):
    """Base class for decoders."""

    def decode(self,
               inputs,
               sequence_length,
               vocab_size=None,
               initial_state=None,
               sampling_probability=None,
               embedding=None,
               output_layer=None,
               mode=tf.estimator.ModeKeys.TRAIN,
               memory=None,
               memory_sequence_length=None,
               return_alignment_history=False):
        """Decodes a full input sequence.

        Usually used for training and evaluation where target sequences are known.

        Args:
          inputs: The input to decode of shape :math:`[B, T, ...]`. [batch, max_dec_steps, emb_dim]
          sequence_length: The length of each input with shape :math:`[B]`. [batch]
          vocab_size: The output vocabulary size. Must be set if :obj:`output_layer`
            is not set.
          initial_state: The initial state as a (possibly nested tuple of...) tensors.
          sampling_probability: The probability of sampling categorically from
            the output ids instead of reading directly from the inputs.
          embedding: The embedding tensor or a callable that takes word ids.
            Must be set when :obj:`sampling_probability` is set.
          output_layer: Optional layer to apply to the output prior sampling.
            Must be set if :obj:`vocab_size` is not set.
          mode: A ``tf.estimator.ModeKeys`` mode.
          memory: (optional) Memory values to query. [batch, len, num_units]
          memory_sequence_length: (optional) Memory values length. [batch]
          return_alignment_history: If ``True``, also returns the alignment
            history from the attention layer (``None`` will be returned if
            unsupported by the decoder).

        Returns:
          A tuple ``(outputs, state, sequence_length)`` or
          ``(outputs, state, sequence_length, alignment_history)``
          if :obj:`return_alignment_history` is ``True``.
        """
        _ = embedding
        if sampling_probability is not None:
            raise ValueError("Scheduled sampling is not supported by this decoder")
        if (not self.support_multi_source
                and memory is not None
                and tf.contrib.framework.nest.is_sequence(memory)):
            raise ValueError("Multiple source encodings are passed to this decoder "
                             "but it does not support multi source context. You should "
                             "instead configure your encoder to merge the different "
                             "encodings.")

        returned_values = self.decode_from_inputs(
            inputs,
            sequence_length,
            initial_state=initial_state,
            mode=mode,
            memory=memory,
            memory_sequence_length=memory_sequence_length)

        if self.support_alignment_history:
            outputs, state, attention = returned_values
        else:
            outputs, state = returned_values
            attention = None

        if output_layer is None:
            output_layer = build_output_layer(self.output_size, vocab_size, dtype=inputs.dtype)
        logits = output_layer(outputs)
        if return_alignment_history:
            return (logits, state, tf.identity(sequence_length), attention)
        return (logits, state, tf.identity(sequence_length))

    @property
    def support_alignment_history(self):
        """Returns ``True`` if this decoder can return the attention as alignment
        history."""
        return False

    @property
    def support_multi_source(self):
        """Returns ``True`` if this decoder supports multiple source context."""
        return False

    def dynamic_decode(self,
                       embedding,
                       start_tokens,
                       end_token,
                       vocab_size=None,
                       initial_state=None,
                       output_layer=None,
                       maximum_iterations=250,
                       minimum_length=0,
                       mode=tf.estimator.ModeKeys.PREDICT,
                       memory=None,
                       memory_sequence_length=None,
                       dtype=None,
                       return_alignment_history=False,
                       sample_from=1,
                       sample_temperature=1):
        """Decodes dynamically from :obj:`start_tokens` with greedy search.

        Usually used for inference.

        Args:
          embedding: The embedding tensor or a callable that takes word ids.
          start_tokens: The start token ids with shape :math:`[B]`.
          end_token: The end token id.
          vocab_size: The output vocabulary size. Must be set if :obj:`output_layer`
            is not set.
          initial_state: The initial state as a (possibly nested tuple of...) tensors.
          output_layer: Optional layer to apply to the output prior sampling.
            Must be set if :obj:`vocab_size` is not set.
          maximum_iterations: The maximum number of decoding iterations.
          minimum_length: The minimum length of decoded sequences (:obj:`end_token`
            excluded).
          mode: A ``tf.estimator.ModeKeys`` mode.
          memory: (optional) Memory values to query.
          memory_sequence_length: (optional) Memory values length.
          dtype: The data type. Required if :obj:`memory` is ``None``.
          return_alignment_history: If ``True``, also returns the alignment
            history from the attention layer (``None`` will be returned if
            unsupported by the decoder).
          sample_from: Sample predictions from the :obj:`sample_from` most likely
            tokens. If 0, sample from the full output distribution.
          sample_temperature: Value dividing logits. In random sampling, a high
            value generates more random samples.

        Returns:
          A tuple ``(predicted_ids, state, sequence_length, log_probs)`` or
          ``(predicted_ids, state, sequence_length, log_probs, alignment_history)``
          if :obj:`return_alignment_history` is ``True``.
        """
        return Decoder.dynamic_decode_and_search(
            self,
            embedding,
            start_tokens,
            end_token,
            vocab_size=vocab_size,
            initial_state=initial_state,
            output_layer=output_layer,
            beam_width=1,
            length_penalty=0.0,
            maximum_iterations=maximum_iterations,
            minimum_length=minimum_length,
            mode=mode,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            dtype=dtype,
            return_alignment_history=return_alignment_history,
            sample_from=sample_from,
            sample_temperature=sample_temperature)

    def dynamic_decode_and_search(self,
                                  embedding,  # [vocab_size, emb_dim]
                                  start_tokens,  # [beam]
                                  end_token,
                                  vocab_size=None,
                                  initial_state=None,
                                  output_layer=None,
                                  beam_width=5,
                                  length_penalty=0.0,
                                  maximum_iterations=250,
                                  minimum_length=0,
                                  mode=tf.estimator.ModeKeys.PREDICT,
                                  memory=None,  # [beam, enc_len, num_units]
                                  memory_sequence_length=None,  # [beam]
                                  dtype=None,
                                  return_alignment_history=False,
                                  sample_from=1,
                                  sample_temperature=1):
        """Decodes dynamically from :obj:`start_tokens` with beam search.

        Usually used for inference.

        Args:
          embedding: The embedding tensor or a callable that takes word ids.
          start_tokens: The start token ids with shape :math:`[B]`.
          end_token: The end token id.
          vocab_size: The output vocabulary size. Must be set if :obj:`output_layer`
            is not set.
          initial_state: The initial state as a (possibly nested tuple of...) tensors.
          output_layer: Optional layer to apply to the output prior sampling.
            Must be set if :obj:`vocab_size` is not set.
          beam_width: The width of the beam.
          length_penalty: The length penalty weight during beam search.
          maximum_iterations: The maximum number of decoding iterations.
          minimum_length: The minimum length of decoded sequences (:obj:`end_token`
            excluded).
          mode: A ``tf.estimator.ModeKeys`` mode.
          memory: (optional) Memory values to query.
          memory_sequence_length: (optional) Memory values length.
          dtype: The data type. Required if :obj:`memory` is ``None``.
          return_alignment_history: If ``True``, also returns the alignment
            history from the attention layer (``None`` will be returned if
            unsupported by the decoder).
          sample_from: Sample predictions from the :obj:`sample_from` most likely
            tokens. If 0, sample from the full output distribution.
          sample_temperature: Value dividing logits. In random sampling, a high
            value generates more random samples.

        Returns:
          A tuple ``(predicted_ids, state, sequence_length, log_probs)`` or
          ``(predicted_ids, state, sequence_length, log_probs, alignment_history)``
          if :obj:`return_alignment_history` is ``True``.
        """
        if sample_from != 1 and beam_width > 1:
            raise ValueError("Sampling decoding is not compatible with beam search, "
                             "set beam_width to 1 instead.")
        batch_size = tf.shape(start_tokens)[0] * beam_width  # beam^2
        if dtype is None:
            if memory is None:
                raise ValueError("dtype argument is required when no memory is set")
            dtype = tf.contrib.framework.nest.flatten(memory)[0].dtype

        if beam_width > 1:
            if initial_state is not None:
                initial_state = tf.contrib.seq2seq.tile_batch(initial_state, multiplier=beam_width)
            if memory is not None:
                memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
            if memory_sequence_length is not None:
                memory_sequence_length = tf.contrib.seq2seq.tile_batch(
                    memory_sequence_length, multiplier=beam_width)

        embedding_fn = get_embedding_fn(embedding)
        step_fn, initial_state = self.step_fn(
            mode,
            batch_size,
            initial_state=initial_state,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            dtype=dtype)
        if output_layer is None:
            if vocab_size is None:
                raise ValueError("vocab_size must be known when the output_layer is not set")
            output_layer = build_output_layer(self.output_size, vocab_size, dtype=dtype)

        state = {"decoder": initial_state}
        if self.support_alignment_history and not tf.contrib.framework.nest.is_sequence(memory):
            state["attention"] = tf.zeros([batch_size, 0, tf.shape(memory)[1]], dtype=dtype)

        def _symbols_to_logits_fn(ids, step,
                                  state):  # ids: Topk sequences decoded so far [batch_size*beam_size, current_decode_step+1]. state is dict contains 'decoer' 'attention'
            if ids.shape.ndims == 2:
                ids = ids[:, -1]
            inputs = embedding_fn(ids)  # [batch_size*beam_size, emb_dim]
            returned_values = step_fn(step, inputs, state["decoder"], mode)
            # outputs:  [batch_size*beam_size, num_units]
            # cache: cache of each layer, in each layer: memory, self_keys, self_values
            # attention  [batch_size*beam_size, enc_len]
            if self.support_alignment_history:
                outputs, state["decoder"], attention = returned_values
                if "attention" in state:
                    state["attention"] = tf.concat([state["attention"], tf.expand_dims(attention, 1)], 1)
            else:
                outputs, state["decoder"] = returned_values
            logits = output_layer(outputs)
            return logits, state

        if beam_width == 1:
            outputs, lengths, log_probs, state = greedy_decode(
                _symbols_to_logits_fn,
                start_tokens,
                end_token,
                decode_length=maximum_iterations,
                state=state,
                return_state=True,
                min_decode_length=minimum_length,
                last_step_as_input=True,
                sample_from=sample_from,
                sample_temperature=sample_temperature)
        else:
            outputs, log_probs, state = beam_search.beam_search(
                _symbols_to_logits_fn,
                start_tokens,
                beam_width,
                maximum_iterations,
                vocab_size,
                length_penalty,
                states=state,
                eos_id=end_token,
                return_states=True,
                tile_states=False,
                min_decode_length=minimum_length)
            lengths = tf.not_equal(outputs, 0)
            lengths = tf.cast(lengths, tf.int32)
            lengths = tf.reduce_sum(lengths, axis=-1) - 1  # Ignore </s>
            outputs = outputs[:, :, 1:]  # Ignore <s>.

        attention = state.get("attention")
        if beam_width == 1:
            # Make shape consistent with beam search.
            outputs = tf.expand_dims(outputs, 1)
            lengths = tf.expand_dims(lengths, 1)
            log_probs = tf.expand_dims(log_probs, 1)
            if attention is not None:
                attention = tf.expand_dims(attention, 1)
        if attention is not None:
            attention = attention[:, :, 1:]  # Ignore attention for <s>.

        if return_alignment_history:
            return (outputs, state["decoder"], lengths, log_probs, attention)
        return (outputs, state["decoder"], lengths, log_probs)

    def decode_from_inputs(self,
                           inputs,
                           sequence_length,
                           initial_state=None,
                           mode=tf.estimator.ModeKeys.TRAIN,
                           memory=None,
                           memory_sequence_length=None):
        """Decodes from full inputs.

        Args:
          inputs: The input to decode of shape :math:`[B, T, ...]`.
          sequence_length: The length of each input with shape :math:`[B]`.
          initial_state: The initial state as a (possibly nested tuple of...) tensors.
          mode: A ``tf.estimator.ModeKeys`` mode.
          memory: (optional) Memory values to query.
          memory_sequence_length: (optional) Memory values length.

        Returns:
          A tuple ``(outputs, state)`` or ``(outputs, state, attention)``
          if ``self.support_alignment_history``.
        """
        raise NotImplementedError()

    def step_fn(self,
                mode,
                batch_size,
                initial_state=None,
                memory=None,
                memory_sequence_length=None,
                dtype=tf.float32):
        """Callable to run decoding steps.

        Args:
          mode: A ``tf.estimator.ModeKeys`` mode.
          batch_size: The batch size.
          initial_state: The initial state to start from as a (possibly nested tuple
            of...) tensors.
          memory: (optional) Memory values to query.
          memory_sequence_length: (optional) Memory values length.
          dtype: The data type.

        Returns:
          A callable with the signature
          ``(step, inputs, state, mode)`` -> ``(outputs, state)`` or
          ``(outputs, state, attention)`` if ``self.support_alignment_history``.
        """
        raise NotImplementedError()


@six.add_metaclass(abc.ABCMeta)
class DecoderV2(tf.keras.layers.Layer):
    """Base class for decoders.

    Note:
      TensorFlow 2.0 version.
    """

    def __init__(self, num_sources=1, **kwargs):
        """Initializes the decoder parameters.

        Args:
          num_sources: The number of source contexts expected by this decoder.
          **kwargs: Additional layer arguments.

        Raises:
          ValueError: if the number of source contexts :obj:`num_sources` is not
            supported by this decoder.
        """
        if num_sources < self.minimum_sources or num_sources > self.maximum_sources:
            raise ValueError("This decoder accepts between %d and %d source contexts, "
                             "but received %d" % (
                                 self.minimum_sources, self.maximum_sources, num_sources))
        super(DecoderV2, self).__init__(**kwargs)
        self.num_sources = num_sources
        self.output_layer = None

    @property
    def minimum_sources(self):
        """The minimum number of source contexts supported by this decoder."""
        return 1

    @property
    def maximum_sources(self):
        """The maximum number of source contexts supported by this decoder."""
        return 1

    def initialize(self, vocab_size=None, output_layer=None):
        """Initializes the decoder configuration.

        Args:
          vocab_size: The target vocabulary size.
          output_layer: The output layer to use.

        Raises:
          ValueError: if both :obj:`vocab_size` and :obj:`output_layer` are not set.
        """
        if output_layer is not None:
            self.output_layer = output_layer
        else:
            if vocab_size is None:
                raise ValueError("One of vocab_size and output_layer must be set")
            self.output_layer = tf.keras.layers.Dense(vocab_size, name="logits")

    def get_initial_state(self, initial_state=None, batch_size=None, dtype=tf.float32):
        """Returns the initial decoder state.

        Args:
          initial_state: An initial state to start from, e.g. the last encoder
            state.
          batch_size: The batch size to use.
          dtype: The dtype of the state.

        Returns:
          A nested structure of tensors representing the decoder state.

        Raises:
          RuntimeError: if the decoder was not initialized.
          ValueError: if one of :obj:`batch_size` or :obj:`dtype` is not set but
            :obj:`initial_state` is not passed.
        """
        self._assert_is_initialized()
        if batch_size is None or dtype is None:
            if initial_state is None:
                raise ValueError("Either initial_state should be set, or batch_size and "
                                 "dtype should be set")
            first_state = tf.nest.flatten(initial_state)[0]
            if batch_size is None:
                batch_size = tf.shape(first_state)[0]
            if dtype is None:
                dtype = first_state.dtype
        return self._get_initial_state(batch_size, dtype, initial_state=initial_state)

    # pylint: disable=arguments-differ
    def call(self,
             inputs,
             length_or_step=None,
             state=None,
             memory=None,
             memory_sequence_length=None,
             training=None):
        """Runs the decoder layer on either a complete sequence (e.g. for training
        or scoring), or a single timestep (e.g. for iterative decoding).

        Args:
          inputs: The inputs to decode, can be a 3D (training) or 2D (iterative
            decoding) tensor.
          length_or_step: For 3D :obj:`inputs`, the length of each sequence. For 2D
            :obj:`inputs`, the current decoding timestep.
          state: The decoder state.
          memory: Memory values to query.
          memory_sequence_length: Memory values length.
          training: Run in training mode.

        Returns:
          A tuple with the logits, the decoder state, and an attention vector.

        Raises:
          RuntimeError: if the decoder was not initialized.
          ValueError: if the :obj:`inputs` rank is different than 2 or 3.
          ValueError: if :obj:`length_or_step` is invalid.
          ValueError: if the number of source contexts (:obj:`memory`) does not
            match the number defined at the decoder initialization.
        """
        self._assert_is_initialized()
        self._assert_memory_is_compatible(memory, memory_sequence_length)
        rank = inputs.shape.ndims
        if rank == 2:
            if length_or_step.shape.ndims != 0:
                raise ValueError("length_or_step should be a scalar with the current timestep")
            outputs, state, attention = self.step(
                inputs,
                length_or_step,
                state=state,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                training=training)
        elif rank == 3:
            if length_or_step.shape.ndims != 1:
                raise ValueError("length_or_step should contain the length of each sequence")
            outputs, state, attention = self.forward(
                inputs,
                sequence_length=length_or_step,
                initial_state=state,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                training=training)
        else:
            raise ValueError("Unsupported input rank %d" % rank)
        logits = self.output_layer(outputs)
        return logits, state, attention

    def forward(self,
                inputs,
                sequence_length=None,
                initial_state=None,
                memory=None,
                memory_sequence_length=None,
                training=None):
        """Runs the decoder on full sequences.

        Args:
          inputs: The 3D decoder input.
          sequence_length: The length of each input sequence.
          initial_state: The initial decoder state.
          memory: Memory values to query.
          memory_sequence_length: Memory values length.
          training: Run in training mode.

        Returns:
          A tuple with the decoder outputs, the decoder state, and the attention
          vector.
        """
        # TODO: implement this using step().
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self,
             inputs,
             timestep,
             state=None,
             memory=None,
             memory_sequence_length=None,
             training=None):
        """Runs one decoding step.

        Args:
          inputs: The 2D decoder input.
          timestep: The current decoding step.
          state: The decoder state.
          memory: Memory values to query.
          memory_sequence_length: Memory values length.
          training: Run in training mode.

        Returns:
          A tuple with the decoder outputs, the decoder state, and the attention
          vector.
        """
        raise NotImplementedError()

    def _assert_is_initialized(self):
        """Raises an expection if the decoder was not initialized."""
        if self.output_layer is None:
            raise RuntimeError("The decoder was not initialized")

    def _assert_memory_is_compatible(self, memory, memory_sequence_length):
        """Raises an expection if the memory layout is not compatible with this decoder."""

        def _num_elements(obj):
            if obj is None:
                return 0
            elif isinstance(obj, (list, tuple)):
                return len(obj)
            else:
                return 1

        num_memory = _num_elements(memory)
        num_length = _num_elements(memory_sequence_length)
        if num_memory != num_length and memory_sequence_length is not None:
            raise ValueError("got %d memory values but %d length vectors" % (num_memory, num_length))
        if num_memory != self.num_sources:
            raise ValueError("expected %d source contexts, but got %d" % (
                self.num_sources, num_memory))


def greedy_decode(symbols_to_logits_fn,
                  initial_ids,
                  end_id,
                  decode_length=None,
                  state=None,
                  return_state=False,
                  min_decode_length=0,
                  last_step_as_input=False,
                  sample_from=1,
                  sample_temperature=1):
    """Greedily decodes from :obj:`initial_ids`.

    Args:
      symbols_to_logits_fn: Interface to the model, to provide logits.
          Shoud take [batch_size, decoded_ids] and return [batch_size, vocab_size].
      initial_ids: Ids to start off the decoding, this will be the first thing
          handed to symbols_to_logits_fn.
      eos_id: ID for end of sentence.
      decode_length: Maximum number of steps to decode for (EOS included).
      states: A dictionnary of (possibly nested) decoding states.
      return_state: If ``True``, also return the updated decoding states.
      min_decode_length: Minimum length of decoded hypotheses (EOS excluded).
      last_step_as_input: If ``True``, only feed the last predicted ids into
        :obj:`symbols_to_logits_fn`.
      sample_from: Sample from the :obj:`sample_from` most likely tokens. If 0,
        sample from the full output distribution.
      sample_temperature: Value dividing logits. In random sampling, a high value
        generates more random samples.

    Returns:
      A tuple with the decoded output, the decoded lengths, the log probabilities,
      and the decoding states (if :obj:`return_state` is ``True``).
    """
    batch_size = tf.shape(initial_ids)[0]
    batch_ids = tf.range(tf.cast(batch_size, initial_ids.dtype))

    def _condition(step, finished, unused_inputs, unused_outputs,
                   unused_lengths, unused_cum_log_probs, unused_state):
        cond = tf.logical_not(tf.reduce_all(finished))
        if decode_length is not None:
            cond = tf.logical_and(cond, step < decode_length)
        return cond

    def _body(step, finished, inputs, outputs, lengths, cum_log_probs, state):
        # Run next step.
        logits, state = symbols_to_logits_fn(inputs, step, state)
        logits = tf.cast(logits, tf.float32)
        if sample_temperature != 1:
            logits /= tf.cast(sample_temperature, logits.dtype)
        log_probs = tf.nn.log_softmax(logits)
        if min_decode_length > 0:
            log_probs = tf.cond(
                step < min_decode_length,
                true_fn=lambda: beam_search.penalize_token(log_probs, end_id),
                false_fn=lambda: log_probs)

        if sample_from == 1:  # Sample best prediction.
            sampled_ids = tf.argmax(log_probs, axis=-1, output_type=inputs.dtype)
        elif sample_from == 0:  # Sample from the full output distribution.
            distribution = tf.distributions.Categorical(probs=tf.exp(log_probs), dtype=inputs.dtype)
            sampled_ids = distribution.sample()
        else:  # Sample from the top K.
            topk_log_probs, topk_ids = tf.nn.top_k(log_probs, k=sample_from)
            topk_ids = tf.cast(topk_ids, inputs.dtype)
            distribution = tf.distributions.Categorical(logits=topk_log_probs, dtype=inputs.dtype)
            topk_sampled_ids = distribution.sample()
            sampled_ids = tf.gather_nd(topk_ids, tf.stack([batch_ids, topk_sampled_ids], axis=-1))

        sampled_log_probs = tf.gather_nd(log_probs, tf.stack([batch_ids, sampled_ids], axis=-1))
        outputs = outputs.write(step, sampled_ids)

        # Don't update finished batches.
        lengths += 1 - tf.cast(finished, lengths.dtype)
        cum_log_probs += sampled_log_probs * (1.0 - tf.cast(finished, sampled_log_probs.dtype))
        finished = tf.logical_or(finished, tf.equal(sampled_ids, end_id))
        if last_step_as_input:
            next_inputs = sampled_ids
        else:
            next_inputs = tf.concat([inputs, tf.expand_dims(sampled_ids, 1)], axis=1)
        return step + 1, finished, next_inputs, outputs, lengths, cum_log_probs, state

    step = tf.constant(0)
    finished = tf.zeros([batch_size], dtype=tf.bool)
    outputs = tf.TensorArray(initial_ids.dtype, size=0, dynamic_size=True)
    lengths = tf.zeros([batch_size], dtype=tf.int32)
    cum_log_probs = tf.zeros([batch_size], dtype=tf.float32)
    if not last_step_as_input:
        initial_ids = tf.expand_dims(initial_ids, 1)

    _, _, _, outputs, lengths, cum_log_probs, state = tf.while_loop(
        _condition,
        _body,
        loop_vars=(step, finished, initial_ids, outputs, lengths, cum_log_probs, state),
        shape_invariants=(
            step.get_shape(),
            finished.get_shape(),
            tf.TensorShape([None] if last_step_as_input else [None, None]),
            tf.TensorShape(None),
            lengths.get_shape(),
            cum_log_probs.get_shape(),
            tf.contrib.framework.nest.map_structure(
                beam_search.get_state_shape_invariants, state)),
        parallel_iterations=1)

    outputs = tf.transpose(outputs.stack())
    if return_state:
        return outputs, lengths, cum_log_probs, state
    return outputs, lengths, cum_log_probs
