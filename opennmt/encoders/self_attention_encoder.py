"""Define the self-attention encoder."""
import tensorflow as tf

from opennmt.layers import transformer
import math
from opennmt.layers.reducer import SumReducer

from opennmt.encoders.encoder import Encoder
from opennmt.layers.position import SinusoidalPositionEncoder
from opennmt.layers import common

FLAGS = tf.app.flags.FLAGS


class SelfAttentionEncoder_with_relation(Encoder):
    """Encoder using self-attention as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(self,
                 num_layers,
                 num_units=512,
                 num_heads=8,
                 ffn_inner_dim=2048,
                 dropout=0.1,
                 attention_dropout=0.1,
                 relu_dropout=0.1,
                 position_encoder=SinusoidalPositionEncoder()):
        """Initializes the parameters of the encoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of hidden units.
          num_heads: The number of heads in the multi-head attention.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          dropout: The probability to drop units from the outputs.
          attention_dropout: The probability to drop units from the attention.
          relu_dropout: The probability to drop units from the ReLU activation in
            the feed forward layer.
          position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
            apply on inputs or ``None``.
        """
        super(SelfAttentionEncoder_with_relation, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_heads = num_heads
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.position_encoder = position_encoder

    def encode(self, inputs, relation_matrix, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
        """

        :param inputs: [batch, enc_len, emb_dim]
        :param sequence_length: [batch]
        :param mode:
        :return: outputs: [batch, len, dim] last layer output
        state: a tuple ([batch, dim]) * num_layers, contains the sum over len of each layer outputs
        sequence_length [batch]
        """
        inputs *= self.num_units ** 0.5
        if self.position_encoder is not None:
            inputs = self.position_encoder(inputs)

        inputs = tf.layers.dropout(
            inputs,
            rate=self.dropout,
            training=mode == tf.estimator.ModeKeys.TRAIN)
        mask = transformer.build_sequence_mask(  # [batch, 1, 1, enc_len]
            sequence_length,
            num_heads=self.num_heads,
            maximum_length=tf.shape(inputs)[1])

        state = ()

        for l in range(self.num_layers):
            with tf.variable_scope("layer_{}".format(l)):
                with tf.variable_scope("multi_head"):
                    context = transformer.multi_head_attention(  # [batch, len, dim]
                        self.num_heads,
                        transformer.norm(inputs),
                        None,
                        mode,
                        relation_matrix,
                        num_units=self.num_units,
                        mask=mask,
                        dropout=self.attention_dropout)
                    context = transformer.drop_and_add(  # [batch, len, dim]
                        inputs,
                        context,
                        mode,
                        dropout=self.dropout)

                with tf.variable_scope("ffn"):
                    transformed = transformer.feed_forward(  # [batch, len, dim]
                        transformer.norm(context),
                        self.ffn_inner_dim,
                        mode,
                        dropout=self.relu_dropout)
                    transformed = transformer.drop_and_add(  # [batch, len, dim]
                        context,
                        transformed,
                        mode,
                        dropout=self.dropout)

                inputs = transformed
                state += (tf.reduce_mean(inputs, axis=1),)

        outputs = transformer.norm(inputs)  # [batch, len, dim]
        return (outputs, state, sequence_length)


class SelfAttentionEncoder_Length(Encoder):
    """Encoder using self-attention as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(self,
                 num_layers,
                 num_units=512,
                 num_heads=8,
                 ffn_inner_dim=2048,
                 dropout=0.1,
                 attention_dropout=0.1,
                 relu_dropout=0.1,
                 position_encoder=SinusoidalPositionEncoder()):
        """Initializes the parameters of the encoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of hidden units.
          num_heads: The number of heads in the multi-head attention.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          dropout: The probability to drop units from the outputs.
          attention_dropout: The probability to drop units from the attention.
          relu_dropout: The probability to drop units from the ReLU activation in
            the feed forward layer.
          position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
            apply on inputs or ``None``.
        """
        super(SelfAttentionEncoder_Length, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_heads = num_heads
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.position_encoder = position_encoder

    def hier_position_encoder(self, inputs):
        timesteps = FLAGS.max_doc_number
        input_dim = inputs.get_shape().as_list()[-1]/2



        positions = tf.range(timesteps) + 1
        # position_encoding = SinusoidalPositionEncoder.encode([positions], input_dim, dtype=inputs.dtype)
        dtype = inputs.dtype
        positions = [positions]

        depth = input_dim
        positions = tf.cast(positions, tf.float32)

        batch_size=tf.shape(positions)[0]
        log_timescale_increment = math.log(10000) / (depth / 2 - 1)
        inv_timescales = tf.exp(tf.range(depth / 2, dtype=tf.float32) * -log_timescale_increment)
        inv_timescales = tf.reshape(tf.tile(inv_timescales, [batch_size]), [batch_size, -1])
        scaled_time = tf.expand_dims(positions, -1) * tf.expand_dims(inv_timescales, 1)
        encoding = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
        position_encoding = tf.cast(encoding, dtype)
        position_encoding = tf.expand_dims(position_encoding, 2)
        batch_size = FLAGS.batch_size
        position_encoding = tf.tile(position_encoding, [batch_size, 1, FLAGS.word_num_in_doc, 1])
        return position_encoding

    def encode(self, inputs, sequence_length=None, hier=False, mode=tf.estimator.ModeKeys.TRAIN):
        """

        :param inputs: [batch, enc_len, emb_dim]
        :param sequence_length: [batch]
        :param mode:
        :return: outputs: [batch, len, dim] last layer output
        state: a tuple ([batch, dim]) * num_layers, contains the sum over len of each layer outputs
        sequence_length [batch]
        """
        inputs *= self.num_units ** 0.5
        if self.position_encoder is not None and hier == False:
            inputs = self.position_encoder(inputs,hier)
        if hier == True:
            local_emb = self.position_encoder(inputs, hier=True)
            local_emb = tf.reshape(local_emb,
                                   [FLAGS.batch_size, FLAGS.max_doc_number, FLAGS.word_num_in_doc, int(FLAGS.hidden_dim/2)])
            inputs = tf.reshape(inputs,
                                [FLAGS.batch_size, FLAGS.max_doc_number, FLAGS.word_num_in_doc, FLAGS.hidden_dim])
            global_emb = self.hier_position_encoder(inputs)
            concat_emb = tf.concat([local_emb, global_emb], -1)
            inputs = SumReducer()([inputs, concat_emb])
            inputs = tf.reshape(inputs,
                                [FLAGS.batch_size , FLAGS.max_doc_number* FLAGS.word_num_in_doc, FLAGS.hidden_dim])

        inputs = tf.layers.dropout(
            inputs,
            rate=self.dropout,
            training=mode == tf.estimator.ModeKeys.TRAIN)
        mask = transformer.build_sequence_mask(  # [batch, 1, 1, enc_len]
            sequence_length,
            num_heads=self.num_heads,
            maximum_length=tf.shape(inputs)[1])

        state = ()

        for l in range(self.num_layers):
            with tf.variable_scope("layer_{}".format(l)):
                with tf.variable_scope("multi_head"):
                    context = transformer.multi_head_attention(  # [batch, len, dim]
                        self.num_heads,
                        transformer.norm(inputs),
                        None,
                        mode,
                        num_units=self.num_units,
                        mask=mask,
                        dropout=self.attention_dropout)
                    context = transformer.drop_and_add(  # [batch, len, dim]
                        inputs,
                        context,
                        mode,
                        dropout=self.dropout)

                with tf.variable_scope("ffn"):
                    transformed = transformer.feed_forward(  # [batch, len, dim]
                        transformer.norm(context),
                        self.ffn_inner_dim,
                        mode,
                        dropout=self.relu_dropout)
                    transformed = transformer.drop_and_add(  # [batch, len, dim]
                        context,
                        transformed,
                        mode,
                        dropout=self.dropout)

                inputs = transformed
                state += (tf.reduce_mean(inputs, axis=1),)

        outputs = transformer.norm(inputs)  # [batch, len, dim]
        return (outputs, state, sequence_length)


class CrossAttentionEncoder(Encoder):
    """Encoder using self-attention as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(self,
                 num_layers,
                 num_units=512,
                 num_heads=8,
                 ffn_inner_dim=2048,
                 dropout=0.1,
                 attention_dropout=0.1,
                 relu_dropout=0.1,
                 position_encoder=SinusoidalPositionEncoder()):
        """Initializes the parameters of the encoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of hidden units.
          num_heads: The number of heads in the multi-head attention.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          dropout: The probability to drop units from the outputs.
          attention_dropout: The probability to drop units from the attention.
          relu_dropout: The probability to drop units from the ReLU activation in
            the feed forward layer.
          position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
            apply on inputs or ``None``.
        """
        super(CrossAttentionEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_heads = num_heads
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.position_encoder = position_encoder

    def encode(self, first_inputs, first_mask=None, second_input=None, mode=tf.estimator.ModeKeys.TRAIN):
        """

        :param first_inputs: [batch, enc_len, emb_dim]
        :param sequence_length: [batch]
        :param first_mask: [batch,enc_len]
        :param mode:
        :return: outputs: [batch, len, dim] last layer output
        state: a tuple ([batch, dim]) * num_layers, contains the sum over len of each layer outputs
        sequence_length [batch]
        """
        first_inputs *= self.num_units ** 0.5
        if self.position_encoder is not None:
            first_inputs = self.position_encoder(first_inputs)

        first_inputs = tf.layers.dropout(
            first_inputs,
            rate=self.dropout,
            training=mode == tf.estimator.ModeKeys.TRAIN)
        first_mask = tf.expand_dims(first_mask, 1)
        first_mask = tf.expand_dims(first_mask, 1)  # (batch,1,1,enc_len)

        state = ()

        for l in range(self.num_layers):
            with tf.variable_scope("layer_{}".format(l)):
                with tf.variable_scope("multi_head"):
                    context = transformer.multi_head_attention(  # [batch, len, dim]
                        self.num_heads,
                        transformer.norm(first_inputs),
                        second_input,
                        mode,
                        num_units=self.num_units,
                        mask=first_mask,
                        dropout=self.attention_dropout)
                    context = transformer.drop_and_add(  # [batch, len, dim]
                        first_inputs,
                        context,
                        mode,
                        dropout=self.dropout)

                with tf.variable_scope("ffn"):
                    transformed = transformer.feed_forward(  # [batch, len, dim]
                        transformer.norm(context),
                        self.ffn_inner_dim,
                        mode,
                        dropout=self.relu_dropout)
                    transformed = transformer.drop_and_add(  # [batch, len, dim]
                        context,
                        transformed,
                        mode,
                        dropout=self.dropout)

                first_inputs = transformed
                state += (tf.reduce_mean(first_inputs, axis=1),)

        outputs = transformer.norm(first_inputs)  # [batch, len, dim]
        return (outputs, state)


class SelfAttentionEncoder_Mask(Encoder):
    """Encoder using self-attention as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(self,
                 num_layers,
                 num_units=512,
                 num_heads=8,
                 ffn_inner_dim=2048,
                 dropout=0.1,
                 attention_dropout=0.1,
                 relu_dropout=0.1,
                 position_encoder=SinusoidalPositionEncoder()):
        """Initializes the parameters of the encoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of hidden units.
          num_heads: The number of heads in the multi-head attention.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          dropout: The probability to drop units from the outputs.
          attention_dropout: The probability to drop units from the attention.
          relu_dropout: The probability to drop units from the ReLU activation in
            the feed forward layer.
          position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
            apply on inputs or ``None``.
        """
        super(SelfAttentionEncoder_Mask, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_heads = num_heads
        self.ffn_inner_dim = ffn_inner_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.position_encoder = position_encoder

    def encode(self, inputs, mask=None, mode=tf.estimator.ModeKeys.TRAIN):
        """

        :param inputs: [batch, enc_len, emb_dim]
        :param sequence_length: [batch]
        :param mode:
        :return: outputs: [batch, len, dim] last layer output
        state: a tuple ([batch, dim]) * num_layers, contains the sum over len of each layer outputs
        sequence_length [batch]
        """
        inputs *= self.num_units ** 0.5
        if self.position_encoder is not None:
            inputs = self.position_encoder(inputs)

        inputs = tf.layers.dropout(
            inputs,
            rate=self.dropout,
            training=mode == tf.estimator.ModeKeys.TRAIN)
        mask = tf.expand_dims(mask, 1)
        mask = tf.expand_dims(mask, 1)

        state = ()

        for l in range(self.num_layers):
            with tf.variable_scope("layer_{}".format(l)):
                with tf.variable_scope("multi_head"):
                    context = transformer.multi_head_attention(  # [batch, len, dim]
                        self.num_heads,
                        transformer.norm(inputs),
                        None,
                        mode,
                        num_units=self.num_units,
                        mask=mask,
                        dropout=self.attention_dropout)
                    context = transformer.drop_and_add(  # [batch, len, dim]
                        inputs,
                        context,
                        mode,
                        dropout=self.dropout)

                with tf.variable_scope("ffn"):
                    transformed = transformer.feed_forward(  # [batch, len, dim]
                        transformer.norm(context),
                        self.ffn_inner_dim,
                        mode,
                        dropout=self.relu_dropout)
                    transformed = transformer.drop_and_add(  # [batch, len, dim]
                        context,
                        transformed,
                        mode,
                        dropout=self.dropout)

                inputs = transformed
                state += (tf.reduce_mean(inputs, axis=1),)

        outputs = transformer.norm(inputs)  # [batch, len, dim]
        return (outputs, state)
