
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.ops import array_ops
from keras import regularizers
from keras import activations
from keras import initializers
from keras import constraints

class EpisodicMemoryModule(Layer):

    def __init__(self,
                 units,
                 emb_dim,
                 batch_size,
                 memory_steps=3,
                 dropout=0.0,
                 reuglarization=None,
                 **kwargs):
        """Initializes the Episodic Memory Module from
         https://arxiv.org/pdf/1506.07285.pdf and https://arxiv.org/pdf/1603.01417.pdf.

        The module has internally 2 dense layers used to compute attention,
        one attention GRU unit, that modifies the layer input based on the computed attention,
        and finally, one Dense layer that generates the new memory.
        Have a look at the call method to get an idea of how everything works.

        Parameters
        ----------
        units : (int)
            The number of hidden units in the attention and memory networks
        memory_steps : (int)
            Number of steps to iterate over the input and generate the memory.
        emb_dim : (int)
            The size of the embeddings, and thus the number of units for the
            attention computation
        batch_size : (int)
            Size of the batch
        dropout : (float)
            The dropout rate for the module
        **kwargs : (arguments)
            Extra arguments
        """

        # TODO: Dropout
        self.dropout=0
        self.memory_steps = memory_steps
        self.dropout = dropout
        self.name = "episodic_memory_module"
        self._input_map = {}
        self.supports_masking = True
        self.units = units
        self.emb_dim = emb_dim
        self.batch_size=batch_size

        self.kernel_initializer = 'glorot_uniform'
        self.recurrent_initializer = 'glorot_uniform'
        self.bias_initializer = 'zeros'
        self.kernel_regularizer = reuglarization
        self.recurrent_regularizer = reuglarization
        self.bias_regularizer = reuglarization
        self.kernel_constraint = None
        self.recurrent_constraint = None
        self.bias_constraint = None
        self.use_bias = True

        # attention net.
        self.activation = activations.get('sigmoid')
        self.memory_activation = activations.get('relu')
        self.recurrent_activation = activations.get('hard_sigmoid')

        # Episode net
        super(EpisodicMemoryModule, self).__init__()

    def get_config(self):
        # TODO: Fix this to allow saving the entire model
        raise NotImplementedError

    def compute_output_shape(self, input_shape):

        q_shape = list(input_shape[1])
        #q_shape[-1] = self.units * 2
        q_shape[-1] = self.units
        # TODO: Fix to get a proper shape

        return tuple(q_shape)

    def build(self, input_shape):

        self.l_1 = self.add_weight(name='l_1',
                                  shape=(input_shape[1][1]*4, self.emb_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)

        self.l_2 = self.add_weight(name='l_2',
                                  shape=(self.emb_dim, 1),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)

        self.memory_net = self.add_weight(name='memory',
                                  shape=(3*self.units, self.units),
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   trainable=True)

        self.recurrent_kernel = self.add_weight(
                                                shape=(self.units, self.units * 2),
                                                name='recurrent_kernel',
                                                initializer=self.recurrent_initializer,
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 2,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

            self.bias_l1 = self.add_weight(shape=(self.emb_dim,),
                                        name='bias_l1',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.bias_l2 = self.add_weight(shape=(1,),
                                        name='bias_l2',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

            self.memory_bias = self.add_weight(shape=(self.units,),
                                           name='memory_bias',
                                           initializer=self.bias_initializer,
                                           regularizer=self.bias_regularizer,
                                           constraint=self.bias_constraint)

        else:
            self.bias = None

        self.kernel_r = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_r = self.recurrent_kernel[:, :self.units]


        self.kernel_h = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_h = self.recurrent_kernel   [:,
                                  self.units:
                                  self.units * 2]

        if self.use_bias:
            self.bias_r = self.bias[:self.units]
            self.bias_h = self.bias[self.units: self.units * 2]
        else:
            self.bias_r = None
            self.bias_h = None

        super(EpisodicMemoryModule, self).build(input_shape)

    def call(self, inputs):
        """Generates a new memory based on thequestion and
        current inputs.

        Parameters
        ----------
        inputs : (list) of (K.Tensor)
            A list of size two, where each element is a tensor. The first one is
            the facts vector, and the second - the question vector.

        Returns
        -------
        K.Tensor
            A memory generated from the question and fact_vectors
        """
        facts = inputs[0]
        question = inputs[1]
        memory = K.identity(question)   # Initialize memory to the question
        fact_list = tf.unstack(facts, axis=1)
        for step in range(self.memory_steps):
            # Adapted from
            # https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow/
            attentions = [tf.squeeze(
                self.compute_attention_gate(fact, question, memory), axis=1)
                for i, fact in enumerate(fact_list)]

            attentions = tf.transpose(tf.stack(attentions))
            attentions = tf.expand_dims(attentions, axis=-1)

            episode, _, _ = K.rnn(self.step,
                                inputs=K.concatenate([facts, attentions], axis=2),
                                constants=[],
                                initial_states=[memory])
            memory = self.memory_activation(K.dot(K.concatenate(
                    [memory, episode, question], axis=1), self.memory_net) + self.memory_bias)
        return memory

    def step(self, inputs, states, ):
        """Computes the output of a single step. Unlike the vanilla GRU, attention is applied to the
        output, as per https://arxiv.org/pdf/1603.01417.pdf
        ----------
        inputs : (K.Tensor)
            A tensor of shape [batch_size, input_size+1]. The last element of each example is the
            attention score.
        states : (K.Tensor)
            Initial (list) of states
        training : (bool)
            Whether the network is in training mode or not.

        Returns
        -------
        (K.Tensor)
            The output for the current step, modified by attention

        """
            # Needs question as an input
        x_i, attn_gate = array_ops.split(inputs,
                                         num_or_size_splits=[self.units, 1], axis=1)
        h_tm1 = states[0]

        inputs_r = x_i
        inputs_h = x_i

        x_r = K.dot(inputs_r, self.kernel_r)
        x_h = K.dot(inputs_h, self.kernel_h)
        if self.use_bias:
            x_r = K.bias_add(x_r, self.bias_r)
            x_h = K.bias_add(x_h, self.bias_h)

        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

        r = self.recurrent_activation(
            x_r + K.dot(h_tm1_r, self.recurrent_kernel_r))

        hh = self.activation(x_h + K.dot(r * h_tm1_h,
                                         self.recurrent_kernel_h))

        h = attn_gate * hh + (1 - attn_gate) * h_tm1

        return h, [h]

    def compute_attention_gate(self, fact, question, memory):
        """Computes an attention score over a single fact vector,
        question and memory
        """
        f_i = [ fact * question,
                fact * memory,
                K.abs(
                    fact - question),
                K.abs(
                    fact - memory),
                ]

        g_t_i = K.tanh(K.dot(K.concatenate(f_i, axis=1), self.l_1) + self.bias_l1)
        g_t_i = K.softmax(K.dot(g_t_i, self.l_2) + self.bias_l2)
        return g_t_i

    def get_config_1(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout}
        base_config = super(EpisodicMemoryModule, self).get_config()


        return dict(list(base_config.items()) + list(config.items()))