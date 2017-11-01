from keras import backend as K

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
#from ..egine import Layer
from keras.engine.topology import Layer
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec
import numpy as np

from keras_rnn import RNN

class SoftAtnnGRUCell(Layer):

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):

        super(SoftAtnnGRUCell, self).__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # TOOD Change these to g_t_i
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]

        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                                        self.units:
                                                        self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = NoneSoft
            self.bias_h = None
        self.built = True

        print("I was built")
        raise SystemExit

    def call(self, inputs, states, training=None):
        # Needs question as an input
        h_tm1 = states[0]  # previous state


        x_i = inputs[0]
        # Gate inputs

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = x_i * dp_mask[0]
                inputs_r = x_i * dp_mask[1]
                inputs_h = x_i * dp_mask[2]
            else:
                inputs_z = x_i
                inputs_r = x_i
                inputs_h = x_i
            x_z = K.dot(inputs_z, self.kernel_z)
            x_r = K.dot(inputs_r, self.kernel_r)
            x_h = K.dot(inputs_h, self.kernel_h)
            if self.use_bias:
                x_z = K.bias_add(x_z, self.bias_z)
                x_r = K.bias_add(x_r, self.bias_r)
                x_h = K.bias_add(x_h, self.bias_h)

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1
            z = self.recurrent_act
            r = self.recurrent_activation(x_r + K.dot(h_tm1_r,
                                                      self.recurrent_kernel_r))

            hh = self.activation(x_h + K.dot(r * h_tm1_h,
                                             self.recurrent_kernel_h))
        else:
            if 0. < self.dropout < 1.:
                x_i *= dp_mask[0]
            matrix_x = K.dot(x_i, self.kernel)
            if self.use_bias:
                matrix_x = K.bias_add(matrix_x, self.bias)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            matrix_inner = K.dot(h_tm1,
                                 self.recurrent_kernel[:, :2 * self.units])

            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            x_h = matrix_x[:, 2 * self.units:]
            recurrent_h = K.dot(r * h_tm1,
                                self.recurrent_kernel[:, 2 * self.units:])
            hh = self.activation(x_h + recurrent_h)
        h = z * h_tm1 + (1 - z) * hh

        # Attention modulated output.
        h = attn_gate*h + (1-attn_gate)*h_tm1

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h]

    def _generate_dropout_mask(self, inputs, training=None):
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range(3)]
        else:
            self._dropout_mask = None

    def _generate_recurrent_dropout_mask(self, inputs, training=None):
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._recurrent_dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range(3)]
        else:
            self._recurrent_dropout_mask = None


class SoftAttnGRU(Recurrent):

    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):

        super(SoftAttnGRU, self).__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self.attn_gate = None
        self.states = [None]
        super(SoftAttnGRU, self).__init__(**kwargs)

    def build(self, input_shape):


        self.input_spec = [InputSpec(shape=input_shape)]
        self.states=self.reset_states()
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                                        self.units:
                                                        self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None
        self.built = True

    def step(self, inputs, states):
            # Needs question as an input
            h_tm1 = states[0]  # previous state
            x_i = inputs
            attn_gate = self.attn_gate
            # Gate inputs

            # dropout matrices for input units
            dp_mask = self._dropout_mask
            # dropout matrices for recurrent units
            rec_dp_mask = self._recurrent_dropout_mask


            if self.implementation == 1:
                if 0. < self.dropout < 1.:
                    inputs_z = x_i * dp_mask[0]
                    inputs_r = x_i * dp_mask[1]
                    inputs_h = x_i * dp_mask[2]
                else:
                    inputs_z = x_i
                    inputs_r = x_i
                    inputs_h = x_i
                x_z = K.dot(inputs_z, self.kernel_z)
                x_r = K.dot(inputs_r, self.kernel_r)
                x_h = K.dot(inputs_h, self.kernel_h)
                if self.use_bias:
                    x_z = K.bias_add(x_z, self.bias_z)
                    x_r = K.bias_add(x_r, self.bias_r)
                    x_h = K.bias_add(x_h, self.bias_h)

                if 0. < self.recurrent_dropout < 1.:
                    h_tm1_z = h_tm1 * rec_dp_mask[0]
                    h_tm1_r = h_tm1 * rec_dp_mask[1]
                    h_tm1_h = h_tm1 * rec_dp_mask[2]
                else:
                    h_tm1_z = h_tm1
                    h_tm1_r = h_tm1
                    h_tm1_h = h_tm1
                z = self.recurrent_act
                r = self.recurrent_activation(x_r + K.dot(h_tm1_r,
                                                          self.recurrent_kernel_r))

                hh = self.activation(x_h + K.dot(r * h_tm1_h,
                                                 self.recurrent_kernel_h))
            else:
                if 0. < self.dropout < 1.:
                    x_i *= dp_mask[0]
                matrix_x = K.dot(x_i, self.kernel)
                if self.use_bias:
                    matrix_x = K.bias_add(matrix_x, self.bias)
                if 0. < self.recurrent_dropout < 1.:
                    h_tm1 *= rec_dp_mask[0]
                matrix_inner = K.dot(h_tm1,
                                     self.recurrent_kernel[:, :2 * self.units])

                x_z = matrix_x[:, :self.units]
                x_r = matrix_x[:, self.units: 2 * self.units]
                recurrent_z = matrix_inner[:, :self.units]
                recurrent_r = matrix_inner[:, self.units: 2 * self.units]

                z = self.recurrent_activation(x_z + recurrent_z)
                r = self.recurrent_activation(x_r + recurrent_r)

                x_h = matrix_x[:, 2 * self.units:]
                recurrent_h = K.dot(r * h_tm1,
                                    self.recurrent_kernel[:, 2 * self.units:])
                hh = self.activation(x_h + recurrent_h)
            h = z * h_tm1 + (1 - z) * hh

            # Attention modulated output.
            h = attn_gate*h + (1-attn_gate)*h_tm1

            if 0 < self.dropout + self.recurrent_dropout:
                if training is None:
                    h._uses_learning_phase = True
            return h, [h]
    def __call__(self, inputs, attn_gate, mask=None, training=None, initial_state=None):
        self._generate_dropout_mask(inputs, training=training)#
        self._generate_recurrent_dropout_mask(inputs, training=training)
        self.attn_gate = attn_gate
        return super(SoftAttnGRU, self).__call__(inputs,
                                     mask=mask,
                                     training=training,
                                     initial_state=initial_state)

    def call(self, inputs, attn_gate, mask=None, training=None, initial_state=None):
        self._generate_dropout_mask(inputs, training=training)
        self._generate_recurrent_dropout_mask(inputs, training=training)
        self.attn_gate = attn_gate
        return super(SoftAttnGRU, self).call(inputs,
                                     mask=mask,
                                     training=training,
                                     initial_state=initial_state)

    def _generate_dropout_mask(self, inputs, training=None):
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range(3)]
        else:
            self._dropout_mask = None

    def _generate_recurrent_dropout_mask(self, inputs, training=None):
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._recurrent_dropout_mask = [K.in_train_phase(
                dropped_inputs,
                ones,
                training=training)
                for _ in range(3)]
        else:
            self._recurrent_dropout_mask = None

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        batch_size = self.input_spec[0].shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')
        # initialize state if None
        if self.states[0] is None:
            self.states = [K.zeros((batch_size, self.units))
                           for _ in self.states]
        elif states is None:
            for state in self.states:
                K.set_value(state, np.zeros((batch_size, self.units)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if value.shape != (batch_size, self.units):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, self.units)) +
                                     ', found shape=' + str(value.shape))
                K.set_value(state, value)
