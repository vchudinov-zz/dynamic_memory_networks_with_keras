from keras import backend as K
import tensorflow as tf
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
#from ..egine import Layer
from keras.engine.topology import Layer
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec
import numpy as np
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops

from keras_rnn import RNN


class SoftAttnGRU(Layer):

    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='uniform',
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
        #self.states=self.reset_states()
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
        super(SoftAttnGRU, self).build(input_shape)

    def step(self, inputs, states, attention, training=None):
            # Needs question as an input
            x_i = inputs
            h_tm1 = states[0]


            # Last entry in states should be the attention gate.
            attn_gate = attention

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

                z = self.recurrent_activation(x_z + K.dot(h_tm1_z, self.recurrent_kernel_z))
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

    def call(self, inputs, attn_gate, mask=None, training=None, initial_state=None, return_last_state=True):
        self._generate_dropout_mask(inputs, training=training)
        self._generate_recurrent_dropout_mask(inputs, training=training)
        self.attn_gate = attn_gate


        kwargs = {}
        #if has_arg(self.layer.call, 'training'):
        self.training = training
        uses_learning_phase = False

        input_shape = K.int_shape(inputs)

        if input_shape[0]:
            # batch size matters, use rnn-based implementation

            last_output, outputs, _ = self.rnn(self.step,
                                  inputs=inputs,
                                  constants=[],
                                  attention=attn_gate,
                                  initial_states=self.get_initial_state(inputs),
                                  input_length=input_shape[1],
                                  unroll=False)
            y = outputs
        else:
            # No batch size specified, therefore the layer will be able
            # to process batches of any size.
            # We can go with reshape-based implementation for performance.
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(inputs)[1]
            # Shape: (num_samples * timesteps, ...). And track the
            # transformation in self._input_map.
            input_uid = _object_list_uid(inputs)
            inputs = K.reshape(inputs, (-1,) + input_shape[2:])
            self._input_map[input_uid] = facts
            # (num_samples * timesteps, ...)

            if hasattr(self, '_uses_learning_phasemy heart is in panama'):
                uses_learning_phase = self._uses_learning_phase
            # Shape: (num_samples, timesteps, ...)
            output_shape = self.compute_output_shape(input_shape)
            y = K.reshape(y, (-1, input_length) + output_shape[2:])

        # Apply activity regularizer if any:
        if return_last_state:
            if (hasattr(self, 'activity_regularizer') and
               self.activity_regularizer is not None):
                regularization_loss = self.activity_regularizer(last_output)
                self.add_loss(regularization_loss, inputs)

            return last_output # TODO verify this is what I want.

        if (hasattr(self, 'activity_regularizer') and
           self.activity_regularizer is not None):
            regularization_loss = self.activity_regularizer(y)
            self.add_loss(regularization_loss, inputs)

        if uses_learning_phase:
            y._uses_learning_phase = True


        timesteps = input_shape[1]
        new_time_steps = list(y.get_shape())
        new_time_steps[1] = timesteps
        y.set_shape(new_time_steps)


        return y



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

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim])
                    for dim in self.state_size]
        else:
            return [K.tile(initial_state, [1, self.state_size])]

    def rnn(self, step_function, inputs, initial_states, attention=None,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):

        ndim = len(inputs.get_shape())
        if ndim < 3:
            raise ValueError('Input should be at least 3D.')

        # Transpose to time-major, i.e.
        # from (batch, time, ...) to (time, batch, ...)
        axes = [1, 0] + list(range(2, ndim))
        inputs = tf.transpose(inputs, (axes))
        attention = tf.transpose(attention, (axes))

        if mask is not None:
            if mask.dtype != tf.bool:
                mask = tf.cast(mask, tf.bool)
            if len(mask.get_shape()) == ndim - 1:
                mask = expand_dims(mask)
            mask = tf.transpose(mask, axes)

        if constants is None:
            constants = []

        global uses_learning_phase
        uses_learning_phase = False

        if unroll:
            if not inputs.get_shape()[0]:
                raise ValueError('Unrolling requires a '
                                 'fixed number of timesteps.')
            states = initial_states
            successive_states = []
            successive_outputs = []

            input_list = tf.unstack(inputs)
            attention_list = tf.unstack(attention)
            if go_backwards:
                input_list.reverse()

            if mask is not None:
                mask_list = tf.unstack(mask)
                if go_backwards:
                    mask_list.reverse()

                for inp, mask_t, att in zip(input_list, mask_list, attention_list):
                    output, new_states = step_function(inp, states, att)
                    if getattr(output, '_uses_learning_phase', False):
                        uses_learning_phase = True

                    # tf.where needs its condition tensor
                    # to be the same shape as its two
                    # result tensors, but in our case
                    # the condition (mask) tensor is
                    # (nsamples, 1), and A and B are (nsamples, ndimensions).
                    # So we need to
                    # broadcast the mask to match the shape of A and B.
                    # That's what the tile call does,
                    # it just repeats the mask along its second dimension
                    # n times.
                    tiled_mask_t = tf.tile(mask_t,
                                           tf.stack([1, tf.shape(output)[1]]))

                    if not successive_outputs:
                        prev_output = zeros_like(output)
                    else:
                        prev_output = successive_outputs[-1]

                    output = tf.where(tiled_mask_t, output, prev_output)

                    return_states = []
                    for state, new_state in zip(states, new_states):
                        # (see earlier comment for tile explanation)
                        tiled_mask_t = tf.tile(mask_t,
                                               tf.stack([1, tf.shape(new_state)[1]]))
                        return_states.append(tf.where(tiled_mask_t,
                                                      new_state,
                                                      state))
                    states = return_states
                    successive_outputs.append(output)
                    successive_states.append(states)
                last_output = successive_outputs[-1]
                new_states = successive_states[-1]
                outputs = tf.stack(successive_outputs)
            else:
                for inp, att in zip(input_list, attention_list):
                    output, states = step_function(inp, states + constants, att)
                    if getattr(output, '_uses_learning_phase', False):
                        uses_learning_phase = True
                    successive_outputs.append(output)
                    successive_states.append(states)
                last_output = successive_outputs[-1]
                new_states = successive_states[-1]
                outputs = tf.stack(successive_outputs)

        else:
            if go_backwards:
                inputs = reverse(inputs, 0)

            states = tuple(initial_states)

            time_steps = tf.shape(inputs)[0]
            outputs, _ = step_function(inputs[0], initial_states, attention[0])
            output_ta = tensor_array_ops.TensorArray(
                dtype=outputs.dtype,
                size=time_steps,
                tensor_array_name='output_ta')
            input_ta = tensor_array_ops.TensorArray(
                dtype=inputs.dtype,
                size=time_steps,
                tensor_array_name='input_ta')
            attention_ta = tensor_array_ops.TensorArray(
                dtype=attention.dtype,
                size=time_steps,
                tensor_array_name='attention_ta')

            input_ta = input_ta.unstack(inputs)
            attention_ta = attention_ta.unstack(attention)
            time = tf.constant(0, dtype='int32', name='time')

            if mask is not None:
                if not states:
                    raise ValueError('No initial states provided! '
                                     'When using masking in an RNN, you should '
                                     'provide initial states '
                                     '(and your step function should return '
                                     'as its first state at time `t` '
                                     'the output at time `t-1`).')
                if go_backwards:
                    mask = reverse(mask, 0)

                mask_ta = tensor_array_ops.TensorArray(
                    dtype=tf.bool,
                    size=time_steps,
                    tensor_array_name='mask_ta')
                mask_ta = mask_ta.unstack(mask)

                def _step(time, output_ta_t, attention, *states):
                    """RNN step function.
                    # Arguments
                        time: Current timestep value.
                        output_ta_t: TensorArray.
                        *states: List of states.
                    # Returns
                        Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
                    """
                    current_input = input_ta.read(time)

                    current_attention = attention_ta.read(time)
                    mask_t = mask_ta.read(time)
                    output, new_states = step_function(current_input,
                                                       tuple(states) +
                                                       tuple(constants),
                                                       current_attention)
                    print("GOT HERE")
                    if getattr(output, '_uses_learning_phase', False):
                        global uses_learning_phase
                        uses_learning_phase = True
                    for state, new_state in zip(states, new_states):
                        new_state.set_shape(state.get_shape())
                    tiled_mask_t = tf.tile(mask_t,
                                           tf.stack([1, tf.shape(output)[1]]))
                    output = tf.where(tiled_mask_t, output, states[0])
                    new_states = [tf.where(tiled_mask_t, new_states[i], states[i]) for i in range(len(states))]
                    output_ta_t = output_ta_t.write(time, output)

                    return (time + 1, output_ta_t) + tuple(new_states)
            else:
                def _step(time, output_ta_t, *states):
                    """RNN step function.
                    # Arguments
                        time: Current timestep value.
                        output_ta_t: TensorArray.
                        *states: List of states.
                    # Returns
                        Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
                    """
                    current_input = input_ta.read(time)
                    current_attention = attention_ta.read(time)
                    output, new_states = step_function(current_input,
                                                       tuple(states) +
                                                       tuple(constants),
                                                       current_attention)

                    if getattr(output, '_uses_learning_phase', False):
                        global uses_learning_phase
                        uses_learning_phase = True
                    for state, new_state in zip(states, new_states):
                        new_state.set_shape(state.get_shape())
                    output_ta_t = output_ta_t.write(time, output)
                    #print("GOt Here")

                    return (time + 1, output_ta_t) + tuple(new_states)


            final_outputs = control_flow_ops.while_loop(
                cond=lambda time, *_: time < time_steps,
                body=_step,
                loop_vars=(time, output_ta) + states,
                parallel_iterations=32,
                swap_memory=True)

            last_time = final_outputs[0]
            output_ta = final_outputs[1]
            new_states = final_outputs[2:]

            outputs = output_ta.stack()
            last_output = output_ta.read(last_time - 1)



        axes = [1, 0] + list(range(2, len(outputs.get_shape())))
        outputs = tf.transpose(outputs, axes)
        last_output._uses_learning_phase = uses_learning_phase

        return last_output, outputs, new_states
