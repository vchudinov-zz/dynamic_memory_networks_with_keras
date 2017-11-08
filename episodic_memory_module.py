from keras.layers import Bidirectional, Dense, Concatenate, Multiply, Subtract, TimeDistributed
from keras.engine.topology import Layer, _object_list_uid, _to_list
from keras.layers.recurrent import GRU
from keras import backend as K
import tensorflow as tf
from keras.utils.generic_utils import has_arg
from attention_cells import SoftAttnGRU
from keras.engine import InputSpec


class AttentionGate(Layer):
    """
    A horrible bastardization of TimeDistributed
    """
    # TODO Fix training mode.
    def __init__(self, units):
        super(AttentionGate, self).__init__()
        self.concat = Concatenate()
        self.sub = Subtract()
        self._input_map = {}
        self.units = units
        self.supports_masking = True

        # Attention parameters
        # TODO: Change units to EMB_DIM
        self.l_1 = Dense(units=units,  activation = 'tanh')
        self.l_2 = Dense(units=1, activation='softmax')


    def build(self, input_shape):

        super(AttentionGate, self).build(input_shape)

    def _object_list_uid(self, object_list):
        object_list = _to_list(object_list)
        return ', '.join([str(abs(id(x))) for x in object_list])

    def compute_output_shape(self, input_shape):

        s_0 = list(input_shape[0]) # Input
        s_0[-1] = 1 # Attention

        return tuple(s_0)


    def call(self, inputs, training=None, mask=None):
        facts = inputs[0]
        question = inputs[1]
        memory = inputs[2]

        kwargs = {}
        if has_arg(self.call, 'training'):
            kwargs['training'] = training
        uses_learning_phase = False

        input_shape = K.int_shape(facts)

        if input_shape[0]:
            # batch size matters, use rnn-based implementation
            def step(inputs, states):

                fact = inputs
                global uses_learning_phase
                f_i = [fact * question, fact * memory, K.abs(fact - question), K.abs(fact - memory)]
                z_t = self.concat(f_i)
                g_t = self.l_1(z_t)
                output = self.l_2(g_t)

                if hasattr(self, '_uses_learning_phase'):
                    uses_learning_phase = (self._uses_learning_phase or
                                           uses_learning_phase)
                return output, []

            _, outputs, _ = K.rnn(step, inputs=facts,
                                  initial_states=[None],
                                  input_length=input_shape[1],
                                  unroll=True)
            y = outputs
        else:
            # No batch size specified, therefore the layer will be able
            # to process batches of any size.
            # We can go with reshape-based implementation for performance.
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(facts)[1]
            # Shape: (num_samples * timesteps, ...). And track the
            # transformation in self._input_map.
            input_uid = self._object_list_uid(facts)

            inputs = K.reshape(facts, (-1,) + tuple(input_shape[2:]))
            self._input_map[input_uid] = facts
            # (num_samples * timesteps, ...)

            y = compute_attention(facts,question, memory)

            if hasattr(self, '_uses_learning_phase'):
                uses_learning_phase = self._uses_learning_phase
            # Shape: (num_samples, timesteps, ...)
            output_shape = list(input_shape)
            output_shape[-1] = 1
            y = K.reshape(y, (-1, input_length) + tuple(output_shape[2:]))

        # Apply activity regularizer if any:
        if (hasattr(self, 'activity_regularizer') and self.activity_regularizer is not None):
            regularization_loss = self.layer.activity_regularizer(y)
            self.add_loss(regularization_loss, inputs)

        if uses_learning_phase:
            y._uses_learning_phase = True
        timesteps = input_shape[1]
        new_time_steps = list(y.get_shape())
        new_time_steps[1] = timesteps
        y.set_shape(new_time_steps)
        return y

class EpisodicMemoryModule(Layer):
    def __init__(self, units, memory_type, memory_steps, **kwargs):
        """
        The episodic memory consists of two nested networks + other details.
        The inner network is used to generate an episode, based on the incoming
        facts F_t that are sent from the input module.
        The outer network applies the attention to the facts to update the memory
        with the new episodes.

        """
        # TODO Readd attention type once this whole thing is solved

        self.memories = []
        self.memory_type = memory_type
        self.memory_steps = memory_steps
        self.name = "episodic_memory_module"
        self.concat = Concatenate()

        self.units = units
        #if memory_type == 'GRU':
        #self.memory_net = GRU(units=units, return_sequences=True, stateful=True)
        #elif memory_type == 'RELU':
        self.memory_net = Dense(units=units)

        self.attention_gate = AttentionGate(50)
        self.attention_GRU = SoftAttnGRU(units=units, return_sequences=False)
        super(EpisodicMemoryModule, self).__init__()


    def compute_output_shape(self, input_shape):

        out_shape = list(input_shape[0])
        out_shape[-1] = self.units
        q_shape = list(input_shape[1])
        q_shape[-1] = self.units
        return tuple(q_shape)



    def build(self, input_shape):
        super(EpisodicMemoryModule, self).build(input_shape)

    def call(self, inputs):
        # TODO: Determine if I need transposes.

        facts = inputs[0]
        question = inputs[1]
        # TODO: Add Dropout and BatchNorm.
        # Initialize memory to the question
        memory = K.identity(question)



        for step in range(self.memory_steps):
            # iteratively update memory
            if self.memory_type == 'GRU':
                memory = self.memory_net(self.generate_episode(memory), memory)[0]
            elif self.memory_type == 'RELU':
                attentions = self.attention_gate([facts, question, memory])
                episode = self.attention_GRU([facts, attentions])
                conc = K.concatenate([memory, episode, question], axis=1)
                memory = self.memory_net(conc)

        return memory
