from keras.layers import Bidirectional, Dense, Concatenate, Multiply, Subtract, TimeDistributed
from keras.engine.topology import Layer
from keras.layers.recurrent import GRU
from keras import backend as K
import tensorflow as tf
from attention_cells import SoftAttnGRU



class AttentionGate(Layer):
    # TODO Fix training mode.
    def __init__(self, units):

        self.concat = Concatenate()
        self.sub = Subtract()

        # Attention parameters
        # TODO: Change units to EMB_DIM
        self.l_1 = Dense(units=units,  activation = 'tanh')
        self.l_2 = Dense(units=1, activation='softmax')
        self.built = True

    def compute_output_shape(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[2:]
        #child_output_shape = self.layer.compute_output_shape(child_input_shape)
        timesteps = input_shape[1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]


    def call(self, facts, question, memory, training=None, mask=None):

        kwargs = {}
        #if has_arg(self.layer.call, 'training'):
        self.training = training
        uses_learning_phase = False

        input_shape = K.int_shape(facts)

        def compute_attention(fact, question, memory):


            f_i = [fact * question, fact * memory, K.abs(fact - question), K.abs(fact - memory)]
            z_t = self.concat(f_i)
            g_t = self.l_1(z_t)
            g_t = self.l_2(g_t)

            return g_t


        if input_shape[0]:
            # batch size matters, use rnn-based implementation
            def step(inputs, constants):
                x = inputs
                question = constants[0]
                memory = constants[1]
                global uses_learning_phase
                output = compute_attention(x, question, memory)

                if hasattr(self, '_uses_learning_phase'):
                    uses_learning_phase = (self._uses_learning_phase or
                                           uses_learning_phase)
                return output, []
            _, outputs, _ = K.rnn(step, inputs=facts,
                                  constants=[question, memory],
                                  initial_states=[],
                                  input_length=input_shape[1],
                                  unroll=False)
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
            input_uid = _object_list_uid(facts)
            inputs = K.reshape(facts, (-1,) + input_shape[2:])
            self._input_map[input_uid] = facts
            # (num_samples * timesteps, ...)
            y = compute_attention(facts,question, memory)
            if hasattr(self, '_uses_learning_phase'):
                uses_learning_phase = self._uses_learning_phase
            # Shape: (num_samples, timesteps, ...)
            output_shape = self.compute_output_shape(input_shape)
            y = K.reshape(y, (-1, input_length) + output_shape[2:])

        # Apply activity regularizer if any:
        #if (hasattr(self.layer, 'activity_regularizer') and
        #   self.layer.activity_regularizer is not None):
        #    regularization_loss = self.layer.activity_regularizer(y)
    #        self.add_loss(regularization_loss, inputs)

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
        super(EpisodicMemoryModule, self).__init__()
        self.memories = []
        self.memory_type = memory_type
        self.memory_steps = memory_steps
        self.name = "episodic_memory_module"
        self.concat = Concatenate()

        self.units = units
        if memory_type == 'GRU':
            self.memory_net = GRU(units=units, return_sequences=True, stateful=True)
        elif memory_type == 'RELU':
            self.memory_net = Dense(units=units, activation='relu')

        self.attention_gate = AttentionGate(50)
        self.attention_GRU = SoftAttnGRU(units=units)



    def build(self, input_shape):
        self.attention_gate.build(input_shape[0])
        self.attention_GRU.build(input_shape[0])
        self.memory_net.build(input_shape[0])

        super(EpisodicMemoryModule, self).build(input_shape)



    def generate_episode(self, facts, question, memory):

        attentions = self.attention_gate.call(facts, question, memory)
        episode = self.attention_GRU.call(facts, attentions)
        return episode


    def call(self, inputs):
        # TODO: Determine if I need transposes.

        facts = inputs[0]
        question = inputs[1]
        # TODO: Add Dropout and BatchNorm.
        self.memories = []
        # Initialize memory to the question
        memory = K.identity(question)
        self.memories.append(memory)
        for step in range(self.memory_steps):
            # iteratively update memory
            if self.memory_type == 'GRU':
                memory = self.memory_net.call(self.generate_episode(memory), memory)[0]
            elif self.memory_type == 'RELU':
                episode = self.generate_episode(facts, question, memory)
                conc = self.concat([memory, episode, question])
                memory = self.memory_net.call(conc)
                #self.memories.append(memory) # Not sure if this will work
        return memory
