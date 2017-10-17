import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

from keras.models import Model
from keras import optimizers
from input_module import InputModule
from episodic_memory_module import EpisodicMemoryModule






class DynamicMemoryNetwork():
    def __init__(self, configs):
        self.gru_units = 0
        pass

    def train_model(x, xq, y, tx, txq, ty):
        checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                                    #monitor='val_loss',
                                                    monitor='train_loss'
                                                    verbose=0,
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode='auto',
                                                    period=1)
        logger = keras.callbacks.CSVLogger(filename, separator=',', append=False)


        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=opt, loss ="categorical_crossentropy", metrics=["accuracy"])

        train_history = model.fit([x, xq], y,
            callbacks = [checkpoint, logger],
            validation_split = 0.1
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.05)


        loss, acc = model.evaluate([tx, txq], ty,
                    batch_size=batch_size)



    def build_inference_graph(inputs, question, lr,decay, ):

        facts, question = InputModule()(inputs, question)
        memory = EpisodicMemoryModule()(facts, question)
        # TODO: What about recurrent activation of memories?
        answer = layers.Dense(units=vocab_size, activation=None)(K.concatenate(memory,question))


        self.model = Model(inputs=[facts, question], outputs=answer)



    def save_model(path):
        model_arch = self.model.to_json()
        self.model.save_model(path)
        pass

    def get_predictions(self, output):
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred

def run_epoch(session, data, num_epoch=0, verbose=2, train=False):

        total_steps = len(data[0]) // config.batch_size
        total_loss = []
        accuracy = 0

        # shuffle data
        p = np.random.permutation(len(data[0]))
        qp, ip, ql, il, im, a, r = data
        qp, ip, ql, il, im, a, r = qp[p], ip[p], ql[p], il[p], im[p], a[p], r[p]

        for step in range(total_steps):
            index = range(step*config.batch_size,(step+1)*config.batch_size)
            feed = {self.question_placeholder: qp[index],
                  self.input_placeholder: ip[index],
                  self.question_len_placeholder: ql[index],
                  self.input_len_placeholder: il[index],
                  self.answer_placeholder: a[index],
                  self.rel_label_placeholder: r[index],
                  self.dropout_placeholder: dp}
            loss, pred, summary, _ = session.run(
              [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch*total_steps + step)

            answers = a[step*config.batch_size:(step+1)*config.batch_size]
            accuracy += np.sum(pred == answers)/float(len(answers))


            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()


        if verbose:
            sys.stdout.write('\r')

return np.mean(total_loss), accuracy/float(total_steps)


# TODO: Define optimizer, epochs, training
# TODO: Define loading, processing data
# TODO: Metrics and logging
# TODO: Saving and deploying
