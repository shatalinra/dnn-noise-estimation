import tensorflow as tf
import logging

class ProgressLogging(tf.keras.callbacks.Callback):
    '''Custom callback for reporting training progress'''
    def __init__(self, evaluation_procedure, input, labels, test_rate):
        '''
        display: Number of epochs to wait before outputting loss
        '''
        self.input = input
        self.labels = labels
        self.test_rate = test_rate
        self.last_loss = None
        self.evaluation_procedure = evaluation_procedure

    def on_epoch_end(self, epoch, logs = None):
        if epoch % self.test_rate != 0: return

        loss, grad = self.evaluation_procedure(self.model, self.input, self.labels)

        delta = loss
        if self.last_loss:
            delta = self.last_loss - loss
        self.last_loss = loss

        logging.info('Epoch %d, loss %0.6f, change %0.6f, grad norm %0.6f, lr %0.6f', epoch, loss, delta, grad, self.model.optimizer.lr)
