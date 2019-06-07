import os
import tensorflow as tf
import time
from keras.callbacks import TensorBoard


class TFLogger(TensorBoard):
    def __init__(self, log_dir, batch_size, log_every=1, **kwargs):
        tf.summary.FileWriterCache.clear()
        #self.perc_train = perc_trainlog_dir
        #self.barcode = barcode
        #self.session = tf.InteractiveSession()
        self.log_dir = log_dir
        self.batch_size = batch_size

        super().__init__(log_dir=self.log_dir, batch_size=self.batch_size, **kwargs)
        self.log_every = log_every
        self.counter = 0
        self.sum_loss = 0
        self.sum_acc = 0
        self.sum_pre = 0
        self.epoch_num = 0
        self.counter_for_mean = 1
        self.epoch_end = False
        #self.STEP_SIZE_TRAIN = STEP_SIZE_TRAIN

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.epoch_end:
            self.epoch_end = False
            self.counter_for_mean = 1
            self.sum_loss = 0
            self.sum_acc = 0
            self.sum_pre = 0
        self.sum_loss += logs['loss']
        self.sum_acc += logs['acc']
        self.sum_pre += logs['precision']
        mean_loss = self.sum_loss / self.counter_for_mean
        mean_acc = self.sum_acc / self.counter_for_mean
        self.counter_for_mean += 1

        if self.counter % self.log_every == 0:
            logs['mean_loss'] = mean_loss
            logs['mean_acc'] = mean_acc
            logs['train_on_batch_loss'] = logs.pop('loss')
            logs['train_on_batch_acc'] = logs.pop('acc')
            logs['train_on_batch_pre'] = logs.pop('precision')
            logs['train_on_batch_recall'] = logs.pop('recall')
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()
        super().on_batch_end(batch, logs)
        if self.STEP_SIZE_TRAIN - self.counter_for_mean == 1:
            self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print('Time taken for validation:', time.time()-self.start)

        self.epoch_num += 1
        self.epoch_end = True
        for name, value in logs.items():
            if (name in ['batch', 'size']) or ('val' not in name):
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

