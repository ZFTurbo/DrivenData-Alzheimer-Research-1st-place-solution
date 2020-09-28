# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


from a00_common_functions import *
import warnings
import shutil
from keras.callbacks import Callback
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score


def validate_files(model, preprocess_input, valid_cubes, valid_answs, shape_size, batch_size, verbose=False):
    start_time = time.time()
    # print('Batch size: {}'.format(batch_size))
    preds = model.predict(valid_cubes, batch_size=batch_size)

    best_score1 = -1
    best_thr1 = -1
    best_score3 = -1
    best_thr3 = -1
    for thr in [x/100 for x in range(2, 98)]:
        preds_binary = preds.copy()
        preds_binary[preds_binary < thr] = 0
        preds_binary[preds_binary >= thr] = 1
        preds_binary = preds_binary.astype(np.uint8)
        score1 = matthews_corrcoef(valid_answs, preds_binary)
        if score1 > best_score1:
            best_score1 = score1
            best_thr1 = thr
        score3 = accuracy_score(valid_answs, preds_binary)
        if score3 > best_score3:
            best_score3 = score3
            best_thr3 = thr
    best_score2 = roc_auc_score(valid_answs, preds)
    print('Score: {:.6f} (THR: {:.2f}) AUC: {:.6f} Acc: {:.6f} (THR: {:.2f}) Time: {:.2f} sec'.format(best_score1, best_thr1, best_score2, best_score3, best_thr3, time.time() - start_time))
    return best_score1, preds


class ModelCheckpoint_Stat(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, filepath_static, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='max', period=1, patience=None, validation_data=()):
        super(ModelCheckpoint_Stat, self).__init__()
        self.interval = period
        self.cubes, self.answs, self.preprocess_input, self.shape_size, self.batch_size = validation_data
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.filepath_static = filepath_static
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.monitor_op = np.greater
        self.best = -np.Inf

        # part for early stopping
        self.epochs_from_best_model = 0
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            self.model.save(self.filepath_static, overwrite=True)
            score, _ = validate_files(self.model, self.preprocess_input, self.cubes, self.answs, self.shape_size, self.batch_size, verbose=False)

            logs['score'] = score
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if score > self.best:
                self.epochs_from_best_model = 0
            else:
                self.epochs_from_best_model += 1

            if self.save_best_only:
                current = score
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                        shutil.copy(filepath, self.filepath_static)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                shutil.copy(filepath, self.filepath_static)

            if self.patience is not None:
                if self.epochs_from_best_model > self.patience:
                    print('Early stopping: {}'.format(self.epochs_from_best_model))
                    self.model.stop_training = True
