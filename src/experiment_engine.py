import logging
import os
import sys
import time

import cv2
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils as keras_utils

from src import experiment_base

import json

def configure_gpus(gpus):
    # set gpu id and tf settings
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpus])

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    K.set_session(tf.Session(config=config))


# loads a saved experiment using the saved parameters.
# runs all initialization steps so that we can use the models right away
def load_experiment_from_dir(from_dir,
                             exp_class: experiment_base.Experiment,
                             load_n=None,
                             load_epoch=None,
                             log_to_dir=False,  # dont log if we are just loading this exp for evaluation
                             do_load_models=True
                             ):
    with open(os.path.join(from_dir, 'arch_params.json'), 'r') as f:
        fromdir_arch_params = json.load(f)
        fromdir_arch_params['exp_dir'] = from_dir
    with open(os.path.join(from_dir, 'data_params.json'), 'r') as f:
        fromdir_data_params = json.load(f)

    exp = exp_class(
        data_params=fromdir_data_params, arch_params=fromdir_arch_params,
        prompt_delete_existing=False, prompt_update_name=True, # in case the experiment was renamed
        log_to_dir=log_to_dir)

    exp.load_data(load_n=load_n)
    exp.create_models()

    if do_load_models:
        loaded_epoch = exp.load_models(load_epoch)
    else:
        loaded_epoch = None

    return exp, loaded_epoch

def run_experiment(exp, run_args,
                   end_epoch,
                   save_every_n_epochs, test_every_n_epochs,
                   ):
    if run_args.debug:
        if run_args.epoch is not None:
            end_epoch = int(run_args.epoch) + 10
        else:
            end_epoch = 10

        if hasattr(run_args, 'loadn') and run_args.loadn is None:
            run_args.loadn = 1
        elif not hasattr(run_args, 'loadn'):
            run_args.loadn = None


        save_every_n_epochs = 2
        test_every_n_epochs = 2

        exp.set_debug_mode(True)

    if run_args.batch_size is None:
        run_args.batch_size = 8

    if not hasattr(run_args, 'ignore_missing'):
        run_args.ignore_missing = False

    exp_dir, figures_dir, logs_dir, models_dir = exp.get_dirs()

    # log to the newly created experiments dir
    formatter = logging.Formatter(
        '[%(asctime)s] %(message)s', "%Y-%m-%d %H:%M:%S")
    lfh = logging.FileHandler(
        filename=os.path.join(exp_dir, 'training.log'))
    lsh = logging.StreamHandler(sys.stdout)
    lfh.setFormatter(formatter)
    lsh.setFormatter(formatter)
    lfh.setLevel(logging.DEBUG)
    lsh.setLevel(logging.DEBUG)

    file_stdout_logger = logging.getLogger('both')
    file_stdout_logger.setLevel(logging.DEBUG)
    file_stdout_logger.addHandler(lfh)
    file_stdout_logger.addHandler(lsh)

    file_logger = logging.getLogger('file')
    file_logger.setLevel(logging.DEBUG)
    file_logger.addHandler(lfh)

    # load the dataset. load fewer if debugging
    exp.load_data(load_n=run_args.loadn)

    # create models and load existing ones if necessary
    exp.create_models()

    start_epoch = exp.load_models(run_args.epoch,
        stop_on_missing=not run_args.ignore_missing,
        init_layers=run_args.init_weights)

    # compile models for training
    exp.compile_models()

    if run_args.init_from:
        exp.init_model_weights(run_args.init_from)

    exp.create_generators(batch_size=run_args.batch_size)

    tbw = tf.summary.FileWriter(logs_dir)

    train_batch_by_batch(
        exp=exp, batch_size=run_args.batch_size,
        start_epoch=start_epoch, end_epoch=end_epoch,
        save_every_n_epochs=save_every_n_epochs,
        test_every_n_epochs=test_every_n_epochs,
        tbw=tbw, file_stdout_logger=file_stdout_logger, file_logger=file_logger,
        run_args=run_args,
    )

    return exp_dir

def train_batch_by_batch(
        exp,
        batch_size,
        start_epoch, end_epoch, save_every_n_epochs, test_every_n_epochs,
        tbw, file_stdout_logger, file_logger,
        run_args,
):
    max_n_batch_per_epoch = 1000  # limits each epoch to batch_size * 1000 examples. i think this is ok.
    n_batch_per_epoch_train = min(max_n_batch_per_epoch, int(np.ceil(exp.get_n_train() / float(batch_size))))
    print(exp.get_n_train())
    max_printed_examples = 8
    print_every = 100000  # set this to be really high at  first
    print_atleast_every = 100
    print_atmost = max(1, max_printed_examples / batch_size)


    # lets say we want 1 new result image every 1 minute
    print_every_n_seconds = run_args.print_every

    # save a new model every 20 minutes? seems reasonable
    auto_save_every_n_epochs = 100
    auto_test_every_n_epochs = 100
    min_save_every_n_epochs = 10
    save_every_n_seconds = 20 * 60

    start_time = time.time()

    # do this once here to flush any setup information to the file
    exp._reopen_log_file()

    for e in range(start_epoch, end_epoch + 1):
        file_stdout_logger.debug('{} training epoch {}/{}'.format(exp.model_name, e, end_epoch + 1))

        if e < end_epoch:
            exp.update_epoch_count(e)

        pb = keras_utils.Progbar(n_batch_per_epoch_train)
        printed_count = 0
        for bi in range(n_batch_per_epoch_train):
            joint_loss, joint_loss_names = exp.train_on_batch()
            batch_count = e * n_batch_per_epoch_train + bi

            # only log to file on the last batch of training, otherwise we'll have too many messages
            training_logger = None
            if bi == n_batch_per_epoch_train - 1:
                training_logger = file_logger

            log_losses(pb, tbw, training_logger,
                                     joint_loss_names,
                                     joint_loss,
                                     batch_count)

            # time how long it takes to do 5 batches
            if batch_count - start_epoch * n_batch_per_epoch_train == 5:
                s_per_batch = (time.time() - start_time) / 5.

                # make this an odd integer in case our experiment is doing
                # different things on alternating batches, so that we can visualize both
                print_every = int(np.ceil(print_every_n_seconds / s_per_batch / 2.)) * 2 + 1
                auto_save_every_n_epochs = save_every_n_seconds / s_per_batch / n_batch_per_epoch_train
                if auto_save_every_n_epochs > 50:  # if interval is big enough, adjust to multiples of 50
                    auto_save_every_n_epochs = max(1, int(np.floor(save_every_n_epochs / 50))) * 50
                else:
                    auto_save_every_n_epochs = max(1, int(np.floor(save_every_n_epochs / min_save_every_n_epochs))) \
                                               * min_save_every_n_epochs


            if ((batch_count % print_every == 0 or batch_count % print_atleast_every == 0)) \
                    and printed_count < print_atmost:
                results_im = exp.make_train_results_im()
                cv2.imwrite(
                    os.path.join(exp.figures_dir,
                                 'train_epoch{}_batch{}.jpg'.format(e, bi)
                                 ),
                    results_im)
                printed_count += 1

        if batch_count >= 10:
            file_stdout_logger.debug('Printing every {} batches, '
                                     'saving every {} and {} epochs, '
                                     'testing every {}'.format(print_every,
                                                               auto_save_every_n_epochs,
                                                               save_every_n_epochs,
                                                               test_every_n_epochs,
                                                               ))

        if (e > 0 and e % auto_save_every_n_epochs == 0 and e > start_epoch) or e == end_epoch or (
                            e > 0 and e % save_every_n_epochs == 0 and e > start_epoch):
            exp.save_models(e, iter_count=e * n_batch_per_epoch_train)
            # TODO: figure out how to flush log file without closing
            file_stdout_logger.handlers[0].close()  # flush our .log file
            lfh = logging.FileHandler(filename=os.path.join(exp.exp_dir, 'training.log'))
            file_stdout_logger.handlers[0] = lfh

            if exp.logger is not None:
                exp.logger.handlers[0].close()
                exp._reopen_log_file()

            tbw.close()  # save to disk and then open a new file so that we can read into tensorboard more easily
            tbw.reopen()

        if (e % auto_test_every_n_epochs == 0 or e % test_every_n_epochs == 0):
            file_stdout_logger.debug('{} testing'.format(exp.model_name))
            pbt = keras_utils.Progbar(1)

            test_loss, test_loss_names = exp.test_batches()

            log_losses(pbt, None, file_logger,
                                     test_loss_names, test_loss,
                                     e * n_batch_per_epoch_train + bi)

            results_im = exp.make_test_results_im()
            if results_im is not None:
                cv2.imwrite(os.path.join(exp.figures_dir, 'test_epoch{}_batch{}.jpg'.format(e, bi)), results_im)

            log_losses(None, tbw, file_logger,
                                     test_loss_names, test_loss,
                                     e * n_batch_per_epoch_train + bi)

            print('\n\n')


def log_losses(progressBar, tensorBoardWriter, logger, loss_names, loss_vals, iter_count):
    if not isinstance(loss_vals, list):  # occurs when model only has one loss
        loss_vals = [loss_vals]

    # update the progress bar displayed in stdout
    if progressBar is not None:
        progressBar.add(1, values=[(loss_names[i], loss_vals[i]) for i in range(len(loss_vals))])

    # write to log using python logging
    if logger is not None:
        logger.debug(', '.join(['{}: {}'.format(loss_names[i], loss_vals[i]) for i in range(len(loss_vals))]))

    # write to tensorboard for pretty plots
    if tensorBoardWriter is not None:
        for i in range(len(loss_names)):
            tensorBoardWriter.add_summary(
                tf.Summary(value=[tf.Summary.Value(tag=loss_names[i], simple_value=loss_vals[i]), ]), iter_count)
            if i >= len(loss_vals):
                break
