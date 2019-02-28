'''
A script to run each segmenter method on the test set. We recommend that
you copy this code to a Jupyter notebook to help with visualization and debugging.
'''
import os
import sys
sys.path.append('../evolving_wilds')

import functools
import keras.backend as K
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

from src import mri_loader, utils
import main

sys.path.append('../neuron')
import neuron.layers as nrn_layers

sys.path.append('../voxelmorph')
import src.losses as vm_losses

# load validation or test dataset
do_final_test = True
ds_key = 'mri-csts2-test'
label_mapping = main.voxelmorph_labels

eval_data_params = main.named_data_params[ds_key]
eval_data_params['load_vols'] = True

eval_ds = mri_loader.MRIDataset(eval_data_params)
_ = eval_ds.load_dataset()

for f in eval_ds.files_labeled_valid:
    print(f)

gpu_ids = [3]
# set gpu id and tf settings
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpu_ids])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

# load trained segmenters
model_files = [
    ### PUT YOUR TRAINED .h5 MODEL FILES HERE ####
]

model_ids = [
    'no-aug',
    'sas-aug',
    'rand-aug',
    'ours-indep',
    'ours-coupled',
    'ours-indep-rand-aug'
    'supervised',
]

for mi, model_file in enumerate(model_files):
    model_id = model_ids[mi]

    if 'experiments/voxelmorph' in model_file:  # SAS
        do_sas = True
        voxelmorph_model = load_model(
            model_file,
            custom_objects={
                'SpatialTransformer': functools.partial(nrn_layers.SpatialTransformer, indexing='xy'),
            },
            compile=False,
        )

        model_name = os.path.splitext(os.path.basename(model_file))[0]
    else:  # trained segmenter network
        do_sas = False
        model_name = os.path.basename(os.path.dirname(os.path.dirname(model_file))) + '_' + \
                     os.path.splitext(os.path.basename(model_file))[0]

        segmenter_model = load_model(model_file,
                                     custom_objects={'np': np}, compile=False)
        segmenter_model.compile(
            loss='categorical_crossentropy',
            metrics=[vm_losses.binary_dice],
            optimizer=Adam(0.0001),
        )

    print(model_name)


    if do_final_test:
        eval_gen = eval_ds.gen_vols_batch(
            ['labeled_test'], batch_size=1, randomize=False, return_ids=True
        )
        n_eval_examples = eval_ds.params['n_test']

    else:
        eval_gen = eval_ds.gen_vols_batch(
            ['labeled_valid'], batch_size=1, randomize=False, return_ids=True
        )

        n_eval_examples = min(eval_ds.params['n_validation'], eval_ds.vols_labeled_valid.shape[0])

    if do_sas:
        # assume single atlas
        source_X = eval_ds.vols_labeled_train[[0]]
        source_Y = eval_ds.segs_labeled_train[[0]]

        eval_cces, eval_dice, eval_accs, eval_ids = utils.eval_seg_sas_from_gen(
            sas_model=voxelmorph_model,
            atlas_vol=source_X, atlas_labels=source_Y,
            eval_gen=eval_gen, label_mapping=label_mapping,
            n_eval_examples=n_eval_examples, batch_size=16)
    else:
        eval_cces, eval_dice, eval_accs, eval_ids = utils.eval_seg_from_gen(
            segmenter_model=segmenter_model,
            eval_gen=eval_gen, label_mapping=label_mapping,
            n_eval_examples=n_eval_examples, batch_size=16)
    print(eval_ids)

    # save results in a .mat file
    results_dir = './segmentation_test_results'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    if do_final_test:
        mode = 'test'
    else:
        mode = 'valid'

    results_file = os.path.join(results_dir, '{}_{}_{}.mat'.format(eval_ds.display_name, mode, model_id))
    print('Saved results to {}'.format(results_file))
    import scipy.io as sio

    sio.savemat(results_file, {
        'cce': eval_cces,
        'dice': eval_dice,
        'acc': eval_accs
    })