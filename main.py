import argparse
import json
import os
import sys

import numpy as np

# external project dependencies
sys.path.append(os.path.join(os.path.dirname(__file__), 'ext', 'neuron'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ext', 'pytools-lib'))

from src import experiment_engine, transform_models, segmenter_model

# the labels used in the voxelmorph paper (https://github.com/voxelmorph/voxelmorph)
voxelmorph_labels = [0,
                     16,  # brain stem
                     10, 49,  # thalamus (second entry)
                     8, 47,  # cerebellum cortex
                     4, 43,  # ventricles
                     7, 46,  # cerebellum wm
                     12, 51,  # putamen
                     2, 41,  # cerebral wm
                     28, 60,  # ventral dc,
                     11, 50,  # caudate,
                     13, 52,  # pallidum,
                     17, 53,  # hippocampus
                     14, 15,  # 3rd 4th vent
                     18, 54,  # amygdala
                     24,  # csf
                     3, 42,  # cerebral cortex
                     31, 63,  # choroid plexus
                     ]


named_data_params = {
    'mri-supervised': {  # supervised experiment
        'use_labels': voxelmorph_labels,
        'use_atlas_as_source': False,
        'use_subjects_as_source': [],
        'do_load_test': False,
        'img_shape': (160, 192, 224, 1),
        'n_shot': 100,  # in addition to source subjects above
        'n_unlabeled': 0,
        'n_validation': 50,
        'do_preload_vols': True,
        'aug_in_gen': True,
        'n_tm_aug': None,
        'n_flow_aug': None,
        'warp_labels': True,
    },
    'mri-100unlabeled': {
        'use_labels': voxelmorph_labels,
        'use_atlas_as_source': False,
        'use_subjects_as_source': ['atlas'], #['OASIS_OAS1_0327_MR1_mri_talairach_orig'] was used in the paper
        'do_load_test': False,
        'img_shape': (160, 192, 224, 1),
        'n_shot': 0,
        'n_unlabeled': 100,
        'n_validation': 50,
        'do_preload_vols': True,
        'aug_in_gen': True,
        'n_tm_aug': None,
        'n_flow_aug': None,
        'warp_labels': True,
    },
    'mri-100unlabeled-test': {
        'use_labels': voxelmorph_labels,
        'do_load_test': True,
        'n_shot': 0,
        'n_unlabeled': 1,
        'n_validation': 1,
        'n_test': 200,
        'test_seed': 17,
        'use_atlas_as_source': False,
        'use_subjects_as_source': ['atlas'],
        'img_shape': (160, 192, 224, 1),
        'do_preload_vols': True,
        'aug_in_gen': True,
        'n_vte_aug': None,
        'n_flow_aug': None,
        'warp_labels': True,
    },
}


if __name__ == '__main__':
    np.random.seed(17)

    ap = argparse.ArgumentParser()
    # common params
    ap.add_argument('exp_type', nargs='*', type=str, help='trans (transform model), fss (few-shot segmentation)')
    ap.add_argument('-g', '--gpu', nargs='*', type=int, help='gpu id(s) to use', default=1)
    ap.add_argument('-b', '--batch_size', nargs='?', type=int, default=16)
    ap.add_argument('-d', '--data', nargs='?', type=str, help='name of dataset', default=None)

    ap.add_argument('-m', '--model', type=str, help='model architecture', default=None)
    ap.add_argument('--epoch', nargs='?', help='epoch number or "latest"', default=None)


    ap.add_argument('--lr', nargs='?', type=float, help='Learning rate', default=1e-4)
    ap.add_argument('--debug', action='store_true', help='Flag for debug mode (saves more often, only runs for 10 epochs)',
                    default=False)
    ap.add_argument('--loadn', type=int, help='Number of volumes to load (instead of full dataset)', default=None)
    ap.add_argument('--print_every', nargs='?', type=int,
                    help='Number of seconds between printing training batches as images. Useful when debugging', default=120)

    ap.add_argument('--from_dir', nargs='?', default=None, help='Load experiment from dir instead of by params')

    ap.add_argument('--flow_from_dir', nargs='?', default=None, help='Load flow params from dir')
    ap.add_argument('--color_from_dir', nargs='?', default=None, help='Load color params from dir')

    ap.add_argument('--init_from', nargs='*', default=None,
                    help='List of model files to try and initialize weights from. Will attempt to match model names')
    ap.add_argument('--init_weights', action='store_true', default=False,
                    help='Load as many models as we can, and give up on any we cannot find')

    # one-shot segmentation params
    ap.add_argument('--aug_sas', action='store_true', default=False,
                    help='do aug with the flow model in arch_params')
    ap.add_argument('--aug_rand', action='store_true', default=False,
                    help='do aug with random flow fields and rand multiplicative intensity')
    ap.add_argument('--aug_tm', action='store_true', default=False,
                    help='do aug with the transform models in arch_params')

    ap.add_argument('--coupled', action='store_true', default=False,
                    help='coupled sampling of targets for transform models and fss')

    # augmentation params
    ap.add_argument('--aug.flow_amp', nargs='?', type=int, default=None,
                    dest='aug_rand_flow_amp',
                    help='Uniform amplitude of random flow field to start with')
    ap.add_argument('--aug.flow_sigma', nargs='?', type=int, default=None,
                    help='Amount to blur random flow field', dest='aug_rand_blur_sigma')
    ap.add_argument('--aug.n_aug', nargs='?', type=int, default=None,
                    help='Number of new augmented examples to add', dest='data_n_aug')

    args = ap.parse_args()
    experiment_engine.configure_gpus(args.gpu)

    if not args.debug:
        end_epoch = 20000
        save_every_n_epochs = 50
        test_every_n_epochs = 50
    else:
        save_every_n_epochs = 4
        test_every_n_epochs = 2
        end_epoch = 10

    if args.from_dir:
        with open(os.path.join(args.from_dir, 'arch_params.json'), 'r') as f:
            fromdir_arch_params = json.load(f)
        with open(os.path.join(args.from_dir, 'data_params.json'), 'r') as f:
            fromdir_data_params = json.load(f)


    for ei, exp_type in enumerate(args.exp_type):
        if exp_type.lower() == 'trans':
            '''''''''''''''''''''''''''
            Transform (spatial or appearance) trainer. 
            The bidirectional spatial transform model should be trained first,
            since the backwards spatial transform is necessary for learning a 
            color transform model in the atlas' reference frame.
            '''''''''''''''''''''''''''
            test_every_n_epochs = 10
            save_every_n_epochs = 10

            named_arch_params = {
                'flow-fwd': {
                    'model_arch': 'flow_fwd',
                    'save_every': 10,
                    'test_every': 25,
                    'transform_reg_flow': 'grad_l2', 'transform_reg_lambda_flow': 1,
                    'recon_loss_Iw': 'cc_vm',
                    'cc_loss_weight': 1, 'cc_win_size_Iw': 9,
                    'end_epoch': 500,
                },
                'flow-bck': {
                    'model_arch': 'flow_bck',
                    'save_every': 10,
                    'test_every': 25,
                    'transform_reg_flow': 'grad_l2', 'transform_reg_lambda_flow': 1,
                    'recon_loss_Iw': 'cc_vm',
                    'cc_loss_weight': 1, 'cc_win_size_Iw': 9,
                    'end_epoch': 500,
                },
                'flow-bidir': {
                    'model_arch': 'flow_bidir_separate',
                    'save_every' : 10,
                    'test_every': 25,
                    'transform_reg_flow': 'grad_l2', 'transform_reg_lambda_flow': 1,
                    'recon_loss_Iw': 'cc_vm',
                    'cc_loss_weight': 1, 'cc_win_size_Iw': 9,
                    'end_epoch': 500,
                },
                'color-unet': {
                    'model_arch': 'color_unet',
                    'save_every': 5,
                    'test_every': 5,
                    'flow_fwd_model': 'trained_models/spatial_transform_model.h5',
                    'flow_bck_model': 'trained_models/spatial_transform_model_bck.h5',
                    'transform_reg_color': 'grad-seg-l2', 'transform_reg_lambda_color': 1,
                    'color_transform_in_tgt_space': False,
                    'do_include_aux_input': False,
                    'recon_loss_I': 'l2-tgt', # compute reconstruction loss (L2) in target space
                    'recon_loss_wt': 50,
                    'end_epoch': 20,
                    'use_aux_reg': 'contours',
                },
            }

            # since this is MRI data, we can only ever train on one pair at a time
            args.batch_size = 1

            if args.model:
                arch_params = named_arch_params[args.model]
            elif args.from_dir:
                with open(os.path.join(args.from_dir, 'arch_params.json'), 'r') as f:
                    arch_params = json.load(f)
                with open(os.path.join(args.from_dir, 'data_params.json'), 'r') as f:
                    data_params = json.load(f)
            else:
                raise IOError('Must specify a transform model to train!')

            # load flow and color architecture params independently
            if args.flow_from_dir:
                with open(os.path.join(args.flow_from_dir, 'arch_params.json'), 'r') as f:
                    arch_params['flow_arch_params'] = json.load(f)
            if args.color_from_dir:
                with open(os.path.join(args.color_from_dir, 'arch_params.json'), 'r') as f:
                    arch_params['color_arch_params'] = json.load(f)

            # override default dataset
            if args.data:
                data_params = named_data_params[args.data]

            arch_params['lr'] = args.lr

            if 'save_every' in arch_params.keys():
                save_every_n_epochs = arch_params['save_every']
            if 'test_every' in arch_params.keys():
                test_every_n_epochs = arch_params['test_every']


            exp = transform_models.TransformModelTrainer(data_params, arch_params)

            end_epoch = arch_params['end_epoch']
            tm_end_epoch = end_epoch
        elif exp_type.lower() == 'seg':
            '''''''''''''''''''''''''''
            One-shot segmentation (with optional augmentation)
            '''''''''''''''''''''''''''
            named_arch_params = {
                'default': {
                    'nf_enc': [32, 32, 64, 64, 128, 128],
                    'n_convs_per_stage': 2,
                    'n_seg_dims': 2, # segment slices (2D)
                    'n_aug_dims': 3, # augment each volume (3D)
                    'end_epoch': 100000,
                    'pretrain_l2': 500,
                    'warpoh': False,
                    'tm_flow_model': ( # transform model (spatial) for augmentation
                        'experiments/voxelmorph/'
                        'vm2_cc_AtoUMS_100k_CStoUMS_xy_iter50000.h5'
                    ),
                    'tm_flow_bck_model': ( # transform model (spatial) for augmentation
                        'experiments/voxelmorph/'
                        'vm2_cc_AtoUMS_100k_UMStoCS_xy_iter50000.h5'
                    ),
                    'tm_color_model': ( # transform model (appearance) for augmentation
                        'experiments/'
                        'TransformModel_mri-tr-vm-valid-vm_100ul_subj-l-OASIS_OAS1_0327_color_unet_grad-seg-l2_regcwt1_l2-tgt-wt50_1'
                    '/models/color_delta_unet_epoch10_iter1000.h5'),
                },
            }

            if args.from_dir:
                with open(os.path.join(args.from_dir, 'arch_params.json'), 'r') as f:
                    arch_params = json.load(f)
                with open(os.path.join(args.from_dir, 'data_params.json'), 'r') as f:
                    data_params = json.load(f)
            else:
                arch_params = named_arch_params['default']

            if args.model:
                arch_params = named_arch_params[args.model]

            if args.data:
                data_params = named_data_params[args.data]

            arch_params['lr'] = args.lr

            rand_aug_params = {
                'randflow_type': None,
                'flow_sigma': None,
                'flow_amp': 200,
                'blur_sigma': 12,
                'mult_amp': 0.5,
            }

            if args.from_dir:
                with open(os.path.join(args.from_dir, 'arch_params.json'), 'r') as f:
                    arch_params = json.load(f)
                with open(os.path.join(args.from_dir, 'data_params.json'), 'r') as f:
                    data_params = json.load(f)

            data_params['aug_tm'] = False
            data_params['aug_rand'] = False
            data_params['aug_sas'] = False
            data_params['aug_randmult'] = False

            if args.aug_rand:
                data_params['do_preload_vols'] = False
                data_params['aug_params'] = rand_aug_params
                data_params['aug_rand'] = args.aug_rand

                if args.aug_rand_flow_amp is not None:    
                    data_params['aug_params']['flow_amp'] = args.aug_rand_flow_amp
                if args.aug_rand_blur_sigma is not None:
                    data_params['aug_params']['blur_sigma'] = args.aug_rand_blur_sigma

            if args.aug_tm:
                data_params['aug_tm'] = True
                data_params['do_preload_vols'] = False

            elif args.aug_sas:
                data_params['do_preload_vols'] = False
                data_params['aug_sas'] = True
                data_params['n_sas_aug'] = data_params['n_unlabeled']
                data_params['aug_in_gen'] = False

            if args.aug_tm or args.aug_sas or args.aug_rand:
                test_every_n_epochs = 200
            else:
                # test no-aug less often because it will be pretty bad and will plateau quickly
                test_every_n_epochs = 500

            save_every_n_epochs = 50

            if args.coupled:
                arch_params['do_coupled_sampling'] = True
            else:
                arch_params['do_coupled_sampling'] = False

            exp = segmenter_model.SegmenterTrainer(data_params, arch_params, debug=args.debug)

            end_epoch = arch_params['end_epoch']
            tm_end_epoch = end_epoch

        prev_exp_dir = experiment_engine.run_experiment(
            exp=exp, run_args=args,
            end_epoch=end_epoch,
            save_every_n_epochs=save_every_n_epochs,
            test_every_n_epochs=test_every_n_epochs)
        print('Done with experiment {}, models saved to {}'.format(exp_type, prev_exp_dir))
