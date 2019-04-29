import glob
import os
import sys
import time

import numpy as np

from src import utils

import pynd.segutils as pynd_segutils

voxelmorph_labels = [0,
                     16,  # brain stem
                     10, 49,  # thalamus (second entry)
                     8, 47,  # cerebellum cortex
                     4, 43,  # ventricles
                     7, 46,  # cerebellum wm
                     12, 51,  # putamen
                     2, 41,  # cerebral wm
                     28, 60,  # ventral dc
                     11, 50,  # caudate
                     13, 52,  # pallidum
                     17, 53,  # hippocampi
                     14, 15,  # 3rd 4th vent
                     18, 54,  # amygdala
                     24,  # csf
                     3, 42,  # cerebral cortex
                     31, 63,  # choroid plexus
                     ]


#################################################
# SETUP: fill in these directories according to where your dataset is stored.
# Also, modify _get_vol_files_list() and _load_vol_and_seg() if necessary based on
# your dataset layout and file formats.
#################################################
# root directory for MRI data, with volumes and segmentations
MRI_TRAIN_SCANS_DIR = os.path.abspath('data/train')
MRI_TRAIN_SEGS_DIR = os.path.abspath('data/train')

MRI_VALIDATION_SCANS_DIR = os.path.abspath('data/validation')
MRI_VALIDATION_SEGS_DIR = os.path.abspath('data/validation')

MRI_TEST_SCANS_DIR  = ''
MRI_TEST_SEGS_DIR = ''

MRI_CONTOURS_DIR = '' # we can save contours so that we don't need to compute them every time


def get_dataset_files_list(mode='train', check_for_matching_segs=False):
    '''
    Gets a list of MRI scan files from the specified directories. Used to filter/prune the
    dataset before actually loading the volumes.

    :param mode: string, either train, validate or test.
    :param check_for_matching_segs: bool, only load scans that have corresponding segmentations
    :return:
    '''

    if mode == 'train':
        vols_files_dir = os.path.join(MRI_TRAIN_SCANS_DIR, '*_vol.npz')
        segs_files_dir = os.path.join(MRI_TRAIN_SEGS_DIR, '*_seg.npz')
    elif mode == 'validate':
        vols_files_dir = os.path.join(MRI_VALIDATION_SCANS_DIR, '*_vol.npz')
        segs_files_dir = os.path.join(MRI_VALIDATION_SEGS_DIR, '*_seg.npz')
    elif mode == 'test':
        vols_files_dir = os.path.join(MRI_TEST_SCANS_DIR, '*_vol.npz')
        segs_files_dir = os.path.join(MRI_TEST_SCANS_DIR, '*_seg.npz')
    else:
        raise NotImplementedError('Mode {} is not supported!'.format(mode))

    vols_files = glob.glob(vols_files_dir)

    if check_for_matching_segs:
        seg_files = glob.glob(segs_files_dir)

        vols_files = [f for f in vols_files
                     if np.any(
                [os.path.splitext(os.path.basename(f))[0].replace('_vol', '') in sf for sf in seg_files])]

    vols_files = sorted(vols_files)

    print('Got list of {} files from {}:'.format(len(vols_files), vols_files_dir))
    for vf in vols_files[:4]:
        print(os.path.basename(vf))
    print('...')

    return vols_files


def load_dataset_files(
        vol_files=None,
        load_n=None,
        mode='train',
        load_segs=True,
        load_contours=False,
        do_mask_vols=False,
        use_labels=None,
):

    if vol_files is None:
        vol_files = get_dataset_files_list(mode=mode)

    if load_n is None:
        load_n = len(vol_files)

    vol_size = (160, 192, 224)
    vols = np.zeros((load_n,) + vol_size + (1,), dtype=np.float32)

    Y_segs = None
    Y_contours = None
    if load_segs:
        Y_segs = np.zeros((load_n,) + vol_size, dtype=int)
    if load_contours:
        # we will store the contours as a binary mask with 1 channel
        Y_contours = np.zeros((load_n,) + vol_size + (1,), dtype=int)

    # load the volumes (and segmentations) from disk
    ids = []
    for i in range(load_n):
        if i % 50 == 0:
            print('Loaded {} of {} files'.format(i, load_n))

        data = load_vol_and_seg(
            vol_files[i],
            load_seg=load_segs,
            load_contours=load_contours,
            do_mask_vol=do_mask_vols,
            keep_labels=use_labels,
        )
        if data is None:
            continue

        vols[i], curr_segs, curr_contours = data
        if load_segs:
            Y_segs[i] = curr_segs
        if load_contours:
            Y_contours[i] = curr_contours

        vol_base_name = os.path.basename(vol_files[i]).split('_vol')[0]
        ids.append(vol_base_name)

    return vols, Y_segs, Y_contours, ids


def load_vol_and_seg(vol_file,
                     load_seg=True, do_mask_vol=False,
                     load_contours=False,
                     keep_labels=None,
                     ):
    if 'train' in vol_file:
        segs_dir = MRI_TRAIN_SEGS_DIR
    elif 'valid' in vol_file:
        segs_dir = MRI_VALIDATION_SEGS_DIR
    else:
        segs_dir = MRI_TEST_SEGS_DIR

    # load volume and corresponding segmentation from file
    vol_name = os.path.splitext(os.path.basename(vol_file))[0].split('_vol')[0]

    img_data = np.load(vol_file)['vol_data'][..., np.newaxis]

    if load_seg or do_mask_vol or load_contours:

        seg_vol_file = os.path.join(segs_dir, '{}_seg.npz'.format(vol_name))

        if not os.path.isfile(seg_vol_file):
            print('Could not find corresponding seg file for {}!'.format(vol_file))
            return None

        seg_data = np.load(seg_vol_file)['seg_data']
    else:
        seg_data = None

    if do_mask_vol:
        # background should always be 0 and other segs should be integers
        img_data *= (seg_data > 0).astype(int)[..., np.newaxis]

    if keep_labels is not None:
        unique_seg_labels = np.unique(seg_data)

        for l in unique_seg_labels:
            if l not in keep_labels:
                seg_data[seg_data==l] = 0

        contours_file = os.path.join(MRI_CONTOURS_DIR, '{}_{}labels.npz'.format(vol_name, len(keep_labels)))
    else:
        contours_file = os.path.join(MRI_CONTOURS_DIR, '{}_alllabels.npz'.format(vol_name))

    contours = None
    if load_contours:
        if not os.path.isfile(contours_file):
            contours = pynd_segutils.seg2contour(seg_data, contour_type='both')[..., np.newaxis]
            contours[contours > 0] = 1
            np.savez(contours_file, contours_data=contours)
        else:
            contours = np.load(contours_file)['contours_data']

    return img_data, seg_data, contours



class MRIDataset(object):
    def __init__(self, params, logger=None, profiler_logger=None):
        # default parameter settings that might not be in keys
        if 'do_preload_vols' not in params:
            params['do_preload_vols'] = False

        if 'n_shot' not in params:
            params['n_shot'] = 0

        self.profiler_logger = profiler_logger
        self.params = params
        self.logger = logger

        self.create_display_name()


    def create_display_name(self):
        '''
        Creates an informative name for this dataset, based on user-specified parameters.
        :return:
        '''
        self.display_name = 'mri'

        self.display_name += '_source'
        for source_subject in self.params['use_subjects_as_source']:
            self.display_name += '-{}'.format('_'.join(source_subject.split('_')[:3]))

        self.display_name += '_{}l'.format(self.params['n_shot'])
        self.display_name += '_{}ul'.format(self.params['n_unlabeled'])

        return self.display_name


    def load_dataset(self, load_n=None, load_segs=True,
                     load_source_segs=True,
                     load_source_contours=True):
        '''
        Main entry point for loading dataset
        :return: (X, Y, ids) tuples for unlabeled set, labeled training set and labeled validation set
        '''
        self._print('Loading mri dataset {}'.format(self.display_name))
        self._print('Params: {}'.format(self.params))

        self._get_files_list()
        self._print('Got lists of {} labeled training, {} unlabeled training, {} labeled validation files!'.format(
            len(self.files_labeled_train), len(self.files_unlabeled_train), len(self.files_labeled_valid)))

        # first load all labeled examples
        self.vols_labeled_train, self.segs_labeled_train, self.contours_labeled_train, self.ids_labeled_train \
            = load_dataset_files(
            vol_files=self.files_labeled_train,
            load_segs=load_source_segs,
            load_contours=load_source_contours,
            do_mask_vols=True,
            load_n=load_n,
            use_labels=self.label_mapping,
        )
        self._print('Source (labeled training) examples:')
        self._print('vols_labeled_train: {}'.format(self.vols_labeled_train.shape))

        if load_segs:
            self._print('segs_labeled_train: {}'.format(self.segs_labeled_train.shape))
        self._print('ids_labeled_train: {}'.format(self.ids_labeled_train))

        # set to None by default
        self.segs_labeled_valid = None
        self.segs_unlabeled_train = None
        if not self.params['do_preload_vols']:
            # just copy the source volume into everything so that we can refer to it for the scan shape
            self.vols_labeled_valid = self.vols_labeled_train[[0]]
            self.vols_unlabeled_train = self.vols_labeled_train[[0]]
            self.contours_labeled_valid = self.contours_labeled_train[[0]]
            self.contours_unlabeled_train = self.contours_labeled_train[[0]]

            if load_segs:
                self.segs_labeled_valid = self.segs_labeled_train[[0]]
                self.segs_unlabeled_train = self.segs_labeled_train[[0]]
        else:
            # load the volumes
            self.vols_labeled_valid, self.segs_labeled_valid, self.contours_labeled_valid, self.ids_labeled_valid \
                = load_dataset_files(
                vol_files=self.files_labeled_valid,
                load_n=load_n,
                load_segs=load_segs,
                load_contours=False,
                do_mask_vols=True,
                use_labels=self.label_mapping,
            )

            self.vols_unlabeled_train, self.segs_unlabeled_train, self.contours_unlabeled_train, self.ids_unlabeled_train \
                = load_dataset_files(
                vol_files=self.files_unlabeled_train,
                load_n=load_n,
                load_segs=load_segs,
                do_mask_vols=True,
                use_labels=self.label_mapping
            )

        # load final test set and filter labels
        if 'do_load_test' in self.params.keys() and self.params['do_load_test']:
            self.files_labeled_test = get_dataset_files_list('test')

            np.random.shuffle(self.files_labeled_test)

            self.files_labeled_test = self.files_labeled_test[:self.params['n_test']]
            self.vols_labeled_test, self.segs_labeled_test, self.contours_labeled_test, self.ids_labeled_test \
                = load_dataset_files(
                vol_files=self.files_labeled_test,
                load_n=self.params['n_test'],
                load_segs=True,
                load_contours=False,
                do_mask_vols=True,
                use_labels=self.label_mapping,
            )

        return (self.vols_unlabeled_train, self.segs_unlabeled_train, self.contours_unlabeled_train, self.files_unlabeled_train), \
                (self.vols_labeled_train, self.segs_labeled_train, self.contours_labeled_train, self.files_labeled_train), \
                (self.vols_labeled_valid, self.segs_labeled_valid, self.contours_labeled_valid, self.files_labeled_valid), \
                self.label_mapping


    def _get_files_list(self):
        '''
        Looks in the user-specified directories for MRI scan files. Splits the training set into labeled and unlabeled
        examples, and does sanity checking on training vs validation overlap.
        :return:
        '''
        if 'use_labels' in self.params.keys():
            # filter by user-specified labels
            self.label_mapping = self.params['use_labels']
        else:
            self.label_mapping = None

        # get a list of scan files from specified training and validation directories
        self.files_train = get_dataset_files_list('train', check_for_matching_segs=False)
        self.files_labeled_valid = get_dataset_files_list('validate', check_for_matching_segs=False)

        np.random.seed(17)
        np.random.shuffle(self.files_train)
        np.random.shuffle(self.files_labeled_valid)

        # only take the number of training/validation examples that were specified in params
        if self.params['n_unlabeled'] is None or self.params['n_unlabeled'] == -1 \
                or self.params['n_shot'] is None or self.params['n_shot'] == -1:
            # if any of n_unlabeled or n_shot is -1, that means to use everything
            n_train = len(self.files_train)
        else:
            # only use the number that was requested
            n_train = self.params['n_unlabeled'] + self.params['n_shot']

        self.files_labeled_valid = self.files_labeled_valid[:self.params['n_validation']]

        source_subject_files = []
        # make sure we include the source subject file in the training list
        for source_subject in self.params['use_subjects_as_source']:
            filenames_train = [os.path.basename(f).split('_vol')[0] for f in self.files_train]

            if source_subject in filenames_train:
                duplicate_idxs = [i for i, fn in enumerate(filenames_train) if fn == source_subject]
                source_subject_files += [self.files_train[i] for i in duplicate_idxs]
                # dont include this in the training set yet because we will prune the training set
                self.files_train = [self.files_train[i] for i in range(len(self.files_train)) if i not in duplicate_idxs]

        n_train = min(n_train, len(self.files_train))
        self.files_train = self.files_train[:n_train]

        # now split our training examples into labeled and unlabeled
        self._print('Using specified subjects as atlases: {}'.format(
            self.params['use_subjects_as_source']))
        self.files_labeled_train = source_subject_files[:]

        if self.params['n_shot'] > 0:
            self._print('Including {} additional subjects in labeled training set'.format(
                self.params['n_shot']))

            if self.params['n_unlabeled'] is not None and self.params['n_unlabeled'] > 0:
                # save at least one example for an unlabeled set
                max_n_labeled = n_train - 1
            else:
                max_n_labeled = n_train

            # select a limited number of labeled training examples
            labeled_idxs_train = np.random.choice(
                n_train, min(max_n_labeled, self.params['n_shot']), replace=False).tolist()
        else:
            labeled_idxs_train = []

        self.files_labeled_train += [self.files_train[i] for i in labeled_idxs_train]
        self._print('Using {} source (atlas) files and {} additional labeled training files!'.format(
            len(source_subject_files), len(self.files_labeled_train)))

        # remaining training examples must be unlabeled examples
        unlabeled_idxs = [i for i in range(n_train) if i not in labeled_idxs_train]

        self.files_unlabeled_train = [self.files_train[i] for i in unlabeled_idxs]

        # sanity check: none of our validation files should be in the training files list
        remove_valid_idxs = [i for i in range(len(self.files_labeled_valid))
                             if self.files_labeled_valid[i] in self.files_train]
        self._print('Removing {} from validation set (also in training)'.format(
            [self.files_labeled_valid[i] for i in remove_valid_idxs]))
        self.files_labeled_valid = [self.files_labeled_valid[i] for i in range(len(self.files_labeled_valid))
                                    if i not in remove_valid_idxs]

        assert set(sorted(self.files_train)).isdisjoint(set(sorted(self.files_labeled_valid)))
        assert set(sorted(self.files_labeled_train)).isdisjoint(set(sorted(self.files_labeled_valid)))
        assert set(sorted(self.files_unlabeled_train)).isdisjoint(set(sorted(self.files_labeled_valid)))


    def _print(self, msg):
        '''
        Prints the message to either the stdout + file logger (if initialized) or stdout as usual
        :param msg: a string
        :return:
        '''
        if self.logger is not None:
            self.logger.debug(msg)
        else:
            print(msg)


    def gen_vols_batch(self, dataset_splits=['labeled_train'],
                       batch_size=1, randomize=True,
                       load_segs=False, load_contours=False,
                       convert_onehot=False,
                       label_mapping=None, return_ids=False):

        if not isinstance(dataset_splits, list):
            dataset_splits = [dataset_splits]

        X_all = []
        Y_all = []
        contours_all = []
        files_list = []
        for ds in dataset_splits:
            if ds == 'labeled_train':
                X_all.append(self.vols_labeled_train)
                Y_all.append(self.segs_labeled_train)
                contours_all.append(self.contours_labeled_train)
                files_list += self.files_labeled_train
            elif ds == 'labeled_valid':
                X_all.append(self.vols_labeled_valid)
                Y_all.append(self.segs_labeled_valid)
                contours_all.append(self.contours_labeled_valid)
                files_list += self.files_labeled_valid
            elif ds == 'unlabeled_train':
                X_all.append(self.vols_unlabeled_train)
                Y_all.append(self.segs_unlabeled_train)
                contours_all.append(self.contours_unlabeled_train)
                files_list += self.files_unlabeled_train
            elif ds == 'labeled_test':
                if self.logger is not None:
                    self.logger.debug('LOOKING FOR FINAL TEST SET')
                else:
                    print('LOOKING FOR FINAL TEST SET')
                X_all.append(self.vols_labeled_test)
                Y_all.append(self.segs_labeled_test)
                contours_all.append(self.contours_labeled_test)
                files_list += self.files_labeled_test

        n_files = len(files_list)
        n_loaded_vols = np.sum([x.shape[0] for x in X_all])
        # if all of the vols are loaded, so we can sample from vols instead of loading from file
        if n_loaded_vols == n_files:
            load_from_files = False

            X_all = np.concatenate(X_all, axis=0)
            if load_segs and len(Y_all) > 0:
                Y_all = np.concatenate(Y_all, axis=0)
            else:
                Y_all = None

            if load_contours and len(contours_all) > 0:
                contours_all = np.concatenate(contours_all, axis=0)
            else:
                contours_all = None
        else:
            load_from_files = True

        if load_from_files:
            self._print('Sampling size {} batches from {} files!'.format(batch_size, n_files))
        else:
            self._print('Sampling size {} batches from {} volumes!'.format(batch_size, n_files))

        if randomize:
            idxs = np.random.choice(n_files, batch_size, replace=True)
        else:
            idxs = np.linspace(0, min(n_files, batch_size), batch_size, endpoint=False, dtype=int)

        while True:
            start = time.time()

            if not load_from_files:
                # if vols are pre-loaded, simply sample them
                X = X_all[idxs]

                if load_segs and Y_all is not None:
                    Y = Y_all[idxs]
                else:
                    Y = None

                if load_contours and contours_all is not None:
                    contours = contours_all[idxs]
                else:
                    contours = None

                batch_files = [files_list[i] for i in idxs]
            else:
                X = [None] * batch_size

                if load_segs:
                    Y = [None] * batch_size
                else:
                    Y = None

                if load_contours:
                    contours = [None] * batch_size
                else:
                    contours = None

                batch_files = []
                # load from files as we go
                for i, idx in enumerate(idxs.tolist()):
                    x, y, curr_contours = load_vol_and_seg(files_list[idx],
                                                           load_seg=load_segs, load_contours=load_contours,
                                                           do_mask_vol=True,
                                                           keep_labels=self.label_mapping,
                                                           )
                    batch_files.append(files_list[idx])
                    X[i] = x[np.newaxis, ...]

                    if load_segs:
                        Y[i] = y[np.newaxis, ...]

                    if load_contours:
                        contours[i] = curr_contours[np.newaxis]

            if self.profiler_logger is not None:
                self.profiler_logger.info('Loading vol took {}'.format(time.time() - start))

            # if we loaded these as lists, turn them into ndarrays
            if isinstance(X, list):
                X = np.concatenate(X, axis=0)

            if load_segs and isinstance(Y, list):
                Y = np.concatenate(Y, axis=0)

            if load_contours and isinstance(contours, list):
                contours = np.concatenate(contours, axis=0)

            # pick idxs for the next batch
            if randomize:
                idxs = np.random.choice(n_files, batch_size, replace=True)
            else:
                idxs += batch_size
                idxs[idxs > n_files - 1] -= n_files

            if load_segs and convert_onehot:
                start = time.time()
                Y = utils.labels_to_onehot(Y, label_mapping=label_mapping)
                if self.profiler_logger is not None:
                    self.profiler_logger.info('Converting vol onehot took {}'.format(time.time() - start))
            elif load_segs and not convert_onehot and not Y.shape[-1] == 1: # make sure we have a channels dim
                Y = Y[..., np.newaxis]

            if not return_ids:
                yield X, Y, contours
            else:
                yield X, Y, contours, batch_files



