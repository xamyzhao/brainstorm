import glob
import numpy as np
import os
import time

import sys
sys.path.append('../evolving_wilds')
from cnn_utils import classification_utils

sys.path.append('../voxelmorph-sandbox')
from voxelmorph.visualization import segutils as vm_segutils


voxelmorph_labels =  [0,
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
                     17, 53,  # hippo
                     14, 15,  # 3rd 4th vent
                     18, 54,  # amyg
                     24,  # csf
                     3, 42,  # cerebral cortex
                     31, 63,  # choroid plexus
                     ]

vm_vol_root = '/data/ddmg/voxelmorph/data/t1_mix/proc/resize256-crop_x32'
mri_contours_root = '/afs/csail.mit.edu/u/x/xamyzhao/adni_data/contours'
buckner_vol_root = '/data/ddmg/voxelmorph/data/buckner/proc/resize256-crop_x32/FromEugenio_prep2'

class MRIDataset(object):
    def __init__(self, params, logger=None, profiler_logger=None):
        # default parameter settings that might not be in keys
        if 'load_vols' not in params.keys():
            params['load_vols'] = False

        if 'dataset_root_train' not in params.keys():
            params['dataset_root_train'] = 'vm'

        if 'dataset_root_valid' not in params.keys():
            params['dataset_root_valid'] = 'vm'

        if 'n_shot' not in params.keys():
            params['n_shot'] = 0

        self.profiler_logger = profiler_logger
        self.params = params
        self.logger = logger

        # TODO: do this later?
        self.create_display_name()


    def create_display_name(self, display_name_base=''):
        self.display_name = 'mri-tr-{}-valid-{}'.format(
            self.params['dataset_root_train'], self.params['dataset_root_valid'])

        self.display_name += '_{}ul'.format(self.params['n_unlabeled'])

        if self.params['use_atlas_as_source']:
            # use only the voxelmorph atlas as a source vol
            self.display_name += '_atlas-l'
        elif 'use_subjects_as_source' in self.params.keys() and self.params['use_subjects_as_source'] is not None:
            if not isinstance(self.params['use_subjects_as_source'], list):
                self.params['use_subjects_as_source'] = [self.params['use_subjects_as_source']]

            self.display_name += '_subj-l'
            for source_subject in self.params['use_subjects_as_source']:
                if buckner_vol_root in source_subject:
                    self.display_name += '-{}'.format(os.path.splitext(os.path.basename(source_subject))[0])
                else:
                    self.display_name += '-{}'.format('_'.join(source_subject.split('_')[:3]))

        if 'split_id' in self.params.keys() and self.params['split_id'] is not None:
            self.display_name += '_split{}'.format(self.params['split_id'])

        return self.display_name


    def _get_files_list(self):
        if 'use_labels' in self.params.keys():
            # filter by user-specified labels
            self.label_mapping = self.params['use_labels']
        else:
            self.label_mapping = None


        # get a list of volume files that we will load from later
        self.train_files = _get_vol_files_list('train', dataset_root=self.params['dataset_root_train'],
            get_unnormalized=True, exclude_PPMI=True,
            check_for_matching_segs=self.params['dataset_root_train']=='buckner',
        )

        # buckner has some missing segs, so we need to filter the volumes
        self.valid_files = _get_vol_files_list('validate', dataset_root=self.params['dataset_root_valid'],
            get_unnormalized=True, exclude_PPMI=True,
            check_for_matching_segs=self.params['dataset_root_valid']=='buckner')

        np.random.seed(17)
        np.random.shuffle(self.train_files)

        if 'valid_seed' in self.params.keys() and self.params['valid_seed'] is not None:
            np.random.seed(self.params['valid_seed'])
        np.random.shuffle(self.valid_files)

        # exclude files that we used in the cvpr test set
        if 'exclude_from_validation_list' in self.params.keys():
            with open(self.params['exclude_from_validation_list'], 'r') as f:
                exclude_subjects = f.readlines()
            exclude_subjects = [f.strip() for f in exclude_subjects]

            remove_valid_idxs = [i for i,f in enumerate(self.valid_files) if f in exclude_subjects]
            remove_valid_files = [self.valid_files[i] for i in remove_valid_idxs]
            self._print('Excluding {} subjects from validation set: {}'.format(len(remove_valid_idxs), remove_valid_files))
            valid_files_new = [self.valid_files[i] for i in range(len(self.valid_files)) if i not in remove_valid_idxs]
            self.valid_files = valid_files_new[:]

        # get validation files from the end
        self.valid_files = self.valid_files[:self.params['n_validation']]

        if self.params['n_unlabeled'] is None or self.params['n_unlabeled'] == -1:
            # if any of n_unlabeled or n_shot is -1, that means to use everything
            self.n_train = len(self.train_files)
        else:
            # only use the number that was requested
            self.n_train = self.params['n_unlabeled'] + self.params['n_shot']

        load_additional_subject_files = []
        if 'use_subjects_as_source' in self.params.keys():
            # make sure we include the source subject file in the training list
            for source_subject in self.params['use_subjects_as_source']:
                print(source_subject)
                subject_file_idx = [i for i, f in enumerate(self.train_files) if source_subject in f]

                if len(subject_file_idx) == 0:
                    # we couldnt find it in the train files, assume we got a full filename
                    subject_file = source_subject
                else:
                    subject_file_idx = subject_file_idx[0]
                    subject_file = self.train_files[subject_file_idx]

                if subject_file in self.train_files:
                    # dont include this in the training set yet because we will prune the training set
                    self.train_files.remove(subject_file)

                load_additional_subject_files.append(subject_file)
        # only store as many files as we need
        self.train_files = list(set(
            self.train_files[:self.n_train] + load_additional_subject_files))
        self.n_train = len(self.train_files)

        self._print('Got {} training files and {} additional source (atlas) files!'.format(
            len(self.train_files), len(load_additional_subject_files)))

        remove_valid_idxs = [i for i in range(len(self.valid_files)) if self.valid_files[i] in self.train_files]
        self._print('Removing {} from validation set (also in training)'.format(
            [self.valid_files[i] for i in remove_valid_idxs]))
        self.valid_files = [self.valid_files[i] for i in range(len(self.valid_files)) if i not in remove_valid_idxs]

        assert set(sorted(self.train_files)).isdisjoint(set(sorted(self.valid_files)))


    def _load_source_vol(self, load_n=None,
                         load_source_segs=False, load_source_contours=False):
        # first set aside our labeled example. Always load this volume even if we are not loading the others
        if self.params['use_atlas_as_source']:
            # just use the atlas as the single labeled example
            self.X_atlas, self.segs_atlas, self.Y_contours_atlas = _load_vol_and_seg(
                'atlas', load_seg=True,
                do_mask_vol=False,
                keep_labels=self.label_mapping,
                load_contours=load_source_contours,
            )
            self.X_atlas = np.expand_dims(self.X_atlas, axis=0)
            self.segs_atlas = np.expand_dims(self.segs_atlas, axis=0)
            if self.Y_contours_atlas is not None:
                self.Y_contours_atlas = np.expand_dims(self.Y_contours_atlas, axis=0)

            self.vols_labeled_train = self.X_atlas
            self.segs_labeled_train = self.segs_atlas
            self.contours_labeled_train = self.Y_contours_atlas

            self.files_labeled_train = ['atlas']
            self.ids_labeled_train = ['atlas']
            labeled_idxs_train = []

        elif 'use_subjects_as_source' in self.params.keys() and self.params['n_shot'] == 0:
            self._print('Using specified subjects as atlases: {}'.format(
                self.params['use_subjects_as_source']))
            # pick out the subject we selected
            labeled_idxs_train = []
            for source_subject in self.params['use_subjects_as_source']:
                labeled_idxs_train += [i for i, f in enumerate(self.train_files) if source_subject in f]

            self.files_labeled_train = [self.train_files[i] for i in labeled_idxs_train]
            self.vols_labeled_train, self.segs_labeled_train, self.contours_labeled_train, self.ids_labeled_train \
                = load_dataset_vols(
                vol_files=[self.train_files[i] for i in labeled_idxs_train],
                load_segs=load_source_segs,
                load_contours=load_source_contours,
                mask_vols=True,
                use_labels=self.label_mapping,
            )
        else:
            self._print('Including {} subjects in training set'.format(
                self.params['n_shot']))
            if 'split_id' in self.params.keys():
                np.random.seed(self.params['split_id'])

            # select a limited number of labeled training examples
            labeled_idxs_train = np.random.choice(
                self.n_train, min(self.n_train - 1, self.params['n_shot']), replace=False)

            self.files_labeled_train = [self.train_files[i] for i in labeled_idxs_train]
            self.vols_labeled_train, self.segs_labeled_train, self.contours_labeled_train, self.ids_labeled_train \
                = load_dataset_vols(
                vol_files=[self.train_files[i] for i in labeled_idxs_train],
                load_segs=load_source_segs,
                mask_vols=True,
                load_n=load_n,
                use_labels=self.label_mapping,
            )
        return labeled_idxs_train

    def _print(self, msg):
        '''
        Prints the message to either the file logger (if initialized) or stdout
        :param msg: a string
        :return:
        '''
        if self.logger is not None:
            self.logger.debug(msg)
        else:
            print(msg)


    def load_dataset(self, load_n=None, load_segs=True, load_source_segs=True,
                     load_source_contours=True, valid_files_list=None):
        '''
        Main entry point for loading dataset
        :return: (X, Y, ids) tuples for unlabeled set, labeled training set and labeled validation set
        '''
        self._print('Loading mri dataset {}'.format(self.display_name))
        self._print('Params: {}'.format(self.params))

        self._get_files_list()

        self._print('Got lists of {} training and {} validation files!'.format(len(self.train_files), len(self.valid_files)))

        labeled_idxs_train = self._load_source_vol(load_source_segs=load_source_segs,
                                                   load_source_contours=load_source_contours,
                                                   load_n=load_n)
        if self.logger is not None:
            self.logger.debug('Labeled train vols:')
            self.logger.debug('X_labeled_train: {}'.format(self.vols_labeled_train.shape))
            if load_segs:
                self.logger.debug('Y_labeled_train: {}'.format(self.segs_labeled_train.shape))
            self.logger.debug('ids_labeled_train: {}'.format(self.ids_labeled_train))

        else:
            print('Labeled train vols:')
            print('X_labeled_train: {}'.format(self.vols_labeled_train.shape))
            if load_segs:
                print('Y_labeled_train: {}'.format(self.segs_labeled_train.shape))
            print('ids_labeled_train: {}'.format(self.ids_labeled_train))


        # now pick out our validation set from the valid set and our "unlabeled" set from the training set
        unlabeled_idxs = [i for i in range(self.n_train) if i not in labeled_idxs_train]
        self.files_unlabeled_train = [self.train_files[i] for i in unlabeled_idxs]
        if valid_files_list is not None:
            self.files_labeled_valid = valid_files_list
            self._print('Setting validation set to {} files in input list'.format(len(valid_files_list)))
        else:
            self.files_labeled_valid = self.valid_files

        # set to None by default
        self.segs_labeled_valid = None
        self.segs_unlabeled_train = None
        if not self.params['load_vols']:
            # load some dummy vols so we can get the correct shapes
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
                = load_dataset_vols(
                vol_files=self.files_labeled_valid,
                load_n=load_n,
                load_segs=load_segs,
                load_contours=False,
                mask_vols=True,
                use_labels=self.label_mapping,
            )

            self.vols_unlabeled_train, self.segs_unlabeled_train, self.contours_unlabeled_train, self.ids_unlabeled_train \
                = load_dataset_vols(
                vol_files=self.files_unlabeled_train,
                load_n=load_n,
                load_segs=load_segs,
                mask_vols=True,
                use_labels=self.label_mapping
            )

        # filter labels only if we have not done so already
        if self.label_mapping is None and load_segs:
            # keep common labels
            atlas_labels = np.unique(self.segs_atlas)
            subject_labels = np.unique(
                np.concatenate([
                    np.unique(self.segs_labeled_train),
                    np.unique(self.segs_labeled_valid),
                    np.unique(self.segs_unlabeled_train)]))

            all_labels = np.unique(np.concatenate([atlas_labels, subject_labels]))
            segs_all_flat = np.concatenate([
                np.reshape(self.segs_atlas, (1, -1)),
                np.reshape(self.segs_labeled_train, (self.segs_labeled_train.shape[0], -1)),
                np.reshape(self.segs_labeled_valid, (self.segs_labeled_valid.shape[0], -1)),
                np.reshape(self.segs_unlabeled_train, (self.segs_unlabeled_train.shape[0], -1))
            ], axis=0)
            self.label_mapping = []
            for l in all_labels:
                if np.all(np.any(segs_all_flat == l, axis=-1), axis=0):
                    # keep the label if it appears in every example
                    self.label_mapping.append(l)

            self.label_mapping = np.asarray(sorted(self.label_mapping))

            # filter all volumes by our label mapping
            if self.logger is not None:
                self.logger.debug('Filtering labels to {}'.format(self.label_mapping))
            else:
                print('Filtering labels to {}'.format(self.label_mapping))

            for l in atlas_labels:
                if l not in self.label_mapping:
                    self.segs_atlas[self.segs_atlas == l] = 0

            for l in subject_labels:
                if l not in self.label_mapping:
                    self.segs_labeled_train[self.segs_labeled_train == l] = 0
                    self.segs_labeled_valid[self.segs_labeled_valid == l] = 0
                    self.segs_unlabeled_train[self.segs_unlabeled_train == l] = 0
        elif self.label_mapping is None and not load_segs:
            self.label_mapping = [0]

        # load final test set and filter labels
        if 'final_test' in self.params.keys() and self.params['final_test']:
            self.files_labeled_test = _get_vol_files_list('test', get_unnormalized=True)

            np.random.seed(self.params['test_seed'])
            np.random.shuffle(self.files_labeled_test)

            self.files_labeled_test = self.files_labeled_test[:self.params['n_test']]
            self.vols_labeled_test, self.segs_labeled_test, self.contours_labeled_test, self.ids_labeled_test \
                = load_dataset_vols(
                vol_files=self.files_labeled_test,
                load_n=self.params['n_test'],
                load_segs=True,
                load_contours=False,
                mask_vols=True,
                use_labels=self.label_mapping,
            )

        return (self.vols_unlabeled_train, self.segs_unlabeled_train, self.contours_unlabeled_train, self.files_unlabeled_train), \
                (self.vols_labeled_train, self.segs_labeled_train, self.contours_labeled_train, self.files_labeled_train), \
                (self.vols_labeled_valid, self.segs_labeled_valid, self.contours_labeled_valid, self.files_labeled_valid), \
                self.label_mapping


    # TODO: make laod_source_contours into a param
    def load_source_target(self, load_n=None, load_source_segs=False):
        '''
        Main entry point for loading unsupervised dataset. Does not load segmentations by default, to save space.
        :param debug:
        :return:
        '''
        if self.params['use_atlas_as_source']:
            (vols_train_unlabeled, segs_train_unlabeled, contours_train_unlabeled, ids_train_unlabeled), \
            (vol_atlas, segs_atlas, contours_atlas, ids_atlas), \
            (vols_valid_labeled, segs_valid_labeled, contours_valid_labeled, ids_valid_labeled), \
            label_mapping \
            = self.load_dataset(
                load_n=load_n, load_segs=False, load_source_segs=load_source_segs, load_source_contours=True,
            )

            # source is atlas, target is unlabeled set.
            # for validation, use atlas as source and "labeled" validation set
            return (vol_atlas, segs_atlas, contours_atlas, ids_atlas), \
                   (vols_train_unlabeled, segs_train_unlabeled, contours_train_unlabeled, ids_train_unlabeled), \
                   (vol_atlas, segs_atlas, contours_atlas, ids_atlas), \
                   (vols_valid_labeled, segs_valid_labeled, contours_valid_labeled, ids_valid_labeled), \
                   self.label_mapping
        elif 'use_subjects_as_source' in self.params.keys() and self.params['use_subjects_as_source'] is not None:

            (vols_train_unlabeled, segs_train_unlabeled, contours_train_unlabeled, ids_train_unlabeled), \
            (vols_train_labeled, segs_train_labeled, contours_train_labeled, ids_train_labeled), \
            (vols_valid_labeled, segs_valid_labeled, contours_valid_labeled, ids_valid_labeled), label_mapping \
            = self.load_dataset(
                load_n=load_n, load_segs=False, load_source_contours=True, load_source_segs=load_source_segs,
            )

            # source is the chosen subject, target is unlabeled set. for validation, use subject as source and "labeled" validation set
            return (vols_train_labeled, segs_train_labeled, contours_train_labeled, ids_train_labeled), \
                   (vols_train_unlabeled, segs_train_unlabeled, contours_train_unlabeled, ids_train_unlabeled), \
                   (vols_train_labeled, segs_train_labeled, contours_train_labeled, ids_train_labeled), \
                   (vols_valid_labeled, segs_valid_labeled, contours_valid_labeled, ids_valid_labeled), \
                   self.label_mapping

        else:
            if self.logger is not None:
                self.logger.debug('Non-canonical source-target split not supported!')
            else:
                print('Non-canonical source-target split not supported!')
            return None


    def gen_slices_batch(self, files_list, batch_size=1, randomize=True,
                       convert_onehot=False,
                       label_mapping=None):

        if randomize:
            np.random.shuffle(files_list)

        n_files = len(files_list)

        slice_idxs = np.linspace(0, batch_size, batch_size, endpoint=False, dtype=int)
        file_idx = 0
        while True:
            start = time.time()
            X, Y = _load_vol_and_seg(
                files_list[file_idx], do_mask_vol=True,
            )
            # TODO: add scaling here since we no longer do it per example

            if self.profiler_logger is not None:
                self.profiler_logger.info('Loading vol took {}'.format(time.time() - start))

            start = time.time()
            # slice the z-axis by default
            n_slices = X.shape[-2]
            if randomize:
                slice_idxs = np.random.choice(n_slices, batch_size, replace=True)
            else:
                # if we are going through the dataset sequentially, make the last batch smaller
                slice_idxs += batch_size
                slice_idxs[slice_idxs > n_slices - 1] = []
                file_idx += 1

            X = np.transpose(X[:, :, slice_idxs], (2, 0, 1, 3))
            Y = np.transpose(Y[:, :, slice_idxs], (2, 0, 1))
            if self.profiler_logger is not None:
                self.profiler_logger.info('Slicing took {}'.format(time.time() - start))

            if randomize:
                # randomize the shuffled files every batch
                file_idx += 1
                if file_idx > n_files - 1:
                    file_idx = 0
            elif not randomize and max(slice_idxs) >= n_slices - 1:
                # if we reached the end of teh slices of the previous file, start the slices again
                slice_idxs = np.linspace(0, batch_size, batch_size, endpoint=False, dtype=int)

            if convert_onehot:
                start = time.time()
                Y = classification_utils.labels_to_onehot(Y, label_mapping=label_mapping)
                if self.profiler_logger is not None:
                    self.profiler_logger.info('Converting slices onehot took {}'.format(time.time() - start))

            yield X, Y


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
                    x, y, curr_contours = _load_vol_and_seg(files_list[idx],
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
                Y = classification_utils.labels_to_onehot(Y, label_mapping=label_mapping)
                if self.profiler_logger is not None:
                    self.profiler_logger.info('Converting vol onehot took {}'.format(time.time() - start))
            elif load_segs and not convert_onehot and not Y.shape[-1] == 1: # make sure we have a channels dim
                Y = Y[..., np.newaxis]

            if not return_ids:
                yield X, Y, contours
            else:
                yield X, Y, contours, batch_files


def _test_mrids_load_source_vol():
    ds = MRIDataset(params={
        'n_shot': 1,
        'n_validation': 1,
        'n_unlabeled': 1,
        'unnormalized': True,
        'load_vols': False,
        'use_atlas_as_source': False,
        'use_subject': 'OASIS_OAS1_0327_MR1_mri_talairach_orig',
        'use_labels': voxelmorph_labels
    })
    ds._get_files_list()
    ds._load_source_vol(load_source_segs=True, load_source_contours=True)
    # Y should consist of segmentation labels (channel 0) and contours (channel 1)
    assert ds.segs_labeled_train.shape[-1] == 2
    assert len(np.unique(ds.segs_labeled_train[..., 0])) == 31  # 30 voxelmorph labels plus bkg
    assert len(np.unique(ds.segs_labeled_train[..., 1])) == 2  # should jsut be 0, 1
    assert ds.vols_labeled_train.shape[0] == 1
    print('MRIDataset load_source_vol unit tests PASSED')


def _test_mrids_load_dataset():
    ds = MRIDataset(params={
        'n_shot': 0,
        'n_validation': 2,
        'n_unlabeled': 5,
        'load_vols': True,
        'use_atlas_as_source': False,
        'use_subject': 'OASIS_OAS1_0327_MR1_mri_talairach_orig',
        'use_labels': voxelmorph_labels
    })
    ds.load_dataset(load_segs=True,
                    load_source_segs=True, load_source_contours=False)

    # check that we loaded the specified number of files
    assert len(ds.files_unlabeled_train) == 5
    assert len(ds.files_labeled_valid) == 2

    # check a few files to make sure they are correct
    check_train_idxs = [0, -1]
    for ti in check_train_idxs:
        vol, seg, contours = _load_vol_and_seg(
            vol_file=ds.files_unlabeled_train[ti],
            load_seg=True, do_mask_vol=False,
            load_contours=False,
            keep_labels=ds.label_mapping,
        )
        diff = vol - ds.vols_unlabeled_train[ti]
        assert np.all(diff < np.finfo(np.float32).eps)

    check_valid_idxs = [0, -1]
    for ti in check_valid_idxs:
        vol, seg, contours = _load_vol_and_seg(
            vol_file=ds.files_labeled_valid[ti],
            do_mask_vol=False
        )
        diff = vol - ds.vols_labeled_valid[ti]
        assert np.all(diff < np.finfo(np.float32).eps)

    # make sure we are not mixing train and validation
    assert np.all(['valid' not in tf for tf in ds.files_unlabeled_train])
    assert np.all(['test' not in tf for tf in ds.files_unlabeled_train])

    assert np.all(['train' not in tf for tf in ds.files_labeled_valid])
    assert np.all(['test' not in tf for tf in ds.files_labeled_valid])
    print('MRIDataset load_dataset unit tests PASSED')


def _get_vol_files_list(mode='train', dataset_root='vm', get_unnormalized=False, exclude_PPMI=True, check_for_matching_segs=False):
    if get_unnormalized:
        vols_folder = 'origs'
    else:
        vols_folder = 'vols'

    if dataset_root == 'vm':
        vols_dir = vm_vol_root + '/{}/{}/*.npz'.format(mode, vols_folder)
        segs_dir = vm_vol_root + '/' + mode + '/asegs/' + '*_aseg.npz'
    else:
        vols_dir = buckner_vol_root + '/{}/*.npz'.format(vols_folder)
        segs_dir = os.path.join(buckner_vol_root, 'asegs', '*.npz')

    vol_files = glob.glob(vols_dir)
    if check_for_matching_segs:
        seg_files = glob.glob(segs_dir)
        # exclude bad scans
        vol_files = [f for f in vol_files if np.any([os.path.splitext(os.path.basename(f))[0] in sf for sf in seg_files])
            and '990128_vc764' not in f]

    vol_files = sorted(vol_files)

    if exclude_PPMI:
        # these volumes are probably from a different modality
        vol_files = [vf for vf in vol_files if 'PPMI' not in vf]

    print('Got list of {} files from {}:'.format(len(vol_files), vols_dir))
    for vf in vol_files[:4]:
        print(os.path.basename(vf))
    print('...')

    return vol_files


def _test_get_vol_files_list():
    # test training set
    train_files = _get_vol_files_list(mode='train', get_unnormalized=True, exclude_PPMI=True)
    assert np.all(['train' in tf for tf in train_files])
    assert np.all(['valid' not in tf for tf in train_files])
    assert np.all(['test' not in tf for tf in train_files])

    # unnormalized should come from the origs folder
    assert np.all(['origs' in tf for tf in train_files])
    # check for no PPMI data
    assert np.all(['PPMI' not in tf for tf in train_files])

    # test validation set
    valid_files = _get_vol_files_list(mode='validate', get_unnormalized=False, exclude_PPMI=True)
    assert np.all(['train' not in tf for tf in valid_files])
    assert np.all(['validate' in tf for tf in valid_files])
    assert np.all(['test' not in tf for tf in valid_files])

    # unnormalized should come from the origs folder
    assert np.all(['origs' not in tf for tf in valid_files])
    assert np.all(['vols' in tf for tf in valid_files])
    # check for no PPMI data
    assert np.all(['PPMI' not in tf for tf in valid_files])
    print('_get_vol_files_list unit tests PASSED')


def _load_vol_and_seg(vol_file,
                      load_seg=True, do_mask_vol=False,
                      load_contours=False,
                      keep_labels=None,
                      ):
    if 'train' in vol_file:
        mode = 'train'
    elif 'valid' in vol_file:
        mode = 'validate'
    else:
        mode = 'test'

    # load volume and corresponding segmentation from file
    vol_name = os.path.splitext(os.path.basename(vol_file))[0]
    if vol_name == 'atlas':
        img_data, seg_data = _load_atlas_vol()
    else:
        if vm_vol_root in vol_file:
            vol_base_name = '_'.join(vol_name.split('_')[:-1])
        else:
            vol_base_name = vol_name

        img_data = np.load(vol_file)['vol_data'][..., np.newaxis]
        if load_seg or do_mask_vol or load_contours:
            if vm_vol_root in vol_file:
                seg_vol_file = vm_vol_root + '/' + mode + '/asegs/' + vol_base_name + '_aseg.npz'
            elif buckner_vol_root in vol_file:
                seg_vol_file = os.path.join(buckner_vol_root, 'asegs', vol_base_name + '.npz')
            else:
                print('Could not find corresponding seg file for {}!'.format(vol_file))
                sys.exit()
            if not os.path.isfile(seg_vol_file):
                print('Could not find corresponding seg file for {}!'.format(vol_file))
                return None

            seg_data = np.load(seg_vol_file)['vol_data']
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

        contours_file = os.path.join(mri_contours_root, vol_name + '_{}labels.npz'.format(len(keep_labels)))
    else:
        contours_file = os.path.join(mri_contours_root, vol_name + '_alllabels.npz')

    contours = None
    if load_contours:
        if not os.path.isfile(contours_file):
            contours = vm_segutils.seg2contour(seg_data, contour_type='both')[..., np.newaxis]
            contours[contours > 0] = 1
            np.savez(contours_file, contours_data=contours)
        else:
            contours = np.load(contours_file)['contours_data']

    return img_data, seg_data, contours


def load_dataset_vols(
        vol_files=None,
        load_n=None,
        dataset_root='vm',
        mode='train',
        get_unnormalized=False,
        load_segs=True,
        load_contours=False,
        mask_vols=False,
        use_labels=None,
):

    if vol_files is None:
        vol_files = _get_vol_files_list(mode=mode, dataset_root=dataset_root, get_unnormalized=get_unnormalized)

    if load_n is None:
        load_n = len(vol_files)

    # compute scaled volume size
    vol_size = (160, 192, 224)
    X = np.zeros((load_n,) + vol_size + (1,), dtype=np.float32)

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

        data = _load_vol_and_seg(
            vol_files[i],
            load_seg=load_segs,
            load_contours=load_contours,
            do_mask_vol=mask_vols,
            keep_labels=use_labels,
        )
        if data is None:
            continue

        X[i], curr_segs, curr_contours	= data
        if load_segs:
            Y_segs[i] = curr_segs
        if load_contours:
            Y_contours[i] = curr_contours

        vol_base_name = os.path.splitext(os.path.basename(vol_files[i]))[0]
        ids.append(vol_base_name)

    return X, Y_segs, Y_contours, ids


def _load_atlas_vol(slice_idx=None):
    atlas_root = '/afs/csail.mit.edu/u/x/xamyzhao/voxelmorph-sandbox/voxelmorph/data'
    atlas_data = np.load(os.path.join(atlas_root, 'atlas_norm.npz'))
    if slice_idx:
        return atlas_data['vol'][:,:,[slice_idx]], atlas_data['seg'][:,:,slice_idx]
    else:
        return atlas_data['vol'][..., np.newaxis], atlas_data['seg']


if __name__ == '__main__':
    _test_get_vol_files_list()
    _test_mrids_load_source_vol()
    _test_mrids_load_dataset()
