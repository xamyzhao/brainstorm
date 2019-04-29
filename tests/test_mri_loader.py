import os
import sys
# include external libraries in path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ext', 'pytools-lib'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ext', 'neuron'))

import numpy as np

from src.mri_loader import get_dataset_files_list, MRIDataset, voxelmorph_labels, load_vol_and_seg

def test_get_vol_files_list():
    # test training set
    train_files = get_dataset_files_list(mode='train')
    assert np.all(['train' in tf for tf in train_files])
    assert np.all(['valid' not in tf for tf in train_files])
    assert np.all(['test' not in tf for tf in train_files])
    # test validation set
    valid_files = get_dataset_files_list(mode='validate')
    assert np.all(['train' not in tf for tf in valid_files])
    assert np.all(['valid' in tf for tf in valid_files])
    assert np.all(['test' not in tf for tf in valid_files])

    print('_get_vol_files_list unit tests PASSED')


def test_load_dataset_unlabeled():
    ds = MRIDataset(params={
        'n_shot': 0,
        'n_validation': 1,
        'n_unlabeled': 1,
        'do_preload_vols': True,
        'use_subjects_as_source': ['atlas'],
        'use_labels': voxelmorph_labels
    })
    ds.load_dataset(load_segs=True,
                    load_source_segs=True, load_source_contours=False)

    # check that we loaded the specified number of files
    assert len(ds.files_labeled_train) == 1
    assert len(ds.files_unlabeled_train) == 1
    assert len(ds.files_labeled_valid) == 1

    # manually load the files and see that they match what is in the dataset
    check_train_idxs = [0]
    for ti in check_train_idxs:
        vol, seg, contours = load_vol_and_seg(
            vol_file=ds.files_labeled_train[ti],
            do_mask_vol=True,
            keep_labels=ds.label_mapping,
        )
        diff = vol - np.reshape(ds.vols_labeled_train[ti], vol.shape)
        assert np.all(diff < np.finfo(np.float32).eps)

    check_valid_idxs = [0]
    for ti in check_valid_idxs:
        vol, seg, contours = load_vol_and_seg(
            vol_file=ds.files_labeled_valid[ti],
            do_mask_vol=True,
            keep_labels=ds.label_mapping,
        )
        diff = vol - np.reshape(ds.vols_labeled_valid[ti], vol.shape)
        assert np.all(diff < np.finfo(np.float32).eps)

    # make sure we are not mixing train and validation
    assert np.all(['valid' not in tf for tf in ds.files_unlabeled_train])
    assert np.all(['test' not in tf for tf in ds.files_unlabeled_train])

    assert np.all(['train' not in tf for tf in ds.files_labeled_valid])
    assert np.all(['test' not in tf for tf in ds.files_labeled_valid])
    print('MRIDataset load_dataset unit tests PASSED')


def test_load_dataset_labeled():
    ds = MRIDataset(params={
        'n_shot': 1,
        'n_validation': 1,
        'n_unlabeled': 0,
        'do_preload_vols': True,
        'use_subjects_as_source': ['atlas'],
        'use_labels': voxelmorph_labels
    })
    ds.load_dataset(load_segs=True,
                    load_source_segs=True, load_source_contours=False)

    # check that we loaded the specified number of files
    assert len(ds.files_labeled_train) == 2
    assert len(ds.files_labeled_valid) == 1

    # check that the training volumes are different
    assert not np.all((ds.vols_labeled_train[1] - ds.vols_labeled_train[0]) < np.finfo(np.float32).eps)

    # make sure we are not mixing train and validation
    assert np.all(['valid' not in tf for tf in ds.files_unlabeled_train])
    assert np.all(['test' not in tf for tf in ds.files_unlabeled_train])

    assert np.all(['train' not in tf for tf in ds.files_labeled_valid])
    assert np.all(['test' not in tf for tf in ds.files_labeled_valid])
    print('MRIDataset load_dataset unit tests PASSED')


def _test_mrids_load_source_vol():
    ds = MRIDataset(params={
        'n_shot': 1,
        'n_validation': 1,
        'n_unlabeled': 1,
        'do_preload_vols': False,
        'use_subjects_as_source': ['OASIS_OAS1_0327_MR1_mri_talairach_orig'],
        'use_labels': voxelmorph_labels
    })
    ds._get_files_list()
    ds.load_dataset(load_source_segs=True, load_source_contours=True)
    # Y should consist of segmentation labels (channel 0) and contours (channel 1)
    assert ds.segs_labeled_train.shape[-1] == 2
    assert len(np.unique(ds.segs_labeled_train[..., 0])) == 31  # 30 voxelmorph labels plus bkg
    assert len(np.unique(ds.segs_labeled_train[..., 1])) == 2  # should jsut be 0, 1
    assert ds.vols_labeled_train.shape[0] == 1
    print('MRIDataset load_source_vol unit tests PASSED')