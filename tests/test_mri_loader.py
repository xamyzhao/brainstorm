import numpy as np

from src.mri_loader import _get_vol_files_list, MRIDataset, voxelmorph_labels, _load_vol_and_seg


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