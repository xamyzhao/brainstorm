# brainstorm
This repository contains the authors' implementation from 
"Data Augmentation using Learned Transformations for One-shot Medical Image Segmentation", which will be 
presented as an oral at CVPR 2019. We provide code for training spatial and appearance transform models, and for using 
the transform models to synthesize training examples for segmentation.   


If you use our code, please cite:

**Data augmentation using learned transforms for one-shot medical image segmentation**  
[Amy Zhao](https://people.csail.mit.edu/xamyzhao), [Guha Balakrishnan](https://people.csail.mit.edu/balakg/), [Fredo Durand](https://people.csail.mit.edu/fredo), [John Guttag](https://people.csail.mit.edu/guttag), [Adrian V. Dalca](adalca.mit.edu)  
CVPR 2019. [eprint arXiv:1902.09383](https://arxiv.org/abs/1902.09383)


# Getting started
## Prerequisites
To run this code, you will need:
* Python 3.6+ (Python 2.7 may work but has not been tested)
* CUDA 10.0+
* Tensorflow 1.13+ and Keras 2.2.4+
* one GPU with 12 GB of memory (we used a single NVIDIA Titan X)

## Download module dependencies
Run the script setup.sh. This will automatically pull the following dependencies and place them in the correct subdirectories:

* https://github.com/adalca/neuron (for SpatialTransformer layer)
* https://github.com/adalca/pytools-lib (for various segmentation utilities) 

## Setting up your dataset
We have included a few sample MRI scans (including volumes and segmentations) in the `data/` folder. If you wish to use the datasets mentioned in the paper, you should download them directly from the respective dataset sites. 

If you wish to use your own dataset, place your volume and segmentation files in the `data/` folder. 
The data loading code in `src/mri_loader.py` expects each example to be stored as a volume file `{example_id}_vol.npz` and, 
if applicable, a corresponding `{example_id}_seg.npz` file, with the data stored in each file using the keys `vol_data` 
and `seg_data` respectively. The functions `load_dataset_files` and `load_vol_and_seg` in `src/mri_loader.py` can be easily 
modified to suit your data format.
 

## Training transform models
This repo does not include any pre-trained models. You may train your own 
spatial and appearance transform models by specifying the GPU ID, dataset name, and the model type.

```
python main.py trans --gpu 0 --data mri-100unlabeled --model flow-fwd
python main.py trans --gpu 0 --data mri-100unlabeled --model flow-bck
python main.py trans --gpu 0 --data mri-100unlabeled --model color-unet
```
The results will be placed in `experiments/`. Note that in order to train an appearance/color transform model, you will want
 to edit `main.py` to point at your trained forward/backward spatial transform models. We have provided pretrained forward/backward 
 spatial transform models for testing.


As described in the paper, each model is implemented using a simple architecture based on [U-Net](https://arxiv.org/abs/1505.04597).
You can change hyperparameters by modifying `transform_model_arch_params` in `main.py`.  We encourage you to experiment with your 
favorite model architecture, and to adjust the model parameters to suit your dataset. 

## Training a segmentation network
You may train a segmentation model by specifying the GPU ID and dataset name.
```
python main.py fss --gpu 0 --data mri-100unlabeled
```
Again, results will be placed under `experiments/`. 


You can use additional flags:
* `--aug_rand` will apply random augmentation to each training example consisting of a random smooth deformation and a random global multiplicative intensity factor.
* `--aug_sas` will pseudo-label any unlabeled examples in the training set using the specified spatial registration model.
* `--aug_tm` will synthesize training examples using our method.

If you wish to use `--aug_sas` or `--aug_tm`, it is important to specify the spatial and appearance transform models to use in
`seg_model_arch_params` in `main.py`.    

# Evaluation
To evaluate trained segmenters, look at the code in `evaluate_segmenters.py`.
You will have to modify the code to point at your trained models.

<sub>Repo name inspired by Magic: The Gathering.</sub>

![Brainstorm](http://gatherer.wizards.com/Handlers/Image.ashx?multiverseid=451037&type=card)
