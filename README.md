# brainstorm
Implementation from the paper "Data Augmentation with Spatial and Appearance Transforms for One-shot Medical Image Segmentation".

This project has dependencies on the following repos. Please place all these repos in the same parent directory.

* https://github.com/xamyzhao/evolving_wilds (for general utils)
* https://github.com/voxelmorph/voxelmorph (for networks and losses)
* https://github.com/adalca/neuron (for SpatialTransformer layer)
* https://github.com/adalca/medipy-lib (for CPU implementation of Dice score)
* https://github.com/adalca/pynd-lib (for visualizations)

# Training transform models
Spatial and appearance transform models can be trained by specifying the GPU ID, dataset name, and model name.

```
python main.py trans --gpu 0 --data adni-100-csts2 --model flow-bds
python main.py trans --gpu 0 --data adni-100-csts2 --model color-unet
```
Each experiment will create a results directory under `./experiments` by default, so make sure that location exists.

# Training a segmentation network
A segmentation network can be trained with the following:
```
python main.py fss --gpu 0 --data adni-100-csts2
```
Again, results will be placed under `.experiments`. To evaluate trained segmenters, check out the code in `evaluate_segmenters.py`.

<sub>Repo name inspired by Magic: The Gathering.</sub>

![Evolving Wilds](http://gatherer.wizards.com/Handlers/Image.ashx?multiverseid=451037&type=card)