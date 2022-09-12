# README

### About this repository

We hereby provide to sourcecode that was used to conceive the DeepMB paper [1].

### About DeepMB

DeepMB is a deep-learning-based framework to reconstruct multispectral optoacoustic tomography images in real-time. DeepMB combines two previously antagonistic properties — image quality and reconstruction time:

1. **State-of-the-art image quality:** DeepMB reconstructs images that are nearly-identical (both quantitatively and qualitatively) to state-of-the-art iterative model-based reconstructions, while being markedly better than commonly-used backprojection reconstructions.

2. **Real-time capability:** DeepMB reconstructs images in 31 ms, which is a comparable rate to lower quality backprojection reconstructions, but about 3000 times faster than state-of-the-art iterative model-based reconstructions.

Further unique features of DeepMB, with respect to existing methods, include:

3. **Tunable speed of sound:** DeepMB supports dynamic adjustments of the speed of sound parameter during imaging, which enables the reconstruction of in-focus images for arbitrary biological tissue types.

4. **In vivo applicability:** DeepMB accurately generalizes to in vivo data after training on synthesized sinograms that are derived from real-world images.

5. **Compatibility with high-end scanners:** DeepMB is directly compatible with state-of-the-art clinical optoacoustic scanners because it supports high throughput data acquisition (sampling rate: 40 MHz; number of transducers: 256) and large image sizes (416×416 pixels).

6. **Straightforward applicability:** DeepMB can be used in the context of other optoacoustic reconstruction approaches that currently suffer from prohibitive reconstruction times, such as frequency-band model-based reconstruction (to disentangle structures of different physical scales) or Bayesian optoacoustic reconstruction (to quantify reconstruction uncertainty). More generally, the underlying methodology of DeepMB can also be exploited to accelerate parametrized (iterative) inversion approaches for other imaging modalities, such as ultrasound, X-ray computed tomography, magnetic resonance imaging, or, more generally, for any parametric partial differential equation.

DeepMB was developed in the Jüstel Lab, Institute for Biological and Medical Imaging at Helmholtz Zentrum München, Chair of Biological Imaging at Technische Universität München, and Institute of Computational Biology at Helmholtz Zentrum München, together with iThera Medical GmbH.

### General working principle of DeepMB

1. During network training, the mapping ``(input sinogram, input speed of sound) ➝ output image`` is learned , using the corresponding model-based reconstruction as the ``reference image``.
2. During network inference, the mapping ``(input sinogram, input speed of sound) ➝ output image`` is applied.

### Citation

Please cite [1] if you use this code and/or any of the provided binary files.

### DeepMB reference

> [1] [Christoph Dehner, Guillaume Zahnd, Dominik Jüstel, and Vasilis Ntziachristos. DeepMB: Deep neural network for real-time model-based optoacoustic image reconstruction with adjustable speed of sound. 2022.](https://arxiv.org/abs/2206.14485)

### Contact

* Christoph Dehner (christoph.dehner@tum.de)
* Guillaume Zahnd (guillaume.zahnd@ithera-medical.com)

## About the folder ``DeepMB > DeepMB_trn_val_tst >``

##### Purpose:

Scripts to perform the training and the inference of the DeepMB network. Three functionalities are available:

1. ``DeepMB > DeepMB_trn_val_tst > run_DeepMB_trn_val_tst.py``: Routine to train the network.
2. ``DeepMB > DeepMB_trn_val_tst > batch_DeepMB_ifr.py``: Script to infer a trained network on all samples of a given dataset.
3. ``DeepMB > DeepMB_trn_val_tst > demo_DeepMB_ifr.py``: Toy example to infer a trained network on a single specific sample, meant to be easily adaptable to your own specific use-case.

##### Prerequisites

Set up a python (3.8 or higher) environment and install the following packages: `numpy`, `matplotlib`, `torch`, `torchvision`, `tensorboard`, `medpy`, and `natsort`. To run the `demo_DeepMB_ifr.py` script, please install `requests`. To run the ONNX model, please install `onnxruntime` and `onnxruntime-gpu`.

#### About the script ``DeepMB > DeepMB_trn_val_tst > run_DeepMB_trn_val_tst.py``

##### How to run the script:

1. Create a local copy of the template file ``DeepMB > DeepMB_trn_val_tst > package_parameters > set_parameters_template.py`` (do not modify this Git-tracked template).
2. Rename the newly-created file into ``DeepMB > DeepMB_trn_val_tst > package_parameters > set_parameters.py`` (this working copy is Git-ignored).
3. In ``DeepMB > DeepMB_trn_val_tst > package_parameters > set_parameters.py``, specify all the variables according to your own local environment and experiment design.
4. Run the script ``DeepMB > DeepMB_trn_val_tst > run_DeepMB_trn_val_tst.py`` to train the network using the specified parameters.

##### Output:

1. The Git-tracked local file ``DeepMB > DeepMB_trn_val_tst > package_parameters > get_parameters.py`` is automatically created to backup your experiment settings.
2. The Git-ignored local file ``SAVE_PATH > EXPERIMENT_NAME > stored_parameters > get_parameters.py`` is automatically created to be utilized during your future inferences.
3. The network weights and bias, corresponding to the validation epoch of minimal loss, are stored in ``SAVE_PATH > EXPERIMENT_NAME > model_checkpoint > model_min_val_loss.pt``.
4. For each epoch, an example image of the training, validation, and testing sets is printed in ``SAVE_PATH > EXPERIMENT_NAME > print_epoch_trn_val_tst > ``.
5. The tensorboard log is saved in ``SAVE_PATH > tensorboard_logs > EXPERIMENT_NAME > ``.

#### About the script ``DeepMB > DeepMB_trn_val_tst > batch_DeepMB_ifr.py``

##### How to run the script:

1. Specify the variables ``SAVE_PATH`` and ``EXPERIMENT_NAME`` to indicate what previously trained network shall be used.
2. Set the flag ``trn_val_tst`` to either ``trn`` (training), ``val`` (validation), or ``tst`` (testing) to indicate which dataset shall be inferred.
3. Set the flags ``SAVE_NII`` and ``SAVE_PNG`` to either ``True`` or ``False`` to indicate if NIFTI files and/or PNG images, respectively, shall be saved.
4. If desired, modify the field ``REGEX_ADDITIONAL_NAME_FILTER`` to restrict the inference to files with particular names (optional).
5. Run the script ``DeepMB > DeepMB_trn_val_tst > batch_DeepMB_ifr.py`` to infer the selected trained network onto the specified dataset.

##### Output:

1. All inferred samples are saved (as NIFTI files and/or PNG images) within ``SAVE_PATH > EXPERIMENT_NAME > batch_ifr >``.

#### About the script ``DeepMB > DeepMB_trn_val_tst > demo_DeepMB_ifr.py``

##### How to run the script:

1. Specify the variable ``sinogram_filename`` to indicate what sample shall be inferred (you can select one file that is already provided on our [GitHub storage](https://github.com/juestellab/deepmb/tree/binaries/in_vivo_data/sinograms).
2. Specify the variable ``sos_value`` to indicate the sound velocity value that shall be used during inference.
3. Specify the variables ``use_tracing`` and ``use_onnx`` to indicate whether optional acceleration features shall be employed.
4. Run the script ``DeepMB > DeepMB_trn_val_tst > demo_DeepMB_ifr.py`` to infer the specified sample (all the necessary data will be automatically downloaded from our GitHub repository)

##### Output:

1. A figure is displayed, showing the input sinogram and the reconstructed DeepMB image.

## About the folder ``DeepMB > processus_data_generation >``

##### Purpose:

Scripts to generate the datasets necessary to train the DeepMB network. Two functionalities are available:

1. ``DeepMB > processus_data_generation > generate_synthetic_dataset.m``: Generate pairs of synthetic (input sinogram, target ground truth model-based images) from initial synthetic images.
2. ``DeepMB > processus_data_generation > generate_invivo_dataset.m``: Generate target ground-truth in vivo model-based images from input in vivo sinograms.

##### Prerequisites

1. Check out the MSOT Model-based Reconstruction Toolbox:
[https://github.com/juestellab/mb-rec-msot](https://github.com/juestellab/mb-rec-msot).

2. Download the Pascal VOC 2012 dataset from either
[http://host.robots.ox.ac.uk/pascal/VOC/index.html](http://host.robots.ox.ac.uk/pascal/VOC/index.html)
or
[https://pjreddie.com/projects/pascal-voc-dataset-mirror/](https://pjreddie.com/projects/pascal-voc-dataset-mirror/).

#### About the script ``DeepMB > processus_data_generation > generate_synthetic_dataset.m``

##### How to run the script:

1. Create a local copy of the template file ``DeepMB > processus_data_generation > set_parameters_for_data_generation_template.m`` (do not modify this Git-tracked template).
2. Rename the newly-created file into ``DeepMB > processus_data_generation > set_parameters_for_data_generation.m`` (this working copy is Git-ignored).
3. In ``DeepMB > processus_data_generation > set_parameters_for_data_generation.m``, specify all the variables according to your own local environment and experiment design.
4. Run the script ``DeepMB > processus_data_generation > generate_synthetic_dataset.m`` to launch the data generation.

##### Output:

1. ``initial_images``: True initial pressure images, re-sized to the specified dimensions.
2. ``sinograms``: Sinograms, resulting from the application of the forward model to the true initial pressure images.
3. ``rec_images``: Model-based reconstructions.
4. ``backprojection``: Backprojection reconstructions.
5. ``sos_sim_and_rec``: Two speed of sound values (SoS used during the forward simulation, and SoS value used during reconstruction).

#### About the script ``DeepMB > processus_data_generation > generate_invivo_dataset.m``

##### How to run the script:

1. Create a local copy of the template file ``DeepMB > processus_data_generation > set_parameters_for_data_generation_template.m`` (do not modify this Git-tracked template).
2. Rename the newly-created file into ``DeepMB > processus_data_generation > set_parameters_for_data_generation.m`` (this working copy is Git-ignored).
3. In ``DeepMB > processus_data_generation > set_parameters_for_data_generation.m``, specify all the variables according to your own local environment and experiment design.
4. Run the script ``DeepMB > processus_data_generation > generate_invivo_dataset.m`` to launch the data generation.

##### Output:

1. ``sinograms``: Sinograms, after pre-processing (bandpass filter, cropping, optional detector-wise interpolation, optional handling of broken detectors).
2. ``rec_images``: Model-based reconstructions.
3. ``backprojection``: Backprojection reconstructions.
4. ``sos_sim_and_rec``: Two speed of sound values (provided SoS value from manual tuning, and SoS value used during reconstruction).

## About the folder ``DeepMB > processus_residual_norms_calculation >``

##### Purpose:

Scripts to calculate the data residual norms and the regularization terms of the reconstructed samples, thus providing quantified information about the reconstruction fidelity. One functionality is available:

1. ``DeepMB > processus_residual_norms_calculation > calculate_residuals_for_gt_and_inferred_recons.m``

##### Prerequisites

1. Check out the MSOT Model-based Reconstruction Toolbox:
[https://github.com/juestellab/mb-rec-msot](https://github.com/juestellab/mb-rec-msot)

#### About the script ``DeepMB > processus_residual_norms_calculation > calculate_residuals_for_gt_and_inferred_recons.m``

##### How to run the script:

1. Create a local copy of the template file ``DeepMB > processus_residual_norms_calculation > set_parameters_for_residual_norms_calculation_template.m`` (do not modify this Git-tracked template).
2. Rename the newly-created file into ``DeepMB > processus_residual_norms_calculation > set_parameters_for_residual_norms_calculation.m`` (this working copy is Git-ignored).
3. In ``DeepMB > processus_residual_norms_calculation > set_parameters_for_residual_norms_calculation.m``, specify all the variables according to your own local environment and experiment design.
4. Run the training script ``DeepMB > processus_residual_norms_calculation > calculate_residuals_for_gt_and_inferred_recons.m`` to launch the calculation.

##### Output:

1. Data residual norms and/or regularization terms, for DeepMB, model-based, and/or backprojection reconstructions.
