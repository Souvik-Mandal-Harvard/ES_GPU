# ES_GPU
### Version 1.0.0

---
# Overview
**Ethoscope is a "free for academic use" software that extracts behaviors from positional bodypoints, derived from pose estimator software (like [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut), [DeepPoseKit](https://github.com/jgraving/DeepPoseKit), [SLEAP](https://sleap.ai/) etc.), using unsupervised models.** 

The workflow in Ethoscope is largely divided into 5 consecutive steps:

### STEP 1: Editing config.yaml
Each animal system and experimental setup can be unique and defined by some parameters. This file contains such parameters necessary to be defined for the proper functioning of the rest of the workflow; users can change the parameters according to the need. For details of the parameters, please click here open the config.yaml file and follow the instructions.

<!-- ### STEP 2: data_formatting.py
* config parameter: 
   * DLC
   * SLEAP
* helper function for converting different data format to npy
 Please note, users do not need to edit this file if the primary source of pose data is DLC or SLEAP. If the user uses some other software for body point estimation, then format the data into npy. -->

### STEP 2: Run preproessing.py
Animals can show same behavior while they have different body orientations. Also, their body size can appear different to the camera due to their distance from the camera. Due to these two factors, same behavior can be flagged as different by an automated system. This preprocessing step accounts for these variations by centering, rotating and scaling the body poses by transforming the raw data for the next steps.

### STEP 3: Run features.py
Behavior is defined by combinations of diffrent body movements called behavioral syllables. Imagine behavioral syalllable as letters - different combination of which produce different words, or in the context of behavior science, different behaviors. Ethoscope uses both body postures as well as the kinematics of the animal to define behavioral syllables. Postural features are calculated using the eucledian distance and angle between different body points. Next, Ethoscope performs a Morlet-wavelet transformation on the postural data to coupute the kinematics features. This step generates 4 .npy files in total for each video sample - one contaning data of the Eucledian distances between the labeled body points (limbs.npy), one for the angular data (angles.npy), and two for the kinematics power spectrogram (limb_power.npy, angle_power.npy).

### STEP 4: embed_gpu.py
**THIS STEP REQUIRES A GPU on a local computer. In future, we will come up with pipeline that can use cloud-GPU computing and/or CPU (much slower).**

Then, Ethoscope uses these multi-dimensional postural and kinematic feature dataset and reduces it to two-dimension first using PCA (to reduce the dimention of kinamatics data) and then UMAP (to reduce the dimension of the kinematic principal components and the postural data). Users can get three different files by running this file - all_embeddings.npy, all_kinematic_embeddings.npy, and all_postural_embeddings.npy.

By default, these outcome files of this step will be a two-dimentional behavioral space. However, users can change the number of final dimention by editing the parameter "n_components" under UMAP heading in the config.yaml file.



### Step 5: cluster.py
**This step may require a GPU depending on the clustering model users select**

Taking the low-dimensional embedding of the behavioral space, this step labels each frame as a behavioral syllable using one of the clustering methods (i.e. Watershed, HDBSCAN). These cluster labels serve as the primary syllables, which can then be utilized to create higher order ethograms.

***Although this version (1.0.0) of Ethoscope requires a GPU-enabled computer, we are in the process of integrating the pipeline to utilize cloud GPU like Google CoLab***

---
# Getting Started

### Setup Environment
Ethoscope uses [RapidsAI](https://rapids.ai/), an open-source software libraries with GPU compatibile models. Currently, RapidAI is compatible only with Linux (Ubuntu 16.04 and up) system. This reduces the computation time to get the behavioral space significantly compared to other pipelines (that primarily uses CPU or multi-CPU). To create a rapids.ai Docker container users can follow the instructions outlined on their [website](https://rapids.ai/start.html).

#### Step 1: Create RapidsAI Docker Container
`docker run --gpus all -it -p 8888:8888 -p 8787:8787 -p 8786:8786 \
   --name <CONTAINER_NAME> \
   -v <PROJECT_PATH>:/rapids/notebooks/host \
   -w /rapids/notebooks/host \
    rapidsai/rapidsai-core:cuda11.0-runtime-ubuntu18.04-py3.8`   

#### Step 2: Install Other Third Party Libraries Into Your Contianer
* Installing [ffmpeg (linux)](https://linuxize.com/post/how-to-install-ffmpeg-on-ubuntu-18-04/)   
`apt update`   
`apt install ffmpeg`   
* Installing [scikit-video](http://www.scikit-video.org/stable/)   
`pip install scikit-video`
* Installing tables
`pip install tables   


### Run Docker Container
Once you have created your Docker contianer, the libraries' versions and environment should all be set for you to run through our pipeline. You do not have to recreate the container after you have everything setup.
### Step 1: Start/Restart Container   
`docker start -i <CONTAINER_NAME>`   
### Step 2: Upload Your Dataset



---
# For Developers

If you don't want to restart your kernel in jupyterlab to update your functions, use the following python codes so that your function of concern is updated whenever changes are made.   
%load_ext autoreload   
* %autoreload 0 - disables the auto-reloading. This is the default setting.
* %autoreload 1 - it will only auto-reload modules that were imported using the %aimport function (e.g %aimport my_module). It’s a good option if you want to specifically auto-reload only a selected module.
* %autoreload 2 - auto-reload all the modules. Great way to make writing and testing your modules much easier.

### TODO
- [X] TODO 1
- [ ] TODO 2
- [ ] TODO 3
- [ ] TODO 4
- [ ] TODO 5





