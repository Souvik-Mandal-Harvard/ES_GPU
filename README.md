# ES_GPU
### Version 1.0.0

---
# Overview
**Ethoscope is a "free for academic use" software that extracts behaviors from positional bodypoints, derived from pose estimator software (like DeepLabCut, DeepPoseKit, SLEAP etc.), using unsupervised models.** 

The workflow in Ethoscope is largely divided into 5 consecutive steps:

### STEP 1: config.yaml
Each animal system and experimental setup can be unique and defined by some parameters. This file contains such parameters necessary to be defined for the proper functioning of the rest of the workflow; users can change the parameters according to the need. For details of the parameters, please open the config file and follow the instructions.

### STEP 2: preproessing.py
Animals can show same behavior while they have different body orientations. Also, their body size can appear different to the camera due to their distance from the camera. Due to these two factors, same behavior can be flagged as different by an automated system. This preprocessing step accounts for these variations by centering, rotating and scaling the body poses by transforming the raw data for the next steps.

### STEP 3: features.py
Behavior is defined by combinations of diffrent body movements called behavioral syllables. Imagine behavioral syalllable as letters - different combination of which produce different words, or in the context of behavior science, different behaviors. Ethoscope uses both body postures as well as the kinematics of the animal to define behavioral syllables. Postural features are calculated using the eucledian distance and angle between different body points. Next, Ethoscope performs a Morlet-wavelet transformation on the postural data to coupute the kinematics features. This step generates 4 .npy files in total for each video sample - one contaning data of the Eucledian distances between the labeled body points (*name of the file?*), one for the angular data (*name of the file?*), and two for the kinematics power spectrogram (*name of the files?*).

### STEP 4: embed_gpu.py
**THIS STEP REQUIRES A GPU on a local computer. In future, we will come up with pipeline that can use cloud-GPU computing and/or CPU (much slower).**

Then, Ethoscope uses these multi-dimensional postural and kinematic feature dataset and reduces it two-dimension using the various dimensional-reduction methods (i.e. UMAP, PCA).

### Step 5: cluster.py
**This step may require a GPU depending on the clustering model users select**

Taking the low-dimensional embedding of the behavioral space, this step labels each frame as a behavioral syllable using one of the clustering methods (i.e. Watershed, HDBSCAN). These cluster labels serve as the primary syllables, which can then be utilized to create higher order ethograms.

***Although this version (1.0.0) of Ethoscope requires a GPU-enabled computer, we are in the process of integrating the pipeline to utilize cloud GPU like Google CoLab***

---
# Getting Started

### Setup Environment
Ethoscope uses [RapidsAI](https://rapids.ai/), an open-source software libraries with GPU compatibile models. This allows for us to compute the behavioral space significantly quicker compared to other pipelines, which primarily uses CPU or multi-CPU. To create a rapids.ai Docker container one should follow the instructions clearly outlined on their [website](https://rapids.ai/start.html).

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





