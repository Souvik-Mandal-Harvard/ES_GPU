# ES_GPU
### Version 1.0.0

---
# Overview
**Ethoscope is a "free for academic use" software which extracts behaviors from positional bodypoints, extracted from pose estimators like DeepLabCut, DeepPoseKit, and SLEAP, that uses unsupervised models.** 

Ethoscope can be largely divided into 5 different steps:

### STEP 1: config.yaml
This file contains the parameters necessary to set-up the pipeline - users can change the parameters according to the need. For details of the parameters, please read this and/or open the config file and follow the instructions.

### STEP 2: preproessing.py
Animals can show same behavior while they have different body orientations. Also, their body size can look different to the camera due to their distance from the camera. Due to these two factors, same behavior can be flagged as different behavior by an automated system. This preprocessing step accounts for these variations by centering, rotating and scaling the body poses by transforming the raw data for the next steps.

### STEP 3: features.py
Behavior is defined by combinations of diffrent behavioral syllables (imagine behavioral syalllable as letters, different combination of which produce different words, or in our case, different behaviors). Ethoscope uses both body postures as well as the kinematics of the animal to define behavioral syllables. Postural features are calculated using the eucledian distance and angle between different body points. Next, Ethoscope performs a Morlet-wavelet transformation on the postural data to coupute the kinematics features. This step generates 4 .npy files in total for each video sample - one contaning data of the Eucledian distances, one for teh angular data, and two for the kinematics power spectrogram.

### STEP 4: embed_gpu.py
**THIS STEP REQUIRES A GPU on a local computer In future, we will come up with pipeline that can use cloud-GPU computing and/or CPU (much slower).**

Then, Ethoscope uses these multi-dimensional postural and kinematic feature dataset and reduces it two-dimension using the various dimensional-reduction methods (i.e. UMAP, PCA).

### Step 5: cluster.py
**This step may require a GPU depending on the clustering model you select**

Using the low-dimensional embedding of the behavioral space, one can use one of many different clustering methods to label each frame. These cluster labels serve as the primary syllables, which can then be utilized to create higher order ethograms.
***Although this Ethoscope version 1.0.0 requires GPU, we are also in the process of integrating the pipeline to utilize cloud GPU***

---
# How to Get Started
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

### Run Docker Container
Once you have created your Docker contianer, the libraries' versions and environment should all be set for you to run through our pipeline.  
### Step 1: Start/Restart Container    
`docker start -i <CONTAINER_NAME>`




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





