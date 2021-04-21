# ES_GPU

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

---
# How to Get Started
python -m ipykernel install --user --name=notebook_analysis
conda activate notebook_analysis

> TODO: dong; create instructions for setting up the environment
conda env export > environment.yml
conda env create -f environment.yml



%load_ext autoreload
---
%autoreload 0 - disables the auto-reloading. This is the default setting.
%autoreload 1 - it will only auto-reload modules that were imported using the %aimport function (e.g %aimport my_module). It’s a good option if you want to specifically auto-reload only a selected module.
%autoreload 2 - auto-reload all the modules. Great way to make writing and testing your modules much easier.






