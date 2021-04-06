# ES_GPU

> here is a change

python -m ipykernel install --user --name=notebook_analysis
conda activate notebook_analysis

> TODO: dong; create instructions for setting up the environment
conda env export > environment.yml
conda env create -f environment.yml



%load_ext autoreload
________________________________
%autoreload 0 - disables the auto-reloading. This is the default setting.
%autoreload 1 - it will only auto-reload modules that were imported using the %aimport function (e.g %aimport my_module). It’s a good option if you want to specifically auto-reload only a selected module.
%autoreload 2 - auto-reload all the modules. Great way to make writing and testing your modules much easier.


------------------------------------------------------------------------------------------------------------------------------------------------------
Ethoscope is a "free for academic use" software which extracts behaviors using unsupervised methods from any body positional data (including markerless pose estimators like DeepLabCut). 

This file contains detailed instructions and descriptions of Ethoscope. Once the user gets body points, the user can take following steps.

STEP 1: config.yaml
This contains the parameters necessary to set-up the pipeline - users can change the parameters according to the need. For details, please open the config file and follow the instructions.

STEP 2: preproessing.py
Animals can show same behavior while they have different body orientation. Also, their body size can look different to the camera due to theor distance from the camera. Due to these two factors, same behavior can be flagged as different behavior for an automated system. This preprocessing step accounts for these variations by centering, rotating and scaling the body poses by transforming the raw data for the next steps.

STEP 3: features.py


STEP 4: embed_gpu.py

