[< Back to README.md](../README.md)

# Configuration File | *config.md*

The configuration file is where you will define the parameters for the models used in this pipeline will be defined. This is also where you will update your camera setup to help inform the rate of different behavioral changes. Hopefully, this document will help you explain some of the parameters in a intuitive manner with possible ethological explanation. 
*Note: It may require you to run multiple iterations of thes pipeline with different parameters to get the most optimal behavioral space*

# BM_GPU Parameter

### fps | *(int) - (0, inf]*
Put your camera frame rate here in frames per seconds. The fps of your camera is very important for the rest of the pipeline because it determines how refine you can track an animal behavior. For instance, if the rate at which an ant's leg moves is 15 cycles per second but your camera only records 25 frames per second, then you may not be able to capture the full stride of each motion.

Furthermore, it is highly recommend that you get as high of a fps camera so that you have enough points between each time point for you to later filter or correct your time series signal for any noises. Depending on how strong the noise is compared to your desired signal, false behavioral clusters may be formed. At the same time, make sure that the higher fps camera does not impede too much of the camera quality. Lower pixel resolution may also lead body point estimation algorithm to misplace points and/or introduce undesired noise. Finding the right balance between fps and pixel quality is a very important step that varies depending on the animal model you are observing. For instance animals like horses are larger but may be very fast, so in this case you may want to focus more on improving your fps over your pixel resolution. If you have an animal like an ant, which is both small and fast, then you want to make sure that you have a camera with high fps and pixel resolution to properly capture the ants' locomotion.

### input_data_path | *(str) - "data path"*
This parameter simply defines the path to your data, which we recommend you place in the data folder. However, in theory, you can place the data anywhere on your computer so long as your provide the entire, absolute path and link the folder with Docker. Moreover, please format your data by placing all of your body pose estimation data (e.g. data.h5 files, etc) in one folder under the name of whatever you would like to call the project or iteration. For example:
* BM_GPU
	* **data**
		* **ants_iter1**
			* *ant1_data1_trial1.h5*
			* *ant1_data1_trial2.h5*
			* *ant2_data1_trial1.h5*
			* *ant2_data1_trial2.h5*
			* *ant1_data2_trial1.h5*
		* **ants_iter2**
		* **dogs_iter1**
		* **cats**
	* **results**
	* **utils**
	* *config.yaml*
	* ...

### result_path | *(str) - "data path"*
This is where Ethoscope will place all of the data and figures for each project you run. Similar to *input_data_path* you can define a separate directory, but we recommend you keep it in the results folder for organization purpose. We also recommend that you keep the same project name directory so that you know exactly which results correspond to the dataset in the *data* folder.

### skeleton | *(list)*
The skeleton parameter is used to help create some of the figures when visualizing the animal locomotion. How you set up the skeleton will NOT affect the outcome of the behavioral space. This parameter takes a list of 2 points, which are the 2 body points you want the figures to connect/draw a line. The index of the bodypoint depends on the sequence and order of the points in the original raw data (e.g. data.h5, etc).
*Put a diagram here*

### likelihood_thresh | *(float) - [0,1]*
This parameter goes hand and hand with the next parameter (i.e. marker_thresh). Both of these variables are created to help the pipeline recognize wich frames have good body point labels for behavioral clustering. Using inaccurate body posture estimates may lead to skewed data and outliers in the two dimensional behavioral space. Therefore, the pipeline disregards these bad frames and places a value of -1 to indicate that no syllable labels were provided. The parameter likelihood_thresh informs at which likelihood (0 being most inaccurate and 1 being most accurate) for a given body point you are willing to tolerate for the pipeline to continue clustering behaviors.

### marker_thresh | *(int) - [1, # of bodypoints]*
marker_thresh supplements the parameter above and informs the pipeline how many body points with below threshold requirement you are willing to tolerate before the pipeline disregards the frame entirely. Note that the pipeline labels a bad frame as -1 during ethogram generation to indicate that no label has been provided.

### bad_fr_pad | *(int) - [0, inf]*
Along with the two parameters above, this parameter sets the number of frames that should pad the bad frames identified above. The purpose of this function is so that when the pipeline computes the kinematic features, which takes into account frames before and after the frame of interest, no misleading information is introduced from the bad frames. Without proper padding, outliers may still be created in the behavioral space, creating inaccurate representaiton of the animal's behaviors. 

### bp_center | *(int) - [0, # bodypoints-1]*
This parameter is used to index the body point that should be used to center the animal of interest. The purpose of this is so that when the behavioral space is clustered, the syllables are not created based on where the subject is relative to the camera's frame of reference. For example, we would not want a subject's walking behavior to cluster differently when it occurs at the top right corner of the image versus the bottom left corner. By centering the bodypoints to a specific point of reference, it will also help analyze the subject behavior across subject and time.

### bp_scale | *(list)*
bp_scale is a list of 2 body point index. The length of these two bodypoint index informs how all the body points should be scaled. The purpose of this functionality is to make sure that different sizes of subject will not affect the behavioral analysis. This is particularly useful in the case of ants where different caste of ants have different sizes. However, this is also useful in the case of mouse or rats that may differ in size. By scaling every subject matter for each video, we can make sure that the behavioral clusters are not biased based on the subject's size.

*figure to help determine which two points should be used for scaling*

### scale | *(false) or (float)*
Sometimes it may be hard to find two points to scale each subject for each video. For example, if the subject rotates in the 3D plain and the body point estimates only provide the 2 coordinates, there most likely be no body segments as a good point of reference. In this case, we allow you to provide one constant scaling factor for all of your subjects and videos. The purpose of this is to lower the overall coordinates of each point, making it easier to run other models later on.

### bp_rotate | *(int) - [0, # bodypoints-1]*
One last point, bp_rotate, is provided to the pipeline so that it is able to rotate the subject matter such that the bp_center and bp_rotate lies only on the vertical axis. The purpose of this is purely for visualization and does affect the final outcome of the behavior. Fixing the model's center of axis to one direction may help researchers focus on each body segment movement rather than the orientation of the animal. 

*figure to help determine which point should be used for rotating*

### markers | *(list)*
"markers" is one of the 3 parameters (the other two being "angles" and "limbs") that define which features are used to create the behavioral space. Markers should be a list of body point indexes, where the x and y cartesian coordinates are used to define the behavior.

*currently we do not use this parameter*

### angles | *(list)*
"angles" is the second category of features that consist of a list of sets, which consist of 3 body points to define the angle. The set should be formatted with 4 keys (a,b,c,method). The values for a, b, and c sets the 3 body points used to compute the angle between body segment (ab) and (bc). 

*{"a": 14, "b": 2, "c": 1, "method": 1}*

### limbs | *(list)*
"limbs" is the third category of features that consist of a list of tuple, each containing two body point index. This allows user to use body segment length between two body posture estimate to define the behavioral space.

# Morlet Wavelet Transformation Parameter
### f_min | *(float)*
The minimum number of frequency, relative to the camera frame rate, you expect the animal to move any given feature. The max frequency that is used to analyze the body movements is Nyquist frequency (i.e. half the camera's frame rate)
### f_bin | *(float)*
The number of frequency bins define how many frequency points between the min frequency (i.e. f_min) and max frequency you would like to use to analyze a feature's signal. The higher the f_bin, the higher you can analyze the kinematic motion. However, note that increast f_bin dramatically increases the dimension of the high-dimensional behavioral space, causeing more memory usage and longer computation time.


# UMAP Parameter
Please read [UMAP API](https://umap-learn.readthedocs.io/en/latest/api.html) to understand the technical aspect of each parameter. We also reccomend that you go through their [user guide](https://umap-learn.readthedocs.io/en/latest/basic_usage.html) to grasp how each parameter affects the manifold. 
### n_components
### n_neighbors
### n_epochs
### min_dist
### spread
### negative_sample_rate
### init
### repulsion_strength















