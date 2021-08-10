[< Back to README.md](../README.md)

# Configuration File | *config.md*

The configuration file is where you will define the parameters for the models used in this pipeline will be defined. This is also where you will update your camera setup to help inform the rate of different behavioral changes. Hopefully, this document will help you explain some of the parameters in a intuitive manner with possible ethological explanation. 
*Note: It may require you to run multiple iterations of thes pipeline with different parameters to get the most optimal behavioral space*

### fps
**(int) - (0, inf]**
Put your camera frame rate here in frames per seconds. The fps of your camera is very important for the rest of the pipeline because it determines how refine you can track an animal behavior. For instance, if the rate at which an ant's leg moves is 15 cycles per second but your camera only records 25 frames per second, then you may not be able to capture the full stride of each motion.

Furthermore, it is highly recommend that you get as high of a fps camera so that you have enough points between each time point for you to later filter or correct your time series signal for any noises. Depending on how strong the noise is compared to your desired signal, false behavioral clusters may be formed. At the same time, make sure that the higher fps camera does not impede too much of the camera quality. Lower pixel resolution may also lead body point estimation algorithm to misplace points and/or introduce undesired noise. Finding the right balance between fps and pixel quality is a very important step that varies depending on the animal model you are observing. For instance animals like horses are larger but may be very fast, so in this case you may want to focus more on improving your fps over your pixel resolution. If you have an animal like an ant, which is both small and fast, then you want to make sure that you have a camera with high fps and pixel resolution to properly capture the ants' locomotion.

### input_data_path
**(str) - "data path"**
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

### result_path
**(str) - "data path"**
This is where Ethoscope will place all of the data and figures for each project you run. Similar to *input_data_path* you can define a separate directory, but we recommend you keep it in the results folder for organization purpose. We also recommend that you keep the same project name directory so that you know exactly which results correspond to the dataset in the *data* folder.

### skeleton
**(list)**
The skeleton parameter is used to help create some of the figures when visualizing the animal locomotion. How you set up the skeleton will NOT affect the outcome of the behavioral space. This parameter takes a list of 2 points, which are the 2 body points you want the figures to connect/draw a line. The index of the bodypoint depends on the sequence and order of the points in the original raw data (e.g. data.h5, etc).
*Put a diagram here*

### likelihood_thresh
**(float) - [0,1]**
This parameter goes hand and hand with the next parameter (i.e. marker_thresh). Both of these variables are created to help the pipeline recognize wich frames have good body point labels for behavioral clustering. Using inaccurate body posture estimates may lead to skewed data and outliers in the two dimensional behavioral space. Therefore, the pipeline disregards these bad frames and places a value of -1 to indicate that no syllable labels were provided. The parameter likelihood_thresh informs at which likelihood (0 being most inaccurate and 1 being most accurate) for a given body point you are willing to tolerate for the pipeline to continue clustering behaviors.

### marker_thresh
**(int) - [1, # of bodypoints]**
marker_thresh supplements the parameter above and informs the pipeline how many body points with below threshold requirement you are willing to tolerate before the pipeline disregards the frame entirely. Note that the pipeline labels a bad frame as -1 during ethogram generation to indicate that no label has been provided.

### bad_fr_pad
**(int) = [0, inf]**
Along with the two parameters above, this parameter sets the number of frames that should pad the bad frames identified above. The purpose of this function is so that when the pipeline computes the kinematic features, which takes into account frames before and after the frame of interest, no misleading information is introduced from the bad frames. Without proper padding, outliers may still be created in the behavioral space, creating inaccurate representaiton of the animal's behaviors. 

