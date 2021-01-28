import os, time
import yaml
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

# Import Visualization
import matplotlib
import matplotlib.pyplot as plt

# Import Helper Function
from helper import _rotational

start_timer = time.time()

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Initialize
INFO = {}
start_fr = 0

for path_i, path in tqdm(enumerate(glob(f"{config['input_data_path']}/**/*.h5"))):
    ### Setup Folders
    folder_name = os.path.basename(path).split("DLC")[0]
    save_path = f"{config['result_path']}/{folder_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ## Setup INFO
    INFO[folder_name] = {}
    INFO[folder_name]['directory'] = save_path
    INFO[folder_name]['order'] = path_i

    ### Import H5 Data
    store = pd.HDFStore(path, mode='a')
    df = store['df_with_missing']
    x_data = df.xs('x', level="coords", axis=1).to_numpy()
    y_data = df.xs('y', level="coords", axis=1).to_numpy()
    likelihood = df.xs('likelihood', level="coords", axis=1).to_numpy()
    store.close()

    ### Format
    DLC_data = np.concatenate((
        np.expand_dims(x_data, axis=-1), 
        np.expand_dims(y_data, axis=-1), 
        np.expand_dims(likelihood, axis=-1)), axis=-1)
    num_fr,_,_ = DLC_data.shape
    if config['save_bodypoints']:
        np.save(f"{save_path}/bodypoints.npy", DLC_data)

    ### Reevaluate likelihood
    # Check if BP exceeds a certain range
    x_bp, y_bp = DLC_data[:,:,0], DLC_data[:,:,1]
    x_mean, x_std = np.mean(x_bp), np.std(x_bp)
    y_mean, y_std = np.mean(y_bp), np.std(y_bp)

    x_bound = np.array([x_mean-config['x_bound_std']*x_std, x_mean+config['x_bound_std']*x_std])
    y_bound = np.array([y_mean-config['y_bound_std']*y_std, y_mean+config['y_bound_std']*y_std])

    x_condition = (DLC_data[:,:,0]>x_bound[1]) | (DLC_data[:,:,0]<x_bound[0])
    y_condition = (DLC_data[:,:,1]>y_bound[1]) | (DLC_data[:,:,1]<y_bound[0])
    (out_bound_fr, out_bound_marker) = np.where(x_condition | y_condition)
    DLC_data[out_bound_fr,out_bound_marker,2] = 0
    # Check if the BP moves to quickly
    marker_change = np.diff(DLC_data[:,:,0:2], axis=0)**2
    marker_velocity = np.sqrt(np.sum(marker_change, axis=2))
    (above_velocity_fr, above_velocity_marker) = np.where(marker_velocity > config['velocity_thresh'])
    above_velocity_fr+=1
    DLC_data[above_velocity_fr, above_velocity_marker, 2] = 0
    # Check if skeleton component is too long
    for joint1_idx, joint2_idx in config['skeleton']:
        joint1 = DLC_data[:,joint1_idx,:]
        joint2 = DLC_data[:,joint2_idx,:]
        skel_i = np.sqrt((joint1[:,0]-joint2[:,0])**2 + (joint1[:,1]-joint2[:,1])**2)
        bad_skel_fr = np.where(skel_i>config['max_limb_length'])[0]
        DLC_data[bad_skel_fr,joint1_idx,2] = 0
        DLC_data[bad_skel_fr,joint2_idx,2] = 0
    # TODO: plot_skeleton_length(skel_len)

    ### Center
    DLC_data[:,:,0:2] -= DLC_data[:,config['bp_center'],0:2][:,np.newaxis,:]
    
    ### Scale
    x_d = DLC_data[:,config['bp_scale'][0],0] - DLC_data[:,config['bp_scale'][1],0]
    y_d = DLC_data[:,config['bp_scale'][0],1] - DLC_data[:,config['bp_scale'][1],1]
    dist = np.sqrt(x_d**2+y_d**2)
    scale_factor = np.median(dist)
    DLC_data /= scale_factor    
    INFO[folder_name]['scale_factor'] = round(scale_factor.tolist(), 3)
    if config['save_scaled_bodypoints']:
        np.save(f"{save_path}/scaled_bodypoints.npy", DLC_data)

    ### Data Correction
    # TODO: make everything with below threshold likelihood as (0,0)
    # TODO: do low pass filter

    ### Rotate
    DLC_data[:,:,0:2], body_orientation = _rotational(data=DLC_data[:,:,0:2], axis_bp=config['bp_rotate'])
    DLC_data[:,:,2] = likelihood

    if config['save_body_orientation_angles']:
        np.save(f"{save_path}/body_orientation_angles.npy", body_orientation)
    if config['save_rotated_bodypoints']:
        np.save(f"{save_path}/rotated_bodypoints.npy", DLC_data)

    ### Record Files
    INFO[folder_name]['number_frames'] = num_fr
    INFO[folder_name]['global_start_fr'] = start_fr
    INFO[folder_name]['global_stop_fr'] = start_fr+num_fr
    start_fr += num_fr

print(f"::: Data Preprocessing ::: Computation Time: {time.time()-start_timer}")

with open(f"{config['result_path']}/INFO.yaml", 'w') as file:
    documents = yaml.dump(INFO, file)


