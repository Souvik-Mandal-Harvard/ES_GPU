import sys, os, time
import yaml
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

# Import Helper Function
from helper import _rotational

def main():
    # grab arguments
    config_name = sys.argv[1]

    # load config file
    with open(config_name) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    start_timer = time.time()

    # Initialize
    INFO = {}
    start_fr = 0
    for path_i, path in tqdm(enumerate(glob(f"{config['input_data_path']}/*.h5"))):
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

        ### nan bp should have low likelihood
        nan_bp_fr = np.where(np.isnan(x_data)|np.isnan(y_data))[0]
        likelihood[nan_bp_fr] = 0.0

        ### Format
        DLC_data = np.concatenate((
            np.expand_dims(x_data, axis=-1), 
            np.expand_dims(y_data, axis=-1), 
            np.expand_dims(likelihood, axis=-1)), axis=-1)
        num_fr,_,_ = DLC_data.shape
        if config['save_bodypoints']:
            np.save(f"{save_path}/bodypoints.npy", DLC_data)
        # TODO: filter

        ### Center
        bad_fr, bad_ax = np.where(np.isnan(DLC_data[:,config['bp_center'],0:2]))
        unique_bad_fr = np.unique(bad_fr)
        # find the unique good fr
        good_idx = np.array([True]*num_fr)
        good_idx[unique_bad_fr] = False

        DLC_data[good_idx,:,0:2] -= DLC_data[good_idx,config['bp_center'],0:2][:,np.newaxis,:]
        
        ### Scale
        if config['scale']:
            DLC_data[:,:,0:2] /= config['scale']
            INFO[folder_name]['scale_factor'] = config['scale']
        else:
            # TODO: see if this is necessary
            # find bad fr that contains nan
            bp_axis = DLC_data[:,config['bp_scale'],0:2]
            bad_fr, bad_bp, bad_ax = np.where(np.isnan(bp_axis))
            unique_bad_fr = np.unique(bad_fr)
            # find the unique good fr
            good_idx = np.array([True]*num_fr)
            good_idx[unique_bad_fr] = False
            good_bp_axis = bp_axis[good_idx,:,:]

            # find the median of these unique good fr
            x_d = good_bp_axis[:,0,0] - good_bp_axis[:,1,0]
            y_d = good_bp_axis[:,0,1] - good_bp_axis[:,1,1]
            dist = np.sqrt(x_d**2+y_d**2)
            scale_factor = np.median(dist)
            DLC_data[:,:,0:2] /= scale_factor
            INFO[folder_name]['scale_factor'] = round(scale_factor.tolist(), 3)
        if config['save_scaled_bodypoints']:
            np.save(f"{save_path}/scaled_bodypoints.npy", DLC_data)

        ### Rotate
        DLC_data[:,:,0:2], body_orientation = _rotational(data=DLC_data[:,:,0:2], axis_bp=config['bp_rotate'])
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

if __name__ == "__main__":
    main()
