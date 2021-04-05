import time
import yaml, pickle
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Import Helper Function
from helper import locate_bad_fr, angle_calc, morlet

def main():
    with open("config_aicrowd.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(f"{config['result_path']}/INFO.yaml") as f:
        INFO = yaml.load(f, Loader=yaml.FullLoader)
        INFO_items = list(INFO.items())
        INFO_items.sort(key=lambda x: x[1]['order'])

    start_timer = time.time()
    postural_features(config, INFO_items)
    print(f"::: Postural Measurements ::: Computation Time: {time.time()-start_timer}")

    start_timer = time.time()
    kinematic_features(config, INFO_items)
    print(f"::: Kinematic Measurements ::: Computation Time: {time.time()-start_timer}")

def postural_features(config, INFO_items):
    # Standardizing Model (Robust Scaling - only using good frames)
    angle_scaler = StandardScaler()
    limb_scaler = StandardScaler()
    
    for key, file in tqdm(INFO_items):
        save_path = file['directory']
        bp = np.load(f"{save_path}/rotated_bodypoints.npy")
        num_fr, _, _ = bp.shape
        good_fr, bad_fr, disregard_fr = locate_bad_fr(config, bp)

        # Compute Joint Angle
        num_angles = len(config['angles'])
        angles = np.zeros((num_fr, num_angles))
        angles = angle_calc(bp[:,:,0:2], config['angles'])
        angle_scaler.partial_fit(angles[good_fr,:]) # collect normalization info
        np.save(f"{save_path}/angles.npy", angles)
        
        # Compute Limb Length
        limbs = np.zeros((num_fr, len(config['limbs'])))
        for i, limb_pts in enumerate(config['limbs']):
            limb_i = bp[:,limb_pts,0:2]
            limbs[:,i] = np.sqrt((limb_i[:,0,0]-limb_i[:,1,0])**2 + (limb_i[:,0,1]-limb_i[:,1,1])**2)
        limb_scaler.partial_fit(limbs[good_fr,:]) # collect normalization info
        np.save(f"{save_path}/limbs.npy", limbs)
    
    # Save Standardization Model
    with open(f"{config['result_path']}/angle_scale_model.pickle", 'wb') as file:
        pickle.dump(angle_scaler, file)
    with open(f"{config['result_path']}/limb_scale_model.pickle", 'wb') as file:
        pickle.dump(limb_scaler, file)

def kinematic_features(config, INFO_items):
    with open (f"{config['result_path']}/angle_scale_model.pickle", 'rb') as file:
        angle_scaler = pickle.load(file)
    with open (f"{config['result_path']}/limb_scale_model.pickle", 'rb') as file:
        limb_scaler = pickle.load(file)

    # Morlet on Angle and Limb Features
    for key, file in tqdm(INFO_items):
        save_path = file['directory']
        angles = np.load(f"{save_path}/angles.npy")
        limbs = np.load(f"{save_path}/limbs.npy")

        # Joint Angle
        stand_angles = angle_scaler.transform(angles) # standarize
        angle_power = morlet(config, stand_angles)
        np.save(f"{save_path}/angle_power.npy", angle_power)

        # Limb Length
        stand_limbs = limb_scaler.transform(limbs) # standarize
        limb_power = morlet(config, stand_limbs)
        np.save(f"{save_path}/limb_power.npy", limb_power)

if __name__ == "__main__":
    main()












