import sys, time
import yaml, pickle
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Import Helper Function
from helper import locate_bad_fr, angle_calc, morlet


def main():
    # grab arguments
    config_name = sys.argv[1]

    with open(config_name) as f:
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
    num_angles = len(config['angles'])
    num_limbs = len(config['limbs'])
    # Standardizing Model (Robust Scaling - only using good frames)
    angle_scaler = MinMaxScaler(feature_range=(0,1))
    limb_scaler = MinMaxScaler(feature_range=(0,1))
    angle_sum, limb_sum = np.zeros(num_angles), np.zeros(num_limbs)
    tot_fr_number = 0

    for key, file in tqdm(INFO_items):
        save_path = file['directory']
        bp = np.load(f"{save_path}/rotated_bodypoints.npy")
        num_fr, _, _ = bp.shape
        good_fr, bad_fr, disregard_fr = locate_bad_fr(config, bp)
        tot_fr_number += len(good_fr)

        # Compute Joint Angle
        angles = np.zeros((num_fr, num_angles))
        angles = angle_calc(bp[:,:,0:2], config['angles'])
        if len(good_fr) != 0:
            angle_scaler.partial_fit(angles[good_fr,:]) # collect normalization info
        angle_sum += np.sum(angles[good_fr,:], axis=0)
        np.save(f"{save_path}/angles.npy", angles)
        
        # Compute Limb Length
        limbs = np.zeros((num_fr, num_limbs))
        for i, limb_pts in enumerate(config['limbs']):
            limb_i = bp[:,limb_pts,0:2]
            limbs[:,i] = np.sqrt((limb_i[:,0,0]-limb_i[:,1,0])**2 + (limb_i[:,0,1]-limb_i[:,1,1])**2)

        if len(good_fr) != 0:
            limb_scaler.partial_fit(limbs[good_fr,:]) # collect normalization info
        limb_sum += np.sum(limbs[good_fr,:], axis=0)
        np.save(f"{save_path}/limbs.npy", limbs)

    angle_scaler.means_ = angle_sum/tot_fr_number
    limb_scaler.means_ = limb_sum/tot_fr_number
    print(limb_sum)
    print(limb_scaler.means_)

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
    angle_scaler.scaled_means_ = angle_scaler.scale_ * angle_scaler.means_
    limb_scaler.scaled_means_ = limb_scaler.scale_ * limb_scaler.means_

    # Morlet on Angle and Limb Features
    for key, file in tqdm(INFO_items):
        save_path = file['directory']
        angles = np.load(f"{save_path}/angles.npy")
        limbs = np.load(f"{save_path}/limbs.npy")

        # Joint Angle
        stand_angles = angle_scaler.transform(angles) - angle_scaler.scaled_means_ # scale
        angle_power = morlet(config, stand_angles)
        np.save(f"{save_path}/angle_power.npy", angle_power)

        # Limb Length
        print("TESTINGINGNGNGNGNG")
        print(np.where(np.isnan(limbs))[0].shape)
        stand_limbs = limb_scaler.transform(limbs) - limb_scaler.scaled_means_  # scale
        print(np.where(np.isnan(stand_limbs))[0].shape)
        print(limb_scaler.scaled_means_ )
        limb_power = morlet(config, limbs)
        print(np.where(np.isnan(limb_power))[0].shape)
        np.save(f"{save_path}/limb_power.npy", limb_power)

    # Save Standardization Model
    with open(f"{config['result_path']}/angle_scale_model.pickle", 'wb') as file:
        pickle.dump(angle_scaler, file)
    with open(f"{config['result_path']}/limb_scale_model.pickle", 'wb') as file:
        pickle.dump(limb_scaler, file)

if __name__ == "__main__":
    main()














