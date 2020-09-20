import os, yaml, matplotlib, random, pickle, cudf, cuml, time
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.signal import morlet2, cwt
from sklearn.mixture import GaussianMixture

from helper import _rotational, angle_calc

start_timer = time.time()
# Configuration
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Initialize
files_ref, bad_frames_ref = {}, {}
frame_start = 0
bp_list, scale_list, angles_list, power_list = [], [], [], []

# Bodypoints Used for Analysis
bp_analyze = []
for angle_points in config["angles"]:
    bp_analyze.extend(angle_points.values())
bp_analyze.append(config["bp_rotate"])
bp_analyze.extend(config["bp_scale"])
bp_analyze.append(config["bp_center"])
bp_analyze = np.unique(bp_analyze)

for path in tqdm(glob(f"{config['input_data_path']}/**/*.h5")):
    # Import
    store = pd.HDFStore(path, mode='a')
    df = store['df_with_missing']
    x_data = df.xs('x', level="coords", axis=1).to_numpy()
    y_data = df.xs('y', level="coords", axis=1).to_numpy()
    likelihood = df.xs('likelihood', level="coords", axis=1).to_numpy()
    store.close()
    
    # Check Likelihood
    below_thresh = np.where(likelihood[:, bp_analyze] < .2)
    fr_bad = np.unique(below_thresh[0])
    bad_frames_ref[path] = fr_bad

    # Center
    x_center = x_data[:,config['bp_center']]
    y_center = y_data[:,config['bp_center']]
    x_data -= x_center[:,np.newaxis]
    y_data -= y_center[:,np.newaxis]
        
    # Format
    DLC_data = np.concatenate((
        np.expand_dims(x_data, axis=-1), 
        np.expand_dims(y_data, axis=-1)), axis=-1)
    num_fr,_,_ = DLC_data.shape
    
    # Scale
    x_d = DLC_data[:,config['bp_scale'][0],0] - DLC_data[:,config['bp_scale'][1],0]
    y_d = DLC_data[:,config['bp_scale'][0],1] - DLC_data[:,config['bp_scale'][1],1]
    dist = np.sqrt(x_d**2+y_d**2)
    norm = np.median(dist)
    scale_list.append(norm)
    DLC_data /= norm

    # Rotate
    ROT_data, body_angle = _rotational(data=DLC_data, axis_bp=config['bp_rotate'])
    bp_list.append(ROT_data)

    # Angles
    angles = angle_calc(ROT_data, config['angles'])
    angles -= np.mean(angles, axis=0)
    angles_list.append(angles)

    # Record Files
    files_ref[path] = (frame_start, frame_start+num_fr)
    frame_start += num_fr

# Combine Bodypoints and Angles Data
tot_bp = np.concatenate(bp_list, axis=0)
tot_angles = np.concatenate(angles_list, axis=0)  

for angles in tqdm(angles_list):
    # Normalize Angles
    angles -= np.mean(tot_angles, axis=0)
    
    # Morlet Wavelet
    num_fr, num_ang = angles.shape
    power = np.zeros((num_ang, config['f_bin'], num_fr))
    max_freq, min_freq = config['fps']/2, 1 # Nyquist Frequency
    freq = max_freq*2**(-1*np.log2(max_freq/min_freq)*
        (np.arange(config['f_bin'],0,-1)-1)/(config['f_bin']-1))
    widths = config['w']*config['fps'] / (2*freq*np.pi)
    
    # Normalization Factor
    s = (config['w'] + np.sqrt(2+config['w']**2))/(4*np.pi*freq)
    C = np.pi**(-0.25)*np.exp(0.25*(config['w']-np.sqrt(config['w']**2+2))**2)/np.sqrt(2*s)

    for i in range(num_ang):
        cwtm = cwt(angles[:,i], morlet2, widths, dtype=None, w=config['w'])
        # power[i] = np.abs(cwtm)**2
        power[i] = (np.abs(cwtm/np.expand_dims(np.sqrt(s),1)))/np.expand_dims(C, axis=(0,2))
    power_list.append(power)

tot_pwr = np.concatenate(power_list, axis=2)
num_ang, num_freq, num_fr = tot_pwr.shape

# Take Out Bad Frames
tot_fr_bad = []
for path, fr_range in files_ref.items():
    tot_fr_bad.extend(bad_frames_ref[path]+fr_range[0])
tot_fr_good = np.delete(np.arange(num_fr), tot_fr_bad)
good_tot_pwr = np.delete(tot_pwr, tot_fr_bad, axis=2)

# Dimensional Reduction
num_angles, num_freq, num_good_fr = good_tot_pwr.shape
power_mod = good_tot_pwr.reshape((num_angles*num_freq, num_good_fr)).T
df = cudf.DataFrame(power_mod)
embed = cuml.UMAP(n_neighbors=config['n_neighbors'], n_epochs=config['n_epochs'], 
                min_dist=config['min_dist'], negative_sample_rate=config['negative_sample_rate'],
                init=config['init'], repulsion_strength=config['repulsion_strength']).fit_transform(df)
np_embed = embed.to_pandas().to_numpy()

# Append Undefined Frames
full_embed = np.empty((num_fr, 2))
full_embed[:] = np.nan
full_embed[tot_fr_good, :] = np_embed

#cu_score = cuml.metrics.trustworthiness(df, embed)
#print(f"UMAP Trustworthiness: {cu_score}")

# Clustering (HDBSCAN, GMM)
# TODO: HDBSCAN
# TODO: GMM

# Ethogram
# TODO: Ethogram

# Save data
if config['save_file_refs']:
    pickle_out = open(f"{config['result_path']}/files_ref.pickle","wb")
    pickle.dump(files_ref, pickle_out)
    pickle_out.close()
if config['save_bad_frames_ref']:
    pickle_out = open(f"{config['result_path']}/bad_frames_ref.pickle","wb")
    pickle.dump(bad_frames_ref, pickle_out)
    pickle_out.close()
if config['save_bp_scales']:
    np.save(f"{config['result_path']}/scales.npy", scale_list)
if config['save_freqs']:
    np.save(f"{config['result_path']}/freq.npy", freq)

for path, fr_range in files_ref.items():
    # set up directory path
    folder = os.path.dirname(path).split("/")[-1]
    dir_path = f"{config['result_path']}/{folder}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        # save requested files
        if config['save_trans_bodypoints']:
            np.save(f"{dir_path}/bodypoints.npy", tot_bp[fr_range[0]:fr_range[1],:,:])
        if config['save_angles']:
            np.save(f"{dir_path}/angles.npy", tot_angles[fr_range[0]:fr_range[1],:])
        if config['save_powers']:
            np.save(f"{dir_path}/power.npy", tot_pwr[:,:,fr_range[0]:fr_range[1]])
        if config['save_embeddings']:
            np.save(f"{dir_path}/embeddings.npy", full_embed[fr_range[0]:fr_range[1],:])

print(good_tot_pwr.shape)
print(np_embed.shape)
print(full_embed.shape)
print(f"Computation Time: {time.time()-start_timer}")







