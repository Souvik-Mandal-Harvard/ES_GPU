import yaml, matplotlib, random, pickle, cudf, cuml
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.signal import morlet2, cwt
from sklearn.mixture import GaussianMixture

from helper import _rotational, angle_calc

# Configuration
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Initialize
files_ref = {}
frame_start = 0
bp_list, scale_list, angles_list, power_list = [], [], [], []

for path in tqdm(glob(f"{config['input_data_path']}/**/*.h5")):
    # Import
    store = pd.HDFStore(path, mode='a')
    df = store['df_with_missing']
    x_data = df.xs('x', level="coords", axis=1).to_numpy()
    y_data = df.xs('y', level="coords", axis=1).to_numpy()
    store.close()
    
    # Center
    x_center = x_data[:,config['bp_center']]
    y_center = y_data[:,config['bp_center']]
    x_data -= x_center[:,np.newaxis]
    y_data -= y_center[:,np.newaxis]
        
    # Format
    DLC_data = np.concatenate((
        np.expand_dims(x_data, axis=-1), 
        np.expand_dims(y_data, axis=-1)), axis=-1)
    
    # Scale
    x_d = DLC_data[:,config['bp_scale'][0],0] - DLC_data[:,config['bp_scale'][1],0]
    y_d = DLC_data[:,config['bp_scale'][0],1] - DLC_data[:,config['bp_scale'][1],1]
    dist = np.sqrt(x_d**2+y_d**2)
    norm = np.median(dist)
    scale_list.append(norm)
    DLC_data /= norm
    
    # Rotate
    ROT_data, body_angle = _rotational(data=DLC_data, axis_bp=config['bp_rotate'])
    
    # Angles
    angles = angle_calc(ROT_data, config['angles'])
    angles -= np.mean(angles, axis=0)
    angles_list.append(angles)
    
    # Morlet Wavelet
    num_fr, num_ang = angles.shape
    power = np.zeros((num_ang, config['f_bin'], num_fr))
    max_freq, min_freq = config['fps']/2, 1 # Nyquist Frequency
    freq = max_freq*2**(-1*np.log2(max_freq/min_freq)*
        (np.arange(config['f_bin'],0,-1)-1)/(config['f_bin']-1))
    widths = config['w']*config['fps'] / (2*freq*np.pi)
    for i in range(num_ang):
        cwtm = cwt(angles[:,i], morlet2, widths, dtype=None, w=config['w'])
        power[i] = np.abs(cwtm)**2
    power_list.append(power)
    
    # Record Files
    files_ref[path] = (frame_start, frame_start+num_fr)
    frame_start += num_fr

tot_bp = np.concatenate(bp_list, axis=0)
tot_angles = np.concatenate(angles_list, axis=0)
tot_pwr = np.concatenate(power_list, axis=2)

# Dimensional Reduction
num_angles, num_freq, num_fr = tot_pwr.shape
power_mod = tot_pwr.reshape((num_angles*num_freq, num_fr)).T
df = cudf.DataFrame(power_mod)
embed = cuml.UMAP(n_neighbors=config['n_neighbors'], n_epochs=config['n_epochs'], 
                min_dist=config['min_dist'], negative_sample_rate=config['negative_sample_rate'],
                init=config['init'], repulsion_strength=config['repulsion_strength']).fit_transform(df)
cu_score = cuml.metrics.trustworthiness(df, embed)

print(f"UMAP Trustworthiness: {cu_score}")

# Figures
# TODO
#     plt.subplots(figsize=(4,4))
#     plt.scatter(ROT_data[:,:,0], ROT_data[:,:,1], alpha=0.4, s=7)
#     plt.show()

# Save data
if config['save_file_refs']:
    pickle_out = open(f"{config['result_path']}/files_ref.pickle","wb")
    pickle.dump(FILES, pickle_out)
    pickle_out.close()
if config['save_bp_scales']:
    np.save(f"{config['result_path']}/scales.npy", scale_list)
if config['save_trans_bodypoints']:
    np.save(f"{config['result_path']}/bodypoints.npy", tot_bp)
if config['save_angles']:
    np.save(f"{config['result_path']}/angles.npy", tot_angles)
if config['save_powers']:
    np.save(f"{config['result_path']}/power.npy", tot_pwr)
if config['save_freqs']:
    np.save(f"{config['result_path']}/freq.npy", freq)
if config['save_embeddings']:
    np.save(f"{config['result_path']}/embeddings.npy", embed.to_pandas().to_numpy())









