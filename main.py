import os, yaml, matplotlib, random, pickle, time, cudf, cuml
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.signal import morlet2, cwt
from sklearn.mixture import GaussianMixture

from helper import _rotational, angle_calc

start_timer = time.time()

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Initialize
INFO = {}

bad_frames_ref = {}
frame_start = 0
bp_unrot_list, bp_list, scale_list, angles_list, rotate_list, power_list = [], [], [], [], [], []

# Bodypoints Used for Analysis
bp_analyze = []
for angle_points in config["angles"]:
    bp_analyze.extend(angle_points.values())
bp_analyze.append(config["bp_rotate"])
bp_analyze.extend(config["bp_scale"])
bp_analyze.append(config["bp_center"])
bp_analyze = np.unique(bp_analyze)


start_fr = 0
for path_i, path in tqdm(enumerate(glob(f"{config['input_data_path']}/**/*.h5"))):
    ### Directory Path
    # folder_name = os.path.dirname(path).split("/")[-1]
    folder_name = os.path.basename(path).split("DLC")[0]
    print(folder_name)
    dir_path = f"{config['result_path']}/{folder_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    ### Set INFO
    INFO[folder_name] = {}
    INFO[folder_name]['directory'] = dir_path
    INFO[folder_name]['order'] = path_i
    
    ### Import
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
        np.save(f"{dir_path}/bodypoints.npy", DLC_data)

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
        np.save(f"{dir_path}/scaled_bodypoints.npy", DLC_data)

    ### Rotate
    DLC_data[:,:,0:2], body_angles = _rotational(data=DLC_data[:,:,0:2], axis_bp=config['bp_rotate'])
    DLC_data[:,:,2] = likelihood
    if config['save_rotated_bodypoints']:
        np.save(f"{dir_path}/rotated_bodypoints.npy", DLC_data)
    if config['save_body_orientation_angles']:
        np.save(f"{dir_path}/body_orientation_angles.npy", body_angles)

    ### Angles
    num_angles = len(config['angles'])
    angles = np.zeros((num_fr,num_angles,2))
    angles[:,:,0] = angle_calc(DLC_data[:,:,0:2], config['angles'])
    # measure angle likelihood
    for ang_idx, angle_key in enumerate(config['angles']):
        angles[:,ang_idx,1] = likelihood[:,angle_key['a']] * likelihood[:,angle_key['b']] * likelihood[:,angle_key['c']]
    if config['save_angles']:
        np.save(f"{dir_path}/angles.npy", angles)
    angles_list.append(angles)
    
    ### Record Files
    INFO[folder_name]['number_frames'] = num_fr
    INFO[folder_name]['global_start_fr'] = start_fr
    INFO[folder_name]['global_stop_fr'] = start_fr+num_fr
    start_fr += num_fr

tot_angles = np.concatenate(angles_list, axis=0)
for (folder_name, val), angles in tqdm(zip(INFO.items(), angles_list)):
    dir_path = f"{config['result_path']}/{folder_name}"

    # Normalize Angles
    angles[:,:,0] -= np.mean(tot_angles[:,:,0], axis=0)
    
    # Morlet Wavelet
    num_fr, num_ang, _ = angles.shape
    power = np.zeros((num_ang, config['f_bin']+1, num_fr))
    max_freq, min_freq = config['fps']/2, 1 # Nyquist Frequency
    freq = max_freq*2**(-1*np.log2(max_freq/min_freq)*
        (np.arange(config['f_bin'],0,-1)-1)/(config['f_bin']-1))
    widths = config['w']*config['fps'] / (2*freq*np.pi)
    INFO[folder_name]["frequencies"] = freq.tolist()
    
    # Normalization Factor
    s = (config['w'] + np.sqrt(2+config['w']**2))/(4*np.pi*freq)
    C = np.pi**(-0.25)*np.exp(0.25*(config['w']-np.sqrt(config['w']**2+2))**2)/np.sqrt(2*s)
    
    for i in range(num_ang):
        cwtm = cwt(angles[:,i,0], morlet2, widths, dtype=None, w=config['w'])
        # power[i] = np.abs(cwtm)**2
        power[i,:-1,:] = (np.abs(cwtm/np.expand_dims(np.sqrt(s),1)))/np.expand_dims(C, axis=(0,2))
        power[i,-1,:] = angles[:,i,1]
    if config['save_powers']:
        np.save(f"{dir_path}/power.npy", power)
    power_list.append(power)
tot_pwr = np.concatenate(power_list, axis=2)

# TEST
(good_ang_idx, good_fr_idx) = np.where(tot_pwr[:,-1,:] == 1)
unique_good_fr = np.unique(good_fr_idx)
###########

# Dimensional Reduction
tot_pwr = tot_pwr[:,:,unique_good_fr]
num_ang, num_freq, num_fr = tot_pwr.shape
power_mod = tot_pwr.reshape((num_ang*num_freq, num_fr)).T
embed = np.zeros((num_fr, config['n_components']+1))
df = cudf.DataFrame(power_mod)

cu_embed = cuml.UMAP(n_components=config['n_components'], n_neighbors=config['n_neighbors'], n_epochs=config['n_epochs'], 
                min_dist=config['min_dist'], negative_sample_rate=config['negative_sample_rate'],
                init=config['init'], repulsion_strength=config['repulsion_strength']).fit_transform(df)
embed[:,0:config['n_components']] = cu_embed.to_pandas().to_numpy()
embed[:,config['n_components']] = np.prod(tot_pwr[:,-1,:], axis=0)

# TEST
fig_base, ax_base = plt.subplots(figsize=(10,10))
ax_base.scatter(embed[:,0], embed[:,1], alpha=0.002, s=1)  
ax_base.set(xlabel='Component 1', ylabel='Component 2', title="Behavioral Manifold")
plt.savefig("embedding")
###########

#cu_score = cuml.metrics.trustworthiness(df, embed)
#print(f"UMAP Trustworthiness: {cu_score}")

# Save data
fr_start = 0
for folder_name, val in INFO.items():
    dir_path = f"{config['result_path']}/{folder_name}"
    if config['save_embeddings']:
        np.save(f"{dir_path}/embeddings.npy", embed[fr_start:fr_start+val['number_frames'],:])
    fr_start += val['number_frames']

with open(f"{config['result_path']}/INFO.yaml", 'w') as file:
    documents = yaml.dump(INFO, file)

print(INFO)
print(embed.shape)
print(f"Computation Time: {time.time()-start_timer}")







