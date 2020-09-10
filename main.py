import random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy.signal import morlet2, cwt, ricker
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder

from helper import _rotational, _rotate, angle_calc

import cudf
import cuml

# constants
fps = 50

KEYS = [{"a": 2, "b": 9, "c": 8},
        {"a": 2, "b": 22, "c": 21}]


length = []

# parameters
interp_samp = 1
interp_fps = fps*interp_samp
w = 5

bp_list, angles_list, power_list = [], [], []
for path in tqdm(glob("data/clean_data/**/*.h5")):
    # IMPORT DATA
    # save data to main dataframe
    store = pd.HDFStore(path, mode='a')
    df = store['df_with_missing']
    x_data = df.xs('x', level="coords", axis=1).to_numpy()
    y_data = df.xs('y', level="coords", axis=1).to_numpy()
    store.close()
    
    # center bp
    x_center = x_data[:,2]
    y_center = y_data[:,2]
    x_data -= x_center[:,np.newaxis]
    y_data -= y_center[:,np.newaxis]
        
    # format data
    DLC_data = np.concatenate((
        np.expand_dims(x_data, axis=-1), 
        np.expand_dims(y_data, axis=-1)), axis=-1)
    
    # scale body points
    x_d = DLC_data[:,1,0] - DLC_data[:,2,0]
    y_d = DLC_data[:,1,1] - DLC_data[:,2,1]
    dist = np.sqrt(x_d**2+y_d**2)
    norm = np.median(dist)
    length.append(norm)
    DLC_data /= norm
    
    # rotate bp
    ROT_data, body_angle = _rotational(data=DLC_data, axis_bp=1)
#     plt.subplots(figsize=(4,4))
#     plt.scatter(ROT_data[:,:,0], ROT_data[:,:,1], alpha=0.4, s=7)
#     plt.show()
    
    # interpolate data
    num_fr, _,_ = ROT_data.shape
    time = np.arange(0, num_fr)
    new_time = np.arange(0, num_fr-1, 1/interp_samp)
    interp_func = interp1d(time, ROT_data, axis=0, kind='cubic')
    new_ROT_data = interp_func(new_time)
    bp_list.append(new_ROT_data)
    
    # compute angles
    angles = angle_calc(new_ROT_data, KEYS)
    angles -= np.mean(angles, axis=0)
    angles_list.append(angles)
    
    # morlet wavelet
    num_interp_fr, num_ang = angles.shape
    num_freq = 20
#     freq = np.linspace(1, fps/2, num_freq)
    max_freq, min_freq = fps/2, 1 # Nyquist Frequency
    freq = max_freq*2**(-1*np.log2(max_freq/min_freq)*(np.arange(num_freq,0,-1)-1)/(num_freq-1))
    widths = w*interp_fps / (2*freq*np.pi)
    power = np.zeros((num_ang, num_freq, num_interp_fr))
    for i in range(num_ang):
        cwtm = cwt(angles[:,i], morlet2, widths, dtype=None, w=w)
        power[i] = np.abs(cwtm)**2
    power_list.append(power)
    

tot_bp = np.concatenate(bp_list, axis=0)
tot_angles = np.concatenate(angles_list, axis=0)
tot_pwr = np.concatenate(power_list, axis=2)

# reshape power data
num_angles, num_freq, num_fr = tot_pwr.shape
power_mod = tot_pwr.reshape((num_angles*num_freq, num_fr)).T

df = cudf.DataFrame(power_mod)
print(df)

embed = cuml.UMAP(n_neighbors=100, n_epochs=500, min_dist=0.1,
                  init="spectral", learning_rate=1.5).fit_transform(df)


result_path = "results/test"
np.save(f"{result_path}/embeddings.npy", embed.head().to_pandas().to_numpy())
np.save(f"{result_path}/bodypoints.npy", tot_bp)
np.save(f"{result_path}/angles.npy", tot_angles)
np.save(f"{result_path}/power.npy", tot_pwr)
