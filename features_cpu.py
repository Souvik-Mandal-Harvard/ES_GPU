import random
import numpy as np
import pandas as pd
from glob2 import glob
from tqdm import tqdm

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import umap
from scipy.signal import morlet2, cwt, ricker
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder

# constants
fps = 50

KEYS = [{"a": 2, "b": 9, "c": 8},
        {"a": 2, "b": 22, "c": 21}]


length = []
count = 0

# parameters
interp_samp = 1
interp_fps = fps*interp_samp
w = 5

bp_list, angles_list, power_list = [], [], []
for path in tqdm(glob("input_data/clean_data/**/*.h5")):
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
    print(power.shape)
    power_list.append(power)
    
    if count == 5:
        break
    else:
        count += 1

tot_bp = np.concatenate(bp_list, axis=0)
tot_angles = np.concatenate(angles_list, axis=0)
tot_pwr = np.concatenate(power_list, axis=2)




# reshape power data
num_angles, num_freq, num_fr = tot_pwr.shape
power_mod = tot_pwr.reshape((num_angles*num_freq, num_fr)).T
# umap
embed = umap.UMAP(n_neighbors=100, n_epochs=100, min_dist=0.1,
                  init="spectral", learning_rate=1.5).fit_transform(power_mod)


fig, ax = plt.subplots(figsize=(5, 5))
ax = sns.kdeplot(embed[:,0], embed[:,1], 
                 shade=True, shade_lowest=True, gridsize=80, levels=30, cmap='jet',cbar=False)
ax.set(xlabel='Component 1', ylabel='Component 2', title="UMAP Training Density Plot")
# ax.grid(True)
plt.show()



max_cluster = 40
cluster_list = range(1, max_cluster)
bic_list = []
for i in tqdm(cluster_list):
    gmm = GaussianMixture(n_components=i)
    gmm.fit(embed)
    bic_list.append(gmm.bic(embed))
    
min_idx = np.argmin(bic_list)
# plot
plt.plot(cluster_list, bic_list, marker="o", c='k', markersize=5)
plt.scatter(min_idx+1, bic_list[min_idx], s=120, edgecolors='r', facecolors='none')
plt.xlabel("Number of Clusters"); plt.ylabel("BIC"); plt.title("Bayesian information criterion")
plt.show()


# Gaussian Mixture Model
gmm = GaussianMixture(n_components=11)
gmm_label = gmm.fit_predict(embed)


num_clusters = np.max(gmm_label) + 1
# compute probability
prob = np.max(gmm.predict_proba(embed), axis=1)

# choose color palette
color_palette = sns.color_palette('hls', num_clusters)
cluster_colors = [color_palette[x] for x in gmm_label]
cluster_member_colors = np.array([sns.desaturate(x, p) for x, p in zip(cluster_colors, prob)])

# create figures
plt.subplots(figsize=(5, 5))
for i in range(num_clusters):
    idx = (gmm_label==i)
    plt.scatter(embed[idx,0], embed[idx,1], 
                c=cluster_member_colors[idx], 
                alpha=1, s=7, label=i)
    plt.annotate(i, gmm.means_[i], fontsize=14, fontweight='bold')
    
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)
plt.xlabel("Component 1"); plt.xlabel("Component 2"); plt.title("Behavioral Manifold")
plt.show()