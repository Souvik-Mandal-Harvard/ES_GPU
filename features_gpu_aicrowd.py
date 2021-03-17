import time
import yaml
import numpy as np
from tqdm import tqdm
import collections

# Import Visualization
import matplotlib
import matplotlib.pyplot as plt

# Import Helper Function
from helper import angle_calc, cuml_umap, cuml_pca, morlet

start_timer = time.time()

with open("config_aicrowd.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
with open(f"{config['result_path']}/INFO.yaml") as f:
    INFO = yaml.load(f, Loader=yaml.FullLoader)
    INFO_items = list(INFO.items())
    INFO_items.sort(key=lambda x: x[1]['order'])

def plot_embedding(embed, title="test", fname="test"):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(embed[:,0], embed[:,1], s=1, alpha=0.01)
    ax.set(title=title)
    plt.savefig(f"figures/{fname}.png")
    return

angles_list, power_list = [], []
tot_bp, tot_angle, tot_limb = [], [], []

# Compute Postural Features
for key, file in tqdm(INFO_items):
    save_path = file['directory']
    bp = np.load(f"{save_path}/rotated_bodypoints.npy")
    num_fr, num_bp, _ = bp.shape
    likelihood = bp[:,:,2]

    ### Locate Bad Frames ###
    # check if below likelihood threshold
    (below_thresh_fr, below_thresh_marker) = np.where(likelihood < config['likelihood_thresh'])
    cnt = collections.Counter(below_thresh_fr)
    cnt_array = np.array(list(cnt.items()))
    # check if above marker threshold
    try:
        bad_fr_idx = np.where(cnt_array[:,1] > config['marker_thresh'])[0]
        bad_fr = cnt_array[bad_fr_idx,0]
        # append pads
        padded_fr = np.array([ list(range(fr-config['bad_fr_pad'], fr+config['bad_fr_pad']+1)) for fr in bad_fr])
        disregard_fr = np.unique(padded_fr.flatten())
        disregard_fr = disregard_fr[(disregard_fr >= 0) & (disregard_fr < num_fr)]
        good_fr_idx = np.array([True]*num_fr)
        good_fr_idx[disregard_fr] = False
        good_fr = np.where(good_fr_idx==True)[0]
    except:
        bad_fr = np.array([])
        disregard_fr = np.array([])
        good_fr = np.arange(num_fr)

    # TODO: modify proportion of good fr
    file['good_fr'] = good_fr.tolist()
    file['bad_fr'] = bad_fr.tolist()
    file['disregard_fr'] = disregard_fr.tolist()

    ### Postural Features ###
    # Marker Position
    if config['include_marker_postural'] or config['include_all_postural'] or config['include_all_features']:
        # remove bad frames
        bp_markers = bp[:,config['markers'],:]
        tot_bp.append(bp_markers[good_fr,:])

    # Joint Angle
    if config['include_angle_postural'] or config['include_all_postural'] or config['include_all_features']:
        # compute angle
        num_angles = len(config['angles'])
        angles = np.zeros((num_fr, num_angles, 2))
        angles[:,:,0] = angle_calc(bp[:,:,0:2], config['angles'])
        # measure angle likelihood
        for ang_idx, angle_key in enumerate(config['angles']):
            angles[:,ang_idx,1] = likelihood[:,angle_key['a']] * likelihood[:,angle_key['b']] * likelihood[:,angle_key['c']]
        if config['save_angles']:
            np.save(f"{save_path}/angles.npy", angles)
        tot_angle.append(angles[good_fr,:,:])

    # Limb Length
    if config['include_limb_postural'] or config['include_all_postural'] or config['include_all_features']:
        limbs = np.zeros((num_fr, len(config['limbs'])))
        for i, limb_pts in enumerate(config['limbs']):
            limb_i = bp[:,limb_pts,0:2]
            limbs[:,i] = np.sqrt((limb_i[:,0,0]-limb_i[:,1,0])**2 + (limb_i[:,0,1]-limb_i[:,1,1])**2)
        if config['save_limbs']:
            np.save(f"{save_path}/limbs.npy", limbs)
        tot_limb.append(limbs[good_fr,:])

# Concat Data
if config['include_marker_postural'] or config['include_all_postural'] or config['include_all_features']:
    tot_bp = np.concatenate(tot_bp)
    print(f"tot_bp shape: {tot_bp.shape}")
if config['include_angle_postural'] or config['include_all_postural'] or config['include_all_features']:
    tot_angle = np.concatenate(tot_angle)
    print(f"tot_angle shape: {tot_angle.shape}")
if config['include_limb_postural'] or config['include_all_postural'] or config['include_all_features']:
    tot_limb = np.concatenate(tot_limb)
    print(f"tot_limb shape: {tot_limb.shape}")

tot_marker_pwr, tot_angle_pwr, tot_limb_pwr = [], [], []

### Kinematic Features
for key, file in tqdm(INFO_items):
    save_path = file['directory']
    bp = np.load(f"{save_path}/rotated_bodypoints.npy")
    num_fr, num_bp, _ = bp.shape
    good_fr = file['good_fr']

    ### Kinematic Features ###
    # Marker Position
    if config['include_marker_kinematic'] or config['include_all_kinematic'] or config['include_all_features']:
        bp_markers = bp[:,config['markers'],:]
        bp_markers[:,:,0:2] -= np.mean(tot_bp[:,:,0:2], axis=0)

        num_fr, num_bp, num_bp_dim = bp_markers.shape
        tot_bp_mod = bp_markers[:,:,0:num_bp_dim-1].reshape(num_fr, num_bp*(num_bp_dim-1))

        marker_power = morlet(config, tot_bp_mod)
        if config['save_powers']:
            np.save(f"{save_path}/marker_power.npy", marker_power)
        tot_marker_pwr.append(marker_power[good_fr,:,:])

    # Joint Angle
    if config['include_angle_kinematic'] or config['include_all_kinematic'] or config['include_all_features']:
        angles = angle_calc(bp[:,:,0:2], config['angles'])
        angles -= np.mean(tot_angle[:,:,0], axis=0)

        angle_power = morlet(config, angles)
        if config['save_powers']:
            np.save(f"{save_path}/angle_power.npy", angle_power)
        tot_angle_pwr.append(angle_power[good_fr,:,:])

    # Limb Length
    if config['include_limb_kinematic'] or config['include_all_kinematic'] or config['include_all_features']:
        limbs = np.zeros((num_fr, len(config['limbs'])))
        for i, limb_pts in enumerate(config['limbs']):
            limb_i = bp[:,limb_pts,0:2]
            limbs[:,i] = np.sqrt((limb_i[:,0,0]-limb_i[:,1,0])**2 + (limb_i[:,0,1]-limb_i[:,1,1])**2)
        
        limbs -= np.mean(tot_limb, axis=0)
        limb_power = morlet(config, limbs)
        if config['save_powers']:
            np.save(f"{save_path}/limb_power.npy", limb_power)
        tot_limb_pwr.append(limb_power[good_fr,:,:])

# Concat Data
if config['include_marker_kinematic'] or config['include_all_kinematic'] or config['include_all_features']:
    tot_marker_pwr = np.concatenate(tot_marker_pwr)
    print(f"tot_marker_pwr shape: {tot_marker_pwr.shape}")
if config['include_angle_kinematic'] or config['include_all_kinematic'] or config['include_all_features']:
    tot_angle_pwr = np.concatenate(tot_angle_pwr)
    print(f"tot_angle_pwr shape: {tot_angle_pwr.shape}")
if config['include_limb_kinematic'] or config['include_all_kinematic'] or config['include_all_features']:
    tot_limb_pwr = np.concatenate(tot_limb_pwr) 
    print(f"tot_limb_pwr shape: {tot_limb_pwr.shape}")  

### Postural Features ###
start_timer = time.time()
# 1) Marker Position
if config['include_marker_postural']:
    print(f"::: Marker Position ::: START")
    num_fr, num_bp, num_bp_dim = tot_bp.shape
    tot_bp_mod = tot_bp[:,:,0:num_bp_dim-1].reshape(num_fr, num_bp*(num_bp_dim-1))
    marker_postural_embed = cuml_umap(config, tot_bp_mod)
    plot_embedding(marker_postural_embed, title="Marker Postural",fname="marker_postural_embedding")
    print(f"::: Marker Position ::: Computation Time: {time.time()-start_timer}")

# 2) Joint Angle
start_timer = time.time()
if config['include_angle_postural']:
    print(f"::: Joint Angle ::: START")
    angle_postural_embed = cuml_umap(config, tot_angle[:,:,0])
    plot_embedding(angle_postural_embed, title="Angle Postural", fname="angle_postural_embedding")
    print(f"::: Joint Angle ::: Computation Time: {time.time()-start_timer}")

# 3) Limb Length
start_timer = time.time()
if config['include_limb_postural']:
    print(f"::: Limb Length ::: START")
    limb_postural_embed = cuml_umap(config, tot_limb)
    plot_embedding(limb_postural_embed, title="Limb Postural", fname="limb_postural_embedding")
    print(f"::: Limb Length ::: Computation Time: {time.time()-start_timer}")

# 4) Marker Position, Joint Angle, & Limb Length (TODO Later)
if config['include_all_postural'] or config['include_all_features']:
    start_timer = time.time()
    print(f"::: All Postural ::: START Timer")
    
    num_fr, num_bp, num_bp_dim = tot_bp.shape
    tot_bp_mod = tot_bp[:,:,0:num_bp_dim-1].reshape(num_fr, num_bp*(num_bp_dim-1))
    
    # UMAP Embedding
    # feature = np.concatenate([bp_pca, bp_angle, bp_limb], axis=1)
    # postural_features = np.concatenate([tot_bp_mod, tot_angle[:,:,0], tot_limb], axis=1)
    postural_features = np.concatenate([tot_angle[:,:,0], tot_limb], axis=1)
    all_postural_embed = cuml_umap(config, postural_features)
    plot_embedding(all_postural_embed, title="All Postural", fname="all_postural_embedding")
    print(f"::: All Postural Features ::: Time Stamp: {time.time()-start_timer}")

### Kinematic Features ###
# 5) Marker Position Morlet
start_timer = time.time()
if config['include_marker_kinematic']:
    print(f"::: Marker (Kinematic) ::: START")
    num_fr, num_freq, num_feat = tot_marker_pwr.shape
    # PCA Embedding
    marker_kinematic_pca, _ = cuml_pca(config, 
        tot_marker_pwr.reshape(num_fr, num_freq*num_feat), 
        components=config['marker_kinematic_pca_components'])
    # UMAP Embedding
    marker_kinematic_embed = cuml_umap(config, marker_kinematic_pca)
    plot_embedding(marker_kinematic_embed, title="Marker Kinematic", fname="marker_kinematic_embedding")
    print(f"::: Marker (Kinematic) ::: Computation Time: {time.time()-start_timer}")

# 6) Joint Angle Morlet
start_timer = time.time()
if config['include_angle_kinematic']:
    print(f"::: Angle (Kinematic) ::: START")
    num_fr, num_freq, num_feat = tot_angle_pwr.shape
    # PCA Embedding
    angle_kinematic_pca, _ = cuml_pca(config, 
        tot_angle_pwr.reshape(num_fr, num_freq*num_feat),
        components=config['angle_kinematic_pca_components'])
    # UMAP Embedding
    angle_kinematic_embed = cuml_umap(config, angle_kinematic_pca)
    plot_embedding(angle_kinematic_embed, title="Angle Kinematic", fname="angle_kinematic_embedding")
    print(f"::: Angle (Kinematic) ::: Computation Time: {time.time()-start_timer}")

# 7) Limb Length Morlet
start_timer = time.time()
if config['include_limb_kinematic']:
    print(f"::: Limb (Kinematic) ::: START")
    num_fr, num_freq, num_feat = tot_limb_pwr.shape
    # PCA Embedding
    limb_kinematic_pca, _ = cuml_pca(config, 
        tot_limb_pwr.reshape(num_fr, num_freq*num_feat), 
        components=config['limb_kinematic_pca_components'])
    # UMAP Embedding
    limb_kinematic_embed = cuml_umap(config, limb_kinematic_pca)
    plot_embedding(limb_kinematic_embed, title="Limb Kinematic", fname="limb_kinematic_embedding")
    print(f"::: Limb (Kinematic) ::: Computation Time: {time.time()-start_timer}")

# 8) All Kinematic Features
start_timer = time.time()
if config['include_all_kinematic'] or config['include_all_features']:
    print(f"::: All Kinematic ::: START")
    num_fr, num_freq, num_marker_feat = tot_marker_pwr.shape
    num_fr, num_freq, num_angle_feat = tot_angle_pwr.shape
    num_fr, num_freq, num_limb_feat = tot_limb_pwr.shape

    # PCA Embedding
    # marker_kinematic_pca, _ = cuml_pca(config, 
    #     tot_marker_pwr.reshape(num_fr, num_freq*num_marker_feat), 
    #     components=config['marker_kinematic_pca_components'])
    
    # angle_kinematic_pca, _ = cuml_pca(config, 
    #     tot_angle_pwr.reshape(num_fr, num_freq*num_angle_feat), 
    #     components=config['angle_kinematic_pca_components'])
    # limb_kinematic_pca, _ = cuml_pca(config, 
    #     tot_limb_pwr.reshape(num_fr, num_freq*num_limb_feat), 
    #     components=config['limb_kinematic_pca_components'])
    
    # kinematic_features = np.concatenate([
    #     # marker_kinematic_pca,
    #     angle_kinematic_pca,
    #     limb_kinematic_pca
    # ], axis=1)

    kinematic_features = np.concatenate([
        # marker_kinematic_pca,
        tot_angle_pwr.reshape(num_fr, num_freq*num_angle_feat),
        tot_limb_pwr.reshape(num_fr, num_freq*num_limb_feat)
    ], axis=1)

    # UMAP Embedding
    all_kinematic_embed = cuml_umap(config, kinematic_features)
    plot_embedding(all_kinematic_embed, title="All Kinematic", fname="all_kinematic_embedding")
    print(f"::: All Kinematic ::: Computation Time: {time.time()-start_timer}")

# 9) Kinematic and Postural Features
start_timer = time.time()
if config['include_all_features']:
    print(f"::: All Features ::: START")
    all_features = np.concatenate([
        postural_features,
        kinematic_features
    ], axis=1)
    all_embed = cuml_umap(
        config, 
        all_features
    )
    plot_embedding(all_embed, title="All Feature", fname="all_feature_embedding")
    print(f"::: All Features ::: Computation Time: {time.time()-start_timer}")


### Save Embedding ###
# TODO: add bad features back into embed
start_timer = time.time()
print(f"::: Embedding Data Saving ::: START")
if config['save_embeddings']:
    start_fr = 0
    for key, file in INFO_items:
        num_fr = file['number_frames']
        good_fr = file['good_fr']
        num_good_fr = len(good_fr)
        # Postural
        if config['include_marker_postural']:
            embed = np.empty((num_fr, config['n_components']))
            embed[:] = np.nan
            embed[good_fr,:] = marker_postural_embed[start_fr:start_fr+num_good_fr]
            np.save(f"{file['directory']}/marker_postural_embeddings.npy", embed)
        
        if config['include_angle_postural']:
            embed = np.empty((num_fr, config['n_components']))
            embed[:] = np.nan
            embed[good_fr,:] = angle_postural_embed[start_fr:start_fr+num_good_fr]
            np.save(f"{file['directory']}/angle_postural_embeddings.npy", embed)
        
        if config['include_limb_postural']:
            embed = np.empty((num_fr, config['n_components']))
            embed[:] = np.nan
            embed[good_fr,:] = limb_postural_embed[start_fr:start_fr+num_good_fr]
            np.save(f"{file['directory']}/limb_postural_embeddings.npy", embed)

        if config['include_all_postural']:
            embed = np.empty((num_fr, config['n_components']))
            embed[:] = np.nan
            embed[good_fr,:] = all_postural_embed[start_fr:start_fr+num_good_fr]
            np.save(f"{file['directory']}/all_postural_embeddings.npy", embed)

        # Kinematic
        if config['include_marker_kinematic']:
            embed = np.empty((num_fr, config['n_components']))
            embed[:] = np.nan
            embed[good_fr,:] = marker_kinematic_embed[start_fr:start_fr+num_good_fr]
            np.save(f"{file['directory']}/marker_kinematic_embeddings.npy", embed)

        if config['include_angle_kinematic']:
            embed = np.empty((num_fr, config['n_components']))
            embed[:] = np.nan
            embed[good_fr,:] = angle_kinematic_embed[start_fr:start_fr+num_good_fr]
            np.save(f"{file['directory']}/angle_kinematic_embeddings.npy", embed)

        if config['include_limb_kinematic']:
            embed = np.empty((num_fr, config['n_components']))
            embed[:] = np.nan
            embed[good_fr,:] = limb_kinematic_embed[start_fr:start_fr+num_good_fr]
            np.save(f"{file['directory']}/limb_kinematic_embeddings.npy", embed)

        if config['include_all_kinematic']:
            embed = np.empty((num_fr, config['n_components']))
            embed[:] = np.nan
            embed[good_fr,:] = all_kinematic_embed[start_fr:start_fr+num_good_fr]
            np.save(f"{file['directory']}/all_kinematic_embeddings.npy", embed)

        # All Features
        if config['include_all_features']:
            embed = np.empty((num_fr, config['n_components']))
            embed[:] = np.nan
            embed[good_fr,:] = all_embed[start_fr:start_fr+num_good_fr]
            np.save(f"{file['directory']}/all_embeddings.npy", embed)

        start_fr += num_good_fr
print(f"::: Embedding Data Saving ::: Computation Time: {time.time()-start_timer}")

start_timer = time.time()
print(f"::: INFO Saving ::: START")
with open(f"{config['result_path']}/INFO.yaml", 'w') as file:
    documents = yaml.dump(dict(INFO_items), file)
print(f"::: INFO Saving ::: Computation Time: {time.time()-start_timer}")














