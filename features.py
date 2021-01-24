import time
import yaml
import numpy as np
from tqdm import tqdm
import collections

# Import Signal Processor
from scipy.signal import morlet2, cwt
# Import Visualization
import matplotlib
import matplotlib.pyplot as plt

# Import Helper Function
from helper import angle_calc, cuml_umap, cuml_pca

start_timer = time.time()

with open("config.yaml") as f:
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

    file['good_fr'] = good_fr.tolist()
    file['bad_fr'] = bad_fr.tolist()
    file['disregard_fr'] = disregard_fr.tolist()

    ### Postural Features ###
    # Marker Position
    if config['include_marker_postural'] or config['include_all_postural']:
        # remove bad frames
        bp_markers = bp[:,config['markers'],:]
        tot_bp.append(bp_markers[good_fr,:])


    # Joint Angle
    if config['include_angle_postural'] or config['include_all_postural']:
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
    if config['include_limb_postural'] or config['include_all_postural']:
        limbs = np.zeros((num_fr, len(config['limbs'])))
        for i, limb_pts in enumerate(config['limbs']):
            limb_i = bp[:,limb_pts,0:2]
            limbs[:,i] = np.sqrt((limb_i[:,0,0]-limb_i[:,1,0])**2 + (limb_i[:,0,1]-limb_i[:,1,1])**2)
        tot_limb.append(limbs[good_fr,:])

    ### Kinematic Features ###
    # TODO

# Concat Data
if config['include_marker_postural'] or config['include_all_postural']:
    tot_bp = np.concatenate(tot_bp)
if config['include_angle_postural'] or config['include_all_postural']:
    tot_angle = np.concatenate(tot_angle)
if config['include_limb_postural'] or config['include_all_postural']:
    tot_limb = np.concatenate(tot_limb)

### Postural Features ###
start_timer = time.time()
# 1) Marker Position
if config['include_marker_postural']:
    print(f"::: Marker Position ::: START")
    num_fr, num_bp, num_bp_dim = tot_bp.shape
    tot_bp_mod = tot_bp[:,:,0:num_bp_dim-1].reshape(num_fr, num_bp*(num_bp_dim-1))
    marker_postural_embed = cuml_umap(config, tot_bp_mod)
    plot_embedding(marker_postural_embed, title="Marker Position",fname="marker_position_embedding")
    print(f"::: Marker Position ::: Computation Time: {time.time()-start_timer}")

# 2) Joint Angle
start_timer = time.time()
if config['include_angle_postural']:
    print(f"::: Joint Angle ::: START")
    angle_postural_embed = cuml_umap(config, tot_angle[:,:,0])
    plot_embedding(angle_postural_embed, title="Joint Angle", fname="joint_angle_embedding")
    print(f"::: Joint Angle ::: Computation Time: {time.time()-start_timer}")

# 3) Limb Length
start_timer = time.time()
if config['include_limb_postural']:
    print(f"::: Limb Length ::: START")
    limb_postural_embed = cuml_umap(config, tot_limb)
    plot_embedding(limb_postural_embed, title="Limb Length", fname="limb_length_embedding")
    print(f"::: Limb Length ::: Computation Time: {time.time()-start_timer}")

# 4) Marker Position, Joint Angle, & Limb Length (TODO Later)
if config['include_all_postural']:
    start_timer = time.time()
    print(f"::: All Postural ::: START Timer")
    
    num_fr, num_bp, num_bp_dim = tot_bp.shape
    tot_bp_mod = tot_bp[:,:,0:num_bp_dim-1].reshape(num_fr, num_bp*(num_bp_dim-1))
    
    # PCA Embedding
    bp_pca, exp_var = cuml_pca(config, tot_bp_mod, components=10) # 21
    print(exp_var)
    print(f"::: All Postural Features (BP PCA) ::: Time Stamp: {time.time()-start_timer}")
    bp_angle, exp_var = cuml_pca(config, tot_angle[:,:,0], components=10) # 12
    print(exp_var)
    print(f"::: All Postural Features (Angle PCA) ::: Time Stamp: {time.time()-start_timer}")
    bp_limb, exp_var = cuml_pca(config, tot_limb, components=10) # 13
    print(exp_var)
    print(f"::: All Postural Features (Limb PCA) ::: Time Stamp: {time.time()-start_timer}")
    
    # UMAP Embedding
    feature = np.concatenate([bp_pca, bp_angle, bp_limb], axis=1)
    postural_embed = cuml_umap(config, feature)
    plot_embedding(postural_embed, title="All Postural", fname="all_postural_embedding")
    print(f"::: All Postural Features ::: Time Stamp: {time.time()-start_timer}")

### Kinematic Features ###
# TODO

### Save Embedding ###
# TODO: add bad features back into embed
if config['save_embeddings']:
    start_fr = 0
    for key, file in INFO_items:
        num_fr = file['number_frames']
        good_fr = file['good_fr']
        num_good_fr = len(good_fr)

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
            embed[good_fr,:] = postural_embed[start_fr:start_fr+num_good_fr]
            np.save(f"{file['directory']}/all_postural_embeddings.npy", embed)

        start_fr += num_good_fr

with open(f"{config['result_path']}/INFO.yaml", 'w') as file:
    documents = yaml.dump(dict(INFO_items), file)












