import time
import yaml
import numpy as np

# Import Signal Processor
from scipy.signal import morlet2, cwt
# Import Visualization
import matplotlib
import matplotlib.pyplot as plt

# Import Helper Function
from helper import angle_calc, cuml_umap

start_timer = time.time()

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
with open(f"{config['result_path']}/INFO.yaml") as f:
    INFO = yaml.load(f, Loader=yaml.FullLoader)
    INFO_values = list(INFO.values())
    INFO_values.sort(key=lambda x: x['order'])

angles_list, power_list = [], []

tot_bp, tot_angle, tot_limb = [], [], []

for file in INFO_values:
    save_path = file['directory']
    bp = np.load(f"{save_path}/rotated_bodypoints.npy")
    num_fr, num_bp, _ = bp.shape

    # Marker Position
    if config['include_marker_postural'] or config['include_marker_postural']:
        tot_bp.append(bp)

    # Joint Angle
    if config['include_angle_postural'] or config['include_angle_postural']:
        # compute angle
        num_angles = len(config['angles'])
        angles = np.zeros((num_fr, num_angles, 2))
        angles[:,:,0] = angle_calc(bp[:,:,0:2], config['angles'])
        # measure angle likelihood
        likelihood = bp[:,:,2]
        for ang_idx, angle_key in enumerate(config['angles']):
            angles[:,ang_idx,1] = likelihood[:,angle_key['a']] * likelihood[:,angle_key['b']] * likelihood[:,angle_key['c']]
        if config['save_angles']:
            np.save(f"{save_path}/angles.npy", angles)
        tot_angle.append(angles)

    # Limb Length
    if config['include_limb_postural'] or config['include_limb_postural']:
        limbs = np.zeros((num_fr, len(config['limbs'])))
        for i, limb_pts in enumerate(config['limbs']):
            limb_i = bp[:,limb_pts,0:2]
            limbs[:,i] = np.sqrt((limb_i[:,0,0]-limb_i[:,1,0])**2 + (limb_i[:,0,1]-limb_i[:,1,1])**2)
        tot_limb.append(limbs)

# Concat Data
if config['include_marker_postural'] or config['include_marker_postural']:
    tot_bp = np.concatenate(tot_bp)
if config['include_angle_postural'] or config['include_angle_postural']:
    tot_angle = np.concatenate(tot_angle)
if config['include_limb_postural'] or config['include_limb_postural']:
    tot_limb = np.concatenate(tot_limb)

### Postural Features ###
start_timer = time.time()
print(f"::: Marker Position ::: START")
# 1) Marker Position
if config['include_marker_postural']:
    num_fr, num_bp, num_bp_dim = tot_bp.shape
    tot_bp_mod = tot_bp[:,:,0:num_bp_dim-1].reshape(num_fr, num_bp*(num_bp_dim-1))
    embed = cuml_umap(tot_bp_mod)
    plot_embedding(embed, title="Marker Position",fname="marker_position_embedding")
    print(f"::: Marker Position ::: Computation Time: {time.time()-start_timer}")

# 2) Joint Angle
start_timer = time.time()
print(f"::: Joint Angle ::: START")
if config['include_angle_postural']:
    embed = cuml_umap(tot_angle[:,:,0])
    plot_embedding(embed, title="Joint Angle", fname="joint_angle_embedding")
    print(f"::: Joint Angle ::: Computation Time: {time.time()-start_timer}")

# 3) Limb Length
start_timer = time.time()
print(f"::: Limb Length ::: START")
if config['include_limb_postural']:
    embed = cuml_umap(tot_limb)
    plot_embedding(embed, title="Limb Length", fname="limb_length_embedding")
    print(f"::: Limb Length ::: Computation Time: {time.time()-start_timer}")

# 4) Marker Position, Joint Angle, & Limb Length (TODO Later)
if False:
    print(f"::: Total Postural Features ::: Computation Time: {time.time()-start_timer}")

### Kinematic Features ###
# TODO

def plot_embedding(embed, title="test", fname="test"):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(embed[:,0], embed[:,1], s=1, alpha=0.05)
    ax.set(title=title)
    plt.savefig(f"figures/{fname}.png")
    return










