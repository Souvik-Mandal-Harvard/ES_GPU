import os, random, matplotlib, pickle, yaml, cudf, cuml
import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from glob import glob

result_path = "results/round3_antennae"
with open(f"{result_path}/INFO.yaml") as f:
    INFO = yaml.load(f, Loader=yaml.FullLoader)
    INFO_values = list(INFO.values())
    INFO_values.sort(key=lambda x: x['order'])
    
config_path = "."
with open(f"{config_path}/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


tot_bp, tot_bp_scaled, tot_bp_rotated, tot_body_orientation, tot_angles, tot_rotations, tot_power, tot_embed = [], [], [], [], [], [], [], []
for file in tqdm(INFO_values):
    tot_bp.append( np.load(f"{file['directory']}/bodypoints.npy") )
    tot_bp_scaled.append( np.load(f"{file['directory']}/scaled_bodypoints.npy") )
    tot_bp_rotated.append( np.load(f"{file['directory']}/rotated_bodypoints.npy") )
    tot_body_orientation.append( np.load(f"{file['directory']}/body_orientation_angles.npy") )
    tot_angles.append( np.load(f"{file['directory']}/angles.npy") )
    tot_power.append( np.load(f"{file['directory']}/power.npy") )
    tot_embed.append( np.load(f"{file['directory']}/embeddings.npy") )

tot_bp = np.concatenate(tot_bp)
tot_bp_scaled = np.concatenate(tot_bp_scaled)
tot_bp_rotated = np.concatenate(tot_bp_rotated)
tot_body_orientation = np.concatenate(tot_body_orientation)
tot_angles = np.concatenate(tot_angles)
tot_power = np.concatenate(tot_power, axis=2)
tot_embed = np.concatenate(tot_embed)


print(f"tot_bp shape: {tot_bp.shape}")
print(f"tot_bp_unrot shape: {tot_bp_scaled.shape}")
print(f"tot_bp_rotated shape: {tot_bp_rotated.shape}")
print(f"tot_angles shape: {tot_angles.shape}")
print(f"tot_power shape: {tot_power.shape}")
print(f"tot_embed shape: {tot_embed.shape}")

### Embed Postural Features

def cuml_umap(feature):
    embed = np.zeros((num_fr, config['n_components']))
    # embed = np.zeros((num_fr, config['n_components']+1))
    df = cudf.DataFrame(feature)
    cu_embed = cuml.UMAP(n_components=config['n_components'], n_neighbors=config['n_neighbors'], n_epochs=config['n_epochs'], 
                    min_dist=config['min_dist'], spread=config['spread'], negative_sample_rate=config['negative_sample_rate'],
                    init=config['init'], repulsion_strength=config['repulsion_strength']).fit_transform(df)
    embed[:,0:config['n_components']] = cu_embed.to_pandas().to_numpy()
    return embed


# 1) Marker Position
num_fr, num_bp, num_bp_dim = tot_bp_scaled.shape
tot_bp_scaled_mod = tot_bp_scaled[:,:,0:num_bp_dim-1].reshape(num_fr, num_bp*(num_bp_dim-1))
embed = cuml_umap(tot_bp_scaled_mod)

fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(embed[:,0], embed[:,1], s=1, alpha=0.05)
ax.set(title="Postural Embedding")
plt.savefig(f"figures/marker_position_embedding.png")

# 2) Joint Angle


# 3) Limb Length


# 4) Marker Position, Joint Angle, & Limb Length
















