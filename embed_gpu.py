import time, sys
import pickle
import yaml
import numpy as np
from tqdm import tqdm

# Import Visualization
import matplotlib
import matplotlib.pyplot as plt

# Import Helper Function
from helper import locate_bad_fr, cuml_umap, cuml_pca
from utils.data import Dataset

def plot_embeddings(embed, title="Embedding", fname="embedding"):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(embed[:,0], embed[:,1], s=1, alpha=0.01)
    ax.set(title=title)
    plt.savefig(f"figures/{fname}.png")
    return

def main():
    # grab arguments
    config_name = sys.argv[1]

    with open(config_name) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config_path = f"{config['GPU_project_path']}/{config_name}"
    PROJECT_PATH = config['GPU_project_path']
    Data = Dataset(PROJECT_PATH, config_path)
    Data.load_data()

    # configuration
    INFO = Data.info
    INFO_items = list(INFO.items())
    INFO_items.sort(key=lambda x: x[1]['order'])
    config = Data.config

    with open (f"{PROJECT_PATH}/{config['result_path']}/angle_scale_model.pickle", 'rb') as file:
        angle_scaler = pickle.load(file)
    with open (f"{PROJECT_PATH}/{config['result_path']}/limb_scale_model.pickle", 'rb') as file:
        limb_scaler = pickle.load(file)

    # features
    tot_bp = Data.data_obj['rotated_bodypoints']*config['postural_weight']
    tot_angle = angle_scaler.transform(Data.data_obj['angles'])*config['postural_weight']
    tot_limb = limb_scaler.transform(Data.data_obj['limbs'])*config['postural_weight']
    tot_marker_pwr = None # Implement this later maybe
    tot_angle_pwr = Data.data_obj['angle_power']*config['kinematic_weight']
    tot_limb_pwr = Data.data_obj['limb_power']*config['kinematic_weight']

    # take out bad frames
    tot_good_fr, tot_bad_fr, tot_disregard_fr = locate_bad_fr(config, tot_bp)

    tot_angle = tot_angle[tot_good_fr]
    tot_limb = tot_limb[tot_good_fr]
    tot_angle_pwr = tot_angle_pwr[tot_good_fr]
    tot_limb_pwr = tot_limb_pwr[tot_good_fr]

    print("******************")
    nan_fr = np.where(np.isnan(tot_limb_pwr))[0]
    print(np.unique(nan_fr))
    print("******************")

    # Postural Embedding
    if config['include_marker_postural']:
        start_timer = time.time()
        print(f"::: Marker Postural ::: START")
        p_marker_embed(config, INFO_items, tot_bp)
        print(f"::: Marker Postural ::: Computation Time: {time.time()-start_timer}")
    if config['include_angle_postural']:
        start_timer = time.time()
        print(f"::: Angle Postural ::: START")
        p_angle_embed(config, INFO_items, tot_angle)
        print(f"::: Angle Postural  ::: Computation Time: {time.time()-start_timer}")
    if config['include_limb_postural']:
        start_timer = time.time()
        print(f"::: Limb Postural  ::: START")
        p_limb_embed(config, INFO_items, tot_limb)
        print(f"::: Limb Postural  ::: Computation Time: {time.time()-start_timer}")
    if config['include_all_postural']:
        start_timer = time.time()
        print(f"::: Angle & Limb Postural ::: START Timer")
        p_angle_limb_embed(config, INFO_items, tot_angle, tot_limb)
        print(f"::: Angle & Limb Postural ::: Time Stamp: {time.time()-start_timer}")

    # Kinematic Embedding
    if config['include_marker_kinematic']:
        start_timer = time.time()
        print(f"::: Marker Kinematic ::: START")
        k_marker_embed(config, INFO_items, tot_marker_pwr)
        print(f"::: Marker Kinematic ::: Computation Time: {time.time()-start_timer}")
    if config['include_angle_kinematic']:
        start_timer = time.time()
        print(f"::: Angle Kinematic ::: START")
        k_angle_embed(config, INFO_items, tot_angle_pwr)
        print(f"::: Angle Kinematic ::: Computation Time: {time.time()-start_timer}")
    if config['include_limb_kinematic']:
        start_timer = time.time()
        print(f"::: Limb Kinematic ::: START")
        k_limb_embed(config, INFO_items, tot_limb_pwr)
        print(f"::: Limb Kinematic ::: Computation Time: {time.time()-start_timer}")
    if config['include_all_kinematic']:
        start_timer = time.time()
        print(f"::: Angle & Limb Kinematic ::: START Timer")
        k_angle_limb_embed(config, INFO_items, tot_angle_pwr, tot_limb_pwr)
        print(f"::: Angle & Limb Kinematic ::: Time Stamp: {time.time()-start_timer}")

    if config['include_all_features']:
        start_timer = time.time()
        print(f"::: Angle & Limb Kinematic & Postural ::: START Timer")
        kp_angle_limb_embed(config, INFO_items, tot_angle, tot_limb, tot_angle_pwr, tot_limb_pwr)
        print(f"::: Angle & Limb Kinematic & Postural ::: Time Stamp: {time.time()-start_timer}")

def p_marker_embed(config, INFO_items, tot_bp):
    num_fr, num_bp, num_dim = tot_bp.shape
    tot_bp_mod = tot_bp[:,:,0:num_dim-1].reshape(num_fr, num_bp*(num_dim-1))
    embeddings = cuml_umap(config, tot_bp_mod)
    plot_embeddings(embeddings, title="Marker Postural",fname="marker_postural_embeddings")
    save_embeddings(config, INFO_items, embeddings, fname="marker_postural_embeddings")
    return
def p_angle_embed(config, INFO_items, tot_angle):
    embeddings = cuml_umap(config, tot_angle)
    plot_embeddings(embeddings, title="Angle Postural", fname="angle_postural_embeddings")
    save_embeddings(config, INFO_items, embeddings, fname="angle_postural_embeddings")
    return
def p_limb_embed(config, INFO_items, tot_limb):
    embeddings = cuml_umap(config, tot_limb)
    plot_embeddings(embeddings, title="Limb Postural", fname="limb_postural_embeddings")
    save_embeddings(config, INFO_items, embeddings, fname="limb_postural_embeddings")
    return
def p_angle_limb_embed(config, INFO_items, tot_angle, tot_limb):
    postural_features = np.concatenate([tot_angle, tot_limb], axis=1)
    embeddings = cuml_umap(config, postural_features)
    plot_embeddings(embeddings, title="Angle & Limb Postural", fname="angle_limb_postural_embeddings")
    save_embeddings(config, INFO_items, embeddings, fname="all_postural_embeddings")
    return

def k_marker_embed(config, INFO_items, tot_marker_pwr):
    num_fr, num_freq, num_feat = tot_marker_pwr.shape
    marker_kinematic_pca, _ = cuml_pca(config, 
        tot_marker_pwr.reshape(num_fr, num_freq*num_feat), 
        components=config['marker_kinematic_pca_components'])
    embeddings = cuml_umap(config, marker_kinematic_pca)
    plot_embeddings(embeddings, title="Marker Kinematic", fname="marker_kinematic_embeddings")
    save_embeddings(config, INFO_items, embeddings, fname="marker_kinematic_embeddings")
    return
def k_angle_embed(config, INFO_items, tot_angle_pwr):
    num_fr, num_freq, num_feat = tot_angle_pwr.shape
    angle_kinematic_pca, _ = cuml_pca(config, 
        tot_angle_pwr.reshape(num_fr, num_freq*num_feat),
        components=config['angle_kinematic_pca_components'])
    embeddings = cuml_umap(config, angle_kinematic_pca)
    plot_embeddings(embeddings, title="Angle Kinematic", fname="angle_kinematic_embeddings")
    save_embeddings(config, INFO_items, embeddings, fname="angle_kinematic_embeddings")
    return
def k_limb_embed(config, INFO_items, tot_limb_pwr):
    num_fr, num_freq, num_feat = tot_limb_pwr.shape
    limb_kinematic_pca, _ = cuml_pca(config, 
        tot_limb_pwr.reshape(num_fr, num_freq*num_feat), 
        components=config['limb_kinematic_pca_components'])
    embeddings = cuml_umap(config, limb_kinematic_pca)
    plot_embeddings(embeddings, title="Limb Kinematic", fname="limb_kinematic_embeddings")
    save_embeddings(config, INFO_items, embeddings, fname="limb_kinematic_embeddings")
    return
def k_angle_limb_embed(config, INFO_items, tot_angle_pwr, tot_limb_pwr):
    num_fr, num_freq, num_angle_feat = tot_angle_pwr.shape
    num_fr, num_freq, num_limb_feat = tot_limb_pwr.shape
    angle_kinematic_pca, _ = cuml_pca(config, 
        tot_angle_pwr.reshape(num_fr, num_freq*num_angle_feat), 
        components=config['angle_kinematic_pca_components'])
    limb_kinematic_pca, _ = cuml_pca(config, 
        tot_limb_pwr.reshape(num_fr, num_freq*num_limb_feat), 
        components=config['limb_kinematic_pca_components'])
    kinematic_features = np.concatenate([
        angle_kinematic_pca,
        limb_kinematic_pca
    ], axis=1)
    embeddings = cuml_umap(config, kinematic_features)
    plot_embeddings(embeddings, title="All Kinematic", fname="all_kinematic_embeddings")
    save_embeddings(config, INFO_items, embeddings, fname="all_kinematic_embeddings")
    return

def kp_angle_limb_embed(config, INFO_items, tot_angle, tot_limb, tot_angle_pwr, tot_limb_pwr):
    postural_features = np.concatenate([tot_angle, tot_limb], axis=1)
    num_fr, num_freq, num_angle_feat = tot_angle_pwr.shape
    num_fr, num_freq, num_limb_feat = tot_limb_pwr.shape
    angle_kinematic_pca, _ = cuml_pca(config, 
        tot_angle_pwr.reshape(num_fr, num_freq*num_angle_feat), 
        components=config['angle_kinematic_pca_components'])
    limb_kinematic_pca, _ = cuml_pca(config, 
        tot_limb_pwr.reshape(num_fr, num_freq*num_limb_feat), 
        components=config['limb_kinematic_pca_components'])
    kinematic_features = np.concatenate([
        angle_kinematic_pca,
        limb_kinematic_pca], axis=1)

    kp_angle_limb_features = np.concatenate([postural_features, kinematic_features], axis=1)
    embeddings = cuml_umap(config, kp_angle_limb_features)
    plot_embeddings(embeddings, title="Kinematic & Postural, Angle & Limb Feature", fname="kp_angle_limb_embeddings")
    save_embeddings(config, INFO_items, embeddings, fname="all_embeddings")
    return

def save_embeddings(config, INFO_items, embeddings, fname="embeddings"):
    start_fr = 0
    for key, file in INFO_items:
        save_path = file['directory']
        bp = np.load(f"{save_path}/rotated_bodypoints.npy")
        num_fr, _, _ = bp.shape
        # locate the good frames
        good_fr, bad_fr, disregard_fr = locate_bad_fr(config, bp)
        num_good_fr = len(good_fr)
        # bad frames default to nan
        full_embeddings = np.empty((num_fr, config['n_components']))
        full_embeddings[:] = np.nan
        full_embeddings[good_fr,:] = embeddings[start_fr:start_fr+num_good_fr]
        np.save(f"{save_path}/{fname}.npy", full_embeddings)
        start_fr += num_good_fr
    return

if __name__ == "__main__":
    main()


