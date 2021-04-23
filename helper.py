import collections
import numpy as np
# import hdbscan
# Import Signal Processor
from scipy.signal import morlet2, cwt

def locate_bad_fr(config, bp):
    num_fr,_,_ = bp.shape
    likelihood = bp[:,:,2]
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
    return good_fr, bad_fr, disregard_fr

def _rotational(data, axis_bp):
    # rotate axis to be vertical; only works with 2 dimensions as of right now
    # data format: num_fr, num_bp, num_dim
    # angle_list: angle of rotation from the vertical per frame
    rot_data = np.copy(data)
    num_bp = rot_data.shape[1]
    axis_vector = rot_data[:,axis_bp,:]
    angle_list = np.sign(axis_vector[:,0]) * np.pi/2 - np.arctan( axis_vector[:,1]/axis_vector[:,0] ) # angle rotated per frame
    # set nan to 0 angle rotation
    angle_list[np.isnan(angle_list)] = 0
    # rotate each body point
    for i in range(num_bp):
        rot = _rotate(rot_data[:,i,:].T, angle_list)
        rot_data[:,i,:] = rot.T
    return (rot_data, angle_list)

def _rotate(data, angle):
    return np.einsum('ijk,jk ->ik', 
        np.array([[np.cos(angle), -1*np.sin(angle)], [np.sin(angle), np.cos(angle)]]), data)
    
def angle_calc(data, keys):
    (num_fr, num_bp, num_dim) = data.shape
    num_feat = len(keys)
    angles = np.zeros((num_fr, num_feat))
    for feat, ele in enumerate(keys):
        a = data[:,ele['a'],:]
        b = data[:,ele['b'],:]
        c = data[:,ele['c'],:]
        # compute vector
        ba = a - b
        bc = c - b
        # find normalize dot and cross product
        ba_norm = ba / np.linalg.norm(ba,axis=1,keepdims=True)
        bc_norm = bc / np.linalg.norm(bc,axis=1,keepdims=True)
        ba_bc_dot = (ba_norm*bc_norm).sum(axis=1)
        ba_bc_cross = bc_norm[:,0]*ba_norm[:,1]-bc_norm[:,1]*ba_norm[:,0]
        # compute angle between two vectors
        if ele['method'] == 0: #[-pi,pi]; cw - positive, ccw - negative
            ang_meas = np.arctan2(ba_bc_cross,ba_bc_dot)
            
        elif ele['method'] == 1: #[0,2pi]; cw - positive, ccw - negative
            ang_meas = np.arctan2(ba_bc_cross,ba_bc_dot)
            neg_idx = (np.sign(ang_meas)==-1)
            ang_meas[neg_idx] += 2*np.pi 
        # account for nan frames
        nan_idx, = np.where(np.isnan(ang_meas))
        ang_meas[nan_idx] = 0.0
        angles[:,feat] = ang_meas

        # # OLD ANGLE METHOD
        # cosine_angle = np.sum(ba*bc,axis=-1)/ (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
        # # fix nan data
        # nan_idx, = np.where(np.isnan(cosine_angle))
        # cosine_angle[nan_idx] = 0.0
        # # fix out of domain data
        # cosine_angle[cosine_angle>1] = 1.0
        # cosine_angle[cosine_angle<-1] = -1.0
        # angles[:,feat] = np.arccos(cosine_angle) # normalize
        # # angles[:,feat] = np.arccos(cosine_angle)/np.pi # normalize
    return angles

def morlet(config, data):
    # data - (frames, features)

    # Morlet Wavelet
    num_fr, num_feat = data.shape
    power = np.zeros((num_feat, config['f_bin'], num_fr))
    max_freq, min_freq = config['f_max'], config['f_min'] # Nyquist Frequency
    freq = max_freq*2**(-1*np.log2(max_freq/min_freq)*
        (np.arange(config['f_bin'],0,-1)-1)/(config['f_bin']-1)) # dyadic frequency bins
    widths = config['w']*config['fps'] / (2*freq*np.pi)
    # Normalization Factor
    s = (config['w'] + np.sqrt(2+config['w']**2))/(4*np.pi*freq)
    C = np.pi**(-0.25)*np.exp(0.25*(config['w']-np.sqrt(config['w']**2+2))**2)/np.sqrt(2*s)
    
    for i in range(num_feat):
        cwtm = cwt(data[:,i], morlet2, widths, dtype=None, w=config['w'])
        # power[i] = np.abs(cwtm)**2
        power[i,:,:] = (np.abs(cwtm/np.expand_dims(np.sqrt(s),1)))/np.expand_dims(C, axis=(0,2))
    return power.T

def cuml_umap(config, feature):
    # Import RAPIDS
    import cudf, cuml
    print("INSIDE CUML_UMAP")
    print(feature.shape)
    num_fr = feature.shape[0]
    embed = np.zeros((num_fr, config['n_components']))

    # TRY THIS LATER!!!!!!!!!!!!!!!!! IF YOU EVER RUN OUT OF SPACE; COMPARE EMBEDDINGS AND SEE IF SMALLER DATA MAKES A DIFFERENCE
    # df = cudf.DataFrame(feature, dtype='float32')
    df = cudf.DataFrame(feature)

    cu_embed = cuml.UMAP(n_components=config['n_components'], n_neighbors=config['n_neighbors'], n_epochs=config['n_epochs'], 
                    min_dist=config['min_dist'], spread=config['spread'], negative_sample_rate=config['negative_sample_rate'],
                    init=config['init'], repulsion_strength=config['repulsion_strength'], output_type='numpy').fit_transform(df)
    embed[:,0:config['n_components']] = cu_embed
    return embed

def cuml_pca(config, feature, components=10):
    # Import RAPIDS
    import cudf, cuml

    num_fr = feature.shape[0]
    embed = np.zeros((num_fr, components))
    df = cudf.DataFrame(feature)
    pca = cuml.PCA(n_components=components, svd_solver='jacobi')
    pca.fit(df)
    cu_embed = pca.transform(df)
    exp_var = pca.explained_variance_ratio_.to_pandas().to_numpy()
    embed[:,0:components] = cu_embed.to_pandas().to_numpy()
    
    print("*** PCA ***")
    # print(exp_var)
    print(f"Sum of Explained Variance: {np.sum(exp_var)}")

    return embed, exp_var


# def HDBSCAN(embed, min_cluster_size=7000, min_samples=10, cluster_selection_epsilon=0, cluster_selection_method="leaf", memory="memory"):
#     # HDBSCAN
#     num_fr = len(embed)
#     (good_fr, good_bp) = np.where( ~np.isnan(embed) )
#     good_fr = np.unique(good_fr)
#     labels = np.ones(num_fr)*-2

#     # hdbscan clustering
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
#                                 min_samples=min_samples,
#                                 cluster_selection_epsilon=cluster_selection_epsilon,
#                                 cluster_selection_method=cluster_selection_method,
#                                 memory=memory
#                                ).fit(embed[good_fr,:])
#     # parameters
#     labels[good_fr] = clusterer.labels_
#     num_clusters = int(np.max(labels)+1)
#     outlier_pts = np.where(labels== -1)[0]
#     print(f"Frac Outlier: {len(outlier_pts)/len(labels)}")
#     print(f"# Clusters: {num_clusters}")
    
#     return labels, num_clusters


