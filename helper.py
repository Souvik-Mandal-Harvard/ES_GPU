import numpy as np

# Import RAPIDS
import cudf, cuml

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
    (num_bp, num_dim, num_fr) = data.shape
    (num_fr, num_bp, num_dim) = data.shape
    num_feat = len(keys)
    angles = np.zeros((num_fr, num_feat))
    for feat, ele in enumerate(keys):
        a = data[:,ele['a'],:]
        b = data[:,ele['b'],:]
        c = data[:,ele['c'],:]
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.sum(ba*bc,axis=-1)/ (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
        angles[:,feat] = np.arccos(cosine_angle)/np.pi # normalize
    return angles

def cuml_umap(config, feature):
    num_fr = feature.shape[0]
    embed = np.zeros((num_fr, config['n_components']))
    # embed = np.zeros((num_fr, config['n_components']+1))
    df = cudf.DataFrame(feature)
    cu_embed = cuml.UMAP(n_components=config['n_components'], n_neighbors=config['n_neighbors'], n_epochs=config['n_epochs'], 
                    min_dist=config['min_dist'], spread=config['spread'], negative_sample_rate=config['negative_sample_rate'],
                    init=config['init'], repulsion_strength=config['repulsion_strength']).fit_transform(df)
    embed[:,0:config['n_components']] = cu_embed.to_pandas().to_numpy()
    return embed

def cuml_pca(config, feature, components=10):
    num_fr = feature.shape[0]
    embed = np.zeros((num_fr, components))
    # embed = np.zeros((num_fr, config['n_components']+1))
    df = cudf.DataFrame(feature)
    pca = cuml.PCA(n_components=components)
    pca.fit(df)
    cu_embed = pca.transform(df)
    exp_var = pca.explained_variance_ratio_.to_pandas().to_numpy()
    embed[:,0:components] = cu_embed.to_pandas().to_numpy()
    
    return embed, exp_var