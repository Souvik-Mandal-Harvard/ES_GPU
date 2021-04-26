import hdbscan
import numpy as np

def HDBSCAN(embed, min_cluster_size=7000, min_samples=10, cluster_selection_epsilon=0, cluster_selection_method="leaf", memory="memory"):
    # HDBSCAN
    num_fr = len(embed)
    (good_fr, good_bp) = np.where( ~np.isnan(embed) )
    good_fr = np.unique(good_fr)
    labels = np.ones(num_fr)*-2

    # hdbscan clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                min_samples=min_samples,
                                cluster_selection_epsilon=cluster_selection_epsilon,
                                cluster_selection_method=cluster_selection_method,
                                memory=memory
                               ).fit(embed[good_fr,:])
    # parameters
    labels[good_fr] = clusterer.labels_
    num_clusters = int(np.max(labels)+1)
    outlier_pts = np.where(labels== -1)[0]
    print(f"Frac Outlier: {len(outlier_pts)/len(labels)}")
    print(f"# Clusters: {num_clusters}")
    
    return labels, num_clusters