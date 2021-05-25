# HDBSCAN Clustering
import hdbscan
import numpy as np

# Watershed Clustering
import numpy as np
import skimage
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from scipy import stats

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
    
    return labels, num_clusters, clusterer




def Watershed(data, grid_dim=100, grid_padding=2, bw_method=None, verbose=False, ROI_thresh=0.001, fig_alpha=0.2, fig_s=7, watershed_line=False):
    # data - [num_fr, 2]
    # grid_dim - number of bins for density and watershed
    # verbose - True creates figures
    # ROI_thresh - controls the area of interest. Lower number limits the area to denser regions
    # take out nan frames for clustering

    num_fr, num_dim = data.shape
    nan_fr, nan_dim = np.where(np.isnan(data))
    np_unique_fr = np.unique(nan_fr)
    good_fr = np.array([True]*num_fr)
    good_fr[np_unique_fr] = False

    good_data = data[good_fr]

    # density grid
    xmin, ymin = good_data.min(axis=0)-grid_padding
    xmax, ymax = good_data.max(axis=0)+grid_padding

    print("Creating 2D Grid System...(1/7)")
    xgrid, ygrid = np.linspace(xmin,xmax,grid_dim), np.linspace(ymin,ymax,grid_dim)
    grid_dim_j = complex(grid_dim)
    X, Y = np.mgrid[xmin:xmax:grid_dim_j, ymin:ymax:grid_dim_j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # compute gaussian kernel desnity
    print("Computing Gaussian Kernel...(2/7)")
    kernel = stats.gaussian_kde(dataset=good_data.T, bw_method=bw_method)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    # find data pt's grid coordinate
    print("Finding Data Point Coordinates...(3/7)")
    stat, xedge, yedge, binnumber = stats.binned_statistic_2d(good_data[:,0], good_data[:,1], None, 'count', bins=[xgrid, ygrid], expand_binnumbers=True)
    
    if verbose:
        fig, ax = plt.subplots(2,3,figsize=(15,10))
        ax[0,0].scatter(good_data[:,0], good_data[:,1], alpha=fig_alpha, s=fig_s)
        ax[0,0].set(title="Data Points", xlabel="X Coord", ylabel="Y Coord")
        ax[0,1].imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r)
        ax[0,1].set(title="Gaussian KDE", xlabel="X Grid", ylabel="Y Grid")
    
    # find basin (local max)
    print("Finding Local Max Basin Point...(4/7)")
    local_max_idx = peak_local_max(Z, min_distance=1)
    local_max_bool = np.zeros(Z.shape, dtype=bool)
    local_max_bool[tuple(local_max_idx.T)] = True
    local_max_loc = np.vstack( (xgrid[local_max_idx[:,0]], ygrid[local_max_idx[:,1]]) ).T
    
    if verbose:
        ax[0,2].scatter(good_data[:,0], good_data[:,1], alpha=fig_alpha, s=fig_s)
        ax[0,2].scatter(local_max_loc[:,0], local_max_loc[:,1], c='r', s=7)
        ax[0,2].set(title="Basin Max Pts", xlabel="X Coord", ylabel="Y Coord")
    
    # create mask of ROI
    print("Creating Mask...(5/7)")
    mask = np.zeros(Z.shape)
    mask[Z>ROI_thresh]=1
    
    if verbose:
        ax[1,0].imshow(np.rot90(mask), cmap=plt.cm.gist_earth_r)
        ax[1,0].set(title="Mask", xlabel="X Grid", ylabel="Y Grid")
    
    # populate each basin region with corresponding marker label
    print("Finding Watershed...(6/7)")
    markers, _ = ndi.label(local_max_bool)
    segmented = skimage.morphology.watershed(np.max(Z)-Z, markers, mask=mask, watershed_line=watershed_line)
    
    if verbose:
        ax[1,1].imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r)
        ax[1,1].imshow(np.rot90(segmented), cmap=plt.cm.nipy_spectral)
        ax[1,1].set(title="Watershed", xlabel="X Grid", ylabel="Y Grid")
    
    # create cluster label of data points
    print("Creating Watershed label...(7/7)")
    watershed_labels = segmented[binnumber[0,:], binnumber[1,:]]
    print("COMPLETE")
    
    if verbose:
        ax[1,2].scatter(good_data[:,0], good_data[:,1], alpha=fig_alpha, s=fig_s, c=watershed_labels)
        ax[1,2].set(xlim=[xmin, xmax], ylim=[ymin, ymax])
        ax[1,2].set(title="Watershed Labels", xlabel="X Coord", ylabel="Y Coord")
        plt.show()

    # shift labels down one. undefined is currently set to 0
    watershed_labels -= 1
    labels = np.ones(num_fr)*-1
    labels[good_fr] = watershed_labels
    return labels, Z, markers, segmented



