import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


def ant_model(bp, skeleton=None, skeleton_color=None, fr=0, bounds=None):
    fig, ax = plt.subplots(figsize=(3,4))
    # plot bodypoints
    ax.scatter(bp[fr,:,0], bp[fr,:,1], s=7, c='k')
    # plot skeleton
    if (skeleton is not None) and (skeleton_color is not None):
        for skeleton_i, color_i in zip(skeleton, skeleton_color):
            ax.plot(bp[fr,skeleton_i,0], bp[fr,skeleton_i,1], alpha=0.7, c=color_i, linewidth=2)
    ax.set(title=f"Ant Model Bodypoints for Frame {fr}",
        xlabel="scaled x coord", ylabel="scaled y coord")
    if bounds is not None:
        ax.set(xlim=[bounds[0],bounds[1]] , ylim=[bounds[2],bounds[3]])
    plt.show()
    return

def plot_skeleton_length(skel_len):
    fig, ax = plt.subplots(figsize=(10,5))
    plt.hist(skel_len.flatten(),200, color='k')
    # plt.xlim([0,50])
    plt.xlabel("Length (pixel)")
    plt.ylabel("Count")
    plt.title("Skeleton Component Length")
    plt.show()


def plot_HDBSCAN(ax, embed, labels, color_palette, marker_size=1, alpha=0.005, xlim=None, ylim=None, toggle_numbering=False):
    num_clusters = int(np.max(labels)+1)
    outlier_pts = np.where(labels== -1)[0]
    labeled_pts = np.where(labels!= -1)[0]

    # cmap: coloring
    cluster_colors = np.array([color_palette[int(x)] if int(x) >= 0
                      else (0.5, 0.5, 0.5)
                      for x in labels])

    # cluster colors
    ax.scatter(embed[outlier_pts,0], embed[outlier_pts,1], 
               c="gray", s=marker_size, alpha=alpha)
    ax.scatter(embed[labeled_pts,0], embed[labeled_pts,1], 
               c=cluster_colors[labeled_pts], s=marker_size, alpha=alpha)
    ax.set(xlabel='UMAP C1', ylabel='UMAP C2', title="All Postural Features HDBSCAN Clusters")
    
    # numbering
    cluster_mean = []
    if toggle_numbering:
        for i in tqdm(range(num_clusters)):
            cluster_mean.append(np.mean(embed[labels==i,:], axis=0).tolist())
        cluster_mean = np.array(cluster_mean)
    
        if (xlim!=None) | (ylim!=None):
            ax.set(xlim=xlim, ylim=ylim)
            
            # plot numbering
            x_cond = (cluster_mean[:,0]>xlim[0]) & (cluster_mean[:,0]<xlim[1])
            y_cond = (cluster_mean[:,1]>ylim[0]) & (cluster_mean[:,1]<ylim[1])
            clust_numb_disp = np.where(x_cond&y_cond)[0]
            for i in tqdm(clust_numb_disp):
                ax.annotate(i, cluster_mean[i], fontsize=10, fontweight='bold')
        else:
            for i in tqdm(range(num_clusters)):
                ax.annotate(i, cluster_mean[i], fontsize=10, fontweight='bold')
    return cluster_mean