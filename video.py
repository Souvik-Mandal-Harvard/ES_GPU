import numpy as np
import matplotlib, random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tqdm.notebook import tqdm

# frame info
# analyze_cluster = 11
# 
class Video:
    def __init__(self, bodypoints, angles, freq, power, embed, cluster, cluster_mean, fps=2, dpi=200):
        self.bodypoints = bodypoints
        self.embed = embed
        self.angles = angles
        self.freq = freq
        self.power = power
        self.cluster = cluster
        self.cluster_mean = cluster_mean
        self.fps = fps
        self.dpi = dpi

        SMALL_SIZE = 4
        matplotlib.rc('font', size=SMALL_SIZE)
        matplotlib.rc('axes', titlesize=SMALL_SIZE)

    def create_video(self, start_fr, stop_fr, filepath, start_pad=50, stop_pad=50):
        start = start_fr-start_pad
        stop = stop_fr+stop_pad
        
        # video info
        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=self.fps)
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(16, 24)
        ax1 = fig.add_subplot(gs[0:8, 0:8])
        ax2 = fig.add_subplot(gs[0:4, 8:])
        ax3 = fig.add_subplot(gs[4:8, 8:])
        ax4 = fig.add_subplot(gs[8:, 0:8])
        ax5 = fig.add_subplot(gs[8:, 8:16])
        ax6 = fig.add_subplot(gs[8:, 16:24])

        # ethogram plot
        num_clusters = np.max(self.cluster)+1
        color_palette = sns.color_palette('rainbow', num_clusters)
        cluster_colors = np.array([color_palette[x] for x in self.cluster])

        # density cluster plot
        sns.kdeplot(self.embed[:,0], self.embed[:,1], shade=True, shade_lowest=True, 
                    gridsize=100, levels=40, cmap='viridis',cbar=False, ax=ax6)
        for clust_i in range(num_clusters):
            ax6.annotate(clust_i, self.cluster_mean[clust_i], fontsize=4, fontweight='bold', color='k')
        ax6.set(xlabel='Component 1', ylabel='Component 2') #, title="UMAP Training Density Plot"

        # scatter cluster plot
        samp_frac = 0.5
        num_fr, _ = self.embed.shape
        idx = random.sample(range(num_fr), int(samp_frac*num_fr))

        with writer.saving(fig, filepath, dpi=self.dpi):
            for fr_i, fr in enumerate(tqdm(np.arange(start, stop), desc="Frame Loop")):
                # ant plot
                ax1.clear()
                ax1.set_xlim([-4,4]); ax1.set_ylim([-5,5]);
                for shadow_i in range(-10,1):
                    if shadow_i == 0:
                        alpha=0.8
                    else:
                        alpha = 0.1
                    ax1.plot(self.bodypoints[fr+shadow_i,0:4,0], self.bodypoints[fr+shadow_i,0:4,1], 
                             c='k', alpha=alpha, 
                             marker="o", markersize=3)
                    ax1.plot(self.bodypoints[fr+shadow_i,8:11,0], self.bodypoints[fr+shadow_i,8:11,1], 
                             c='tab:blue', alpha=alpha, 
                             marker="o", markersize=3)
                    ax1.plot(self.bodypoints[fr+shadow_i,21:24,0], self.bodypoints[fr+shadow_i,21:24,1], 
                             c='tab:orange', alpha=alpha,
                             marker="o", markersize=3)
                
                # angle plot
                ax2.clear()
                ax2.set_xlim([start,stop-1]); ax2.set_ylim([-1.0,1.0]);
                for i in range(2):
                    ax2.plot(np.arange(start, fr+1), self.angles[start:fr+1,i], alpha=0.5, linewidth=1, label=f"ang {i}")
                ax2.legend()
                ax2.set(xlabel='Frame', ylabel='Normalized Angle')
                
                # ethogram plot
                ax3.clear()
                ax3.set_xlim([start,stop-1]); ax3.set_ylim([-0.5,num_clusters]);
                ax3.scatter(np.arange(start, fr+1), self.cluster[start:fr+1], c=cluster_colors[start:fr+1], 
                            alpha=1, s=2, marker="s")
                ax3.set_yticks(range(0,num_clusters))
                ax3.set(xlabel='Frame', ylabel='Ethogram')
                
                # power spectrogram plot
                ax4.clear()
                for i in range(2):
                    ax4.plot(self.freq, self.power[i,:,fr].T, label=f"ang {i}", alpha=0.5, linewidth=1)
                ax4.set_xlabel("freq"); ax4.set_ylabel("power")
                ax4.set_ylim([-0.05,0.8])
                ax4.legend()
                
                # scatter cluster plot
                ax5.clear()
                ax5.scatter(self.embed[idx,0], self.embed[idx,1], 
                        c=cluster_colors[idx], 
                        alpha=0.2, s=0.1)
                ax5.plot(self.embed[start:fr+1,0], self.embed[start:fr+1,1],
                        c='k', linewidth=1, alpha=0.5)
                ax5.scatter(self.embed[fr,0], self.embed[fr,1],
                        c='k', s=5, marker="x")
                ax5.set(xlabel='Component 1', ylabel='Component 2', title=f"frame {fr}", xlim=ax6.get_xlim(), ylim=ax6.get_ylim())
                
                # take snapshot
                writer.grab_frame()
            writer.grab_frame()
            plt.close()