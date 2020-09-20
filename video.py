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
        
        self.setup()

    def setup(self):
        # video info
        self.fig = plt.figure(constrained_layout=True)
        gs = self.fig.add_gridspec(16, 24)
        self.ax1 = self.fig.add_subplot(gs[0:8, 0:8])
        self.ax2 = self.fig.add_subplot(gs[0:4, 8:])
        self.ax3 = self.fig.add_subplot(gs[4:8, 8:])
        self.ax4 = self.fig.add_subplot(gs[8:, 0:8])
        self.ax5 = self.fig.add_subplot(gs[8:, 8:16])
        self.ax6 = self.fig.add_subplot(gs[8:, 16:24])

        # ethogram plot
        num_clusters = np.max(self.cluster)+1
        color_palette = sns.color_palette('rainbow', num_clusters)
        self.cluster_colors = np.array([color_palette[x] for x in self.cluster])

        # density cluster plot
        sns.kdeplot(self.embed[:,0], self.embed[:,1], shade=True, shade_lowest=True, 
                    gridsize=100, levels=40, cmap='jet',cbar=False, ax=self.ax6)
        for clust_i in range(num_clusters):
            self.ax6.annotate(clust_i, self.cluster_mean[clust_i], fontsize=4, fontweight='bold', color='k')
        self.ax6.set(xlabel='Component 1', ylabel='Component 2')

    def create_video(self, start, stop, filepath):
        num_clusters = np.max(self.cluster)+1

        # scatter cluster plot
        samp_frac = 0.5
        num_fr, _ = self.embed.shape
        idx = random.sample(range(num_fr), int(samp_frac*num_fr))

        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=self.fps)
        with writer.saving(self.fig, filepath, dpi=self.dpi):
            for fr_i, fr in enumerate(tqdm(np.arange(start, stop), desc="Frame Loop")):
                # ant plot
                self.ax1.clear()
                self.ax1.set_xlim([-4,4]); self.ax1.set_ylim([-5,8]);
                for shadow_i in range(-10,1):
                    if shadow_i == 0:
                        alpha=0.8
                    else:
                        alpha = 0.1
                    self.ax1.plot(self.bodypoints[fr+shadow_i,0:4,0], self.bodypoints[fr+shadow_i,0:4,1], 
                             c='k', alpha=alpha, 
                             marker="o", markersize=3)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,8:11,0], self.bodypoints[fr+shadow_i,8:11,1], 
                             c='tab:blue', alpha=alpha, 
                             marker="o", markersize=3)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,21:24,0], self.bodypoints[fr+shadow_i,21:24,1], 
                             c='tab:orange', alpha=alpha,
                             marker="o", markersize=3)
                
                # angle plot
                self.ax2.clear()
                self.ax2.set_xlim([start,stop-1]); self.ax2.set_ylim([-1.0,1.0]);
                for i in range(2):
                    self.ax2.plot(np.arange(start, fr+1), self.angles[start:fr+1,i], alpha=0.5, linewidth=1, label=f"ang {i}")
                self.ax2.legend()
                self.ax2.set(xlabel='Frame', ylabel='Normalized Angle')
                
                # ethogram plot
                self.ax3.clear()
                self.ax3.set_xlim([start,stop-1]); self.ax3.set_ylim([-0.5,num_clusters]);
                self.ax3.scatter(np.arange(start, fr+1), self.cluster[start:fr+1], c=self.cluster_colors[start:fr+1], 
                            alpha=1, s=2, marker="s")
                self.ax3.set_yticks(range(0,num_clusters))
                self.ax3.set(xlabel='Frame', ylabel='Ethogram')
                
                # power spectrogram plot
                self.ax4.clear()
                for i in range(2):
                    self.ax4.plot(self.freq, self.power[i,:,fr].T, label=f"ang {i}", alpha=0.5, linewidth=1)
                self.ax4.set_xlabel("freq"); self.ax4.set_ylabel("power")
                self.ax4.set_ylim([-0.05,0.8])
                self.ax4.legend()
                
                # scatter cluster plot
                self.ax5.clear()
                self.ax5.scatter(self.embed[idx,0], self.embed[idx,1], 
                        c=self.cluster_colors[idx], 
                        alpha=0.2, s=0.1)
                self.ax5.plot(self.embed[start:fr+1,0], self.embed[start:fr+1,1],
                        c='k', linewidth=1, alpha=0.5)
                self.ax5.scatter(self.embed[fr,0], self.embed[fr,1],
                        c='k', s=5, marker="x")
                self.ax5.set(xlabel='Component 1', ylabel='Component 2', title=f"frame {fr}", xlim=self.ax6.get_xlim(), ylim=self.ax6.get_ylim())
                
                # take snapshot
                writer.grab_frame()
            writer.grab_frame()
            plt.close()

    def create_video_leg_antennae(self, start, stop, filepath):
        num_clusters = np.max(self.cluster)+1
        
        # scatter cluster plot
        samp_frac = 0.5
        num_fr, _ = self.embed.shape
        idx = random.sample(range(num_fr), int(samp_frac*num_fr))

        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=self.fps)
        with writer.saving(self.fig, filepath, dpi=self.dpi):
            for fr_i, fr in enumerate(tqdm(np.arange(start, stop), desc="Frame Loop")):
                # ant plot
                self.ax1.clear()
                self.ax1.set_xlim([-4,4]); self.ax1.set_ylim([-5,5]);
                for shadow_i in range(-10,1):
                    if shadow_i == 0:
                        alpha=0.8
                    else:
                        alpha = 0.1
                    self.ax1.plot(self.bodypoints[fr+shadow_i,0:4,0], self.bodypoints[fr+shadow_i,0:4,1], 
                             c='k', alpha=alpha, 
                             marker="o", markersize=3)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,8:11,0], self.bodypoints[fr+shadow_i,8:11,1], 
                             c='tab:blue', alpha=alpha, 
                             marker="o", markersize=3)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,21:24,0], self.bodypoints[fr+shadow_i,21:24,1], 
                             c='tab:orange', alpha=alpha,
                             marker="o", markersize=3)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,5:8,0], self.bodypoints[fr+shadow_i,5:8,1], 
                             c='tab:green', alpha=alpha, 
                             marker="o", markersize=3)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,18:21,0], self.bodypoints[fr+shadow_i,18:21,1], 
                             c='tab:red', alpha=alpha,
                             marker="o", markersize=3)
                
                # angle plot
                self.ax2.clear()
                self.ax2.set_xlim([start,stop-1]); self.ax2.set_ylim([-1.0,1.0]);
                for i in range(4):
                    self.ax2.plot(np.arange(start, fr+1), self.angles[start:fr+1,i], alpha=0.5, linewidth=1, label=f"ang {i}")
                self.ax2.legend()
                self.ax2.set(xlabel='Frame', ylabel='Normalized Angle')
                
                # ethogram plot
                self.ax3.clear()
                self.ax3.set_xlim([start,stop-1]); self.ax3.set_ylim([-0.5,num_clusters]);
                self.ax3.scatter(np.arange(start, fr+1), self.cluster[start:fr+1], c=self.cluster_colors[start:fr+1], 
                            alpha=1, s=2, marker="s")
                self.ax3.set_yticks(range(0,num_clusters))
                self.ax3.set(xlabel='Frame', ylabel='Ethogram')
                
                # power spectrogram plot
                self.ax4.clear()
                for i in range(4):
                    self.ax4.plot(self.freq, self.power[i,:,fr].T, label=f"ang {i}", alpha=0.5, linewidth=1)
                self.ax4.set_xlabel("freq"); self.ax4.set_ylabel("power")
                self.ax4.set_ylim([-0.05,0.8])
                self.ax4.legend()
                
                # scatter cluster plot
                self.ax5.clear()
                self.ax5.scatter(self.embed[idx,0], self.embed[idx,1], 
                        c=self.cluster_colors[idx], 
                        alpha=0.2, s=0.1)
                self.ax5.plot(self.embed[start:fr+1,0], self.embed[start:fr+1,1],
                        c='k', linewidth=1, alpha=0.5)
                self.ax5.scatter(self.embed[fr,0], self.embed[fr,1],
                        c='k', s=5, marker="x")
                self.ax5.set(xlabel='Component 1', ylabel='Component 2', title=f"frame {fr}", xlim=self.ax6.get_xlim(), ylim=self.ax6.get_ylim())
                
                # take snapshot
                writer.grab_frame()
            writer.grab_frame()
            plt.close()