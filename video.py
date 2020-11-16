import numpy as np
import matplotlib, random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tqdm.notebook import tqdm
import skvideo.io

class Video:
    def __init__(self, config, INFO, bodypoints, angles, power, embed, cluster, fps=25):
        self.config = config
        self.INFO = INFO
        
        self.bodypoints = bodypoints
        self.angles = angles
        self.power = power
        self.embed = embed
        self.cluster = cluster.astype(int)
        
        self.fps = fps
        self.num_fr, self.num_angles, _ = angles.shape
        self.freq = np.array(list(INFO.values())[0]['frequencies'])
        self.num_freq = len(self.freq)
        self.num_clusters = int(np.max(self.cluster)+1)
        
        self.outlier_pts = (cluster == -1)
        self.labeled_pts = (cluster != -1)
                
        SMALL_SIZE = 8
        matplotlib.rc('font', size=SMALL_SIZE)
        matplotlib.rc('axes', titlesize=SMALL_SIZE)

    def setup(self):
        # video info
        self.fig = plt.figure(constrained_layout=True, figsize=(15,10))
        gs = self.fig.add_gridspec(24, 36)
        self.ax1 = self.fig.add_subplot(gs[0:12, 0:12])
        self.ax2 = self.fig.add_subplot(gs[0:6, 12:])
        self.ax3 = self.fig.add_subplot(gs[6:12, 12:])
        self.ax4 = self.fig.add_subplot(gs[12:, 0:12])
        self.ax5 = self.fig.add_subplot(gs[12:, 12:24])
        self.ax6 = self.fig.add_subplot(gs[12:, 24:36])
    
        # angle palette
        self.ang_palette = sns.color_palette('tab10', 10)
        
        # cluster palette
        color_palette = sns.color_palette('rainbow', int(self.num_clusters))
        self.cluster_colors = np.array([color_palette[x] for x in self.cluster])
        
        # cluster mean
        self.cluster_mean = np.zeros((self.num_clusters, 2))
        for clust_i in range(self.num_clusters):
            self.cluster_mean[clust_i,:] = np.mean(self.embed[clusterer.labels_==clust_i, :-1], axis=0)
        
        # ax6: density cluster plot
        samp_idx = random.sample(range(self.num_fr), int(self.num_fr*1/20))
        sns.kdeplot(x=self.embed[samp_idx ,0], y=self.embed[samp_idx ,1], 
            shade=True, gridsize=100, 
            levels=40, thresh=0,
            cmap='icefire',
            cbar=False, ax=self.ax6)
        for clust_i in range(self.num_clusters):
            self.ax6.annotate(clust_i, self.cluster_mean[clust_i], fontsize=6, fontweight='bold', color='white')
        self.ax6.set(xlabel='Component 1', ylabel='Component 2')


    def create_video_legs(self, start, stop, save_path): 
        # find video filename
        INFO_values = list(self.INFO.values())
        INFO_values.sort(key=lambda x: x['order'])
        
        global_start_frames = np.array([val['global_start_fr'] for val in INFO_values])
        global_stop_frames = np.array([val['global_stop_fr'] for val in INFO_values])
        global_directories = np.array([val['directory'] for val in INFO_values])
        
        # animal video data
        file_bool = [a and b for a, b in zip(start >= global_start_frames, stop < global_stop_frames)]
        if any(file_bool):
            file_start_fr = global_start_frames[file_bool].item()
            file_path = global_directories[file_bool].item()
            print(file_path)
            file_key = file_path.split("/")[-1]
            # video_path = glob(f"{file_path}/*.avi")[0]
            video_path = glob(f"/home/murthyhacker/dong/Ant_Videos/ant_field_round2/{file_key}.avi")[0]
            video = skvideo.io.vread(video_path)
        else:
            return # don't create a video
        
        # video format        
        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=self.fps)
        with writer.saving(self.fig, save_path, dpi=700):
            for fr_i, fr in enumerate(tqdm(np.arange(start, stop), desc="Frame Loop")):
                # ax1: bodypoints
                self.ax1.clear()
                self.ax1.set(xlabel=file_key)
                # self.ax1.set_xlim([-5,5]); self.ax1.set_ylim([-5,5]);
                # for shadow_i in range(-10,1):
                for shadow_i in range(0,1):
                    if shadow_i == 0: alpha=0.9
                    else: alpha = 0.1
                    # video frame
                    self.ax1.imshow(video[fr-file_start_fr])
                    
                    # left side
                    bp_linewidth = 2
                    bp_markersize = 3
                    self.ax1.plot(self.bodypoints[fr+shadow_i,0:4,0], self.bodypoints[fr+shadow_i,0:4,1], 
                             c='w', alpha=alpha, 
                             marker="o", linewidth=bp_linewidth, markersize=bp_markersize)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,5:8,0], self.bodypoints[fr+shadow_i,5:8,1], 
                             c=self.ang_palette[0], alpha=alpha, 
                             marker="o", linewidth=bp_linewidth, markersize=bp_markersize)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,8:11,0], self.bodypoints[fr+shadow_i,8:11,1], 
                             c=self.ang_palette[1], alpha=alpha, 
                             marker="o", linewidth=bp_linewidth, markersize=bp_markersize)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,11:14,0], self.bodypoints[fr+shadow_i,11:14,1], 
                             c=self.ang_palette[2], alpha=alpha, 
                             marker="o", linewidth=bp_linewidth, markersize=bp_markersize)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,14:17,0], self.bodypoints[fr+shadow_i,14:17,1], 
                             c=self.ang_palette[3], alpha=alpha, 
                             marker="o", linewidth=bp_linewidth, markersize=bp_markersize)
                    # right side
                    self.ax1.plot(self.bodypoints[fr+shadow_i,18:21,0], self.bodypoints[fr+shadow_i,18:21,1], 
                             c=self.ang_palette[4], alpha=alpha,
                             marker="o", linewidth=bp_linewidth, markersize=bp_markersize)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,21:24,0], self.bodypoints[fr+shadow_i,21:24,1], 
                             c=self.ang_palette[5], alpha=alpha,
                             marker="o", linewidth=bp_linewidth, markersize=bp_markersize)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,24:27,0], self.bodypoints[fr+shadow_i,24:27,1], 
                             c=self.ang_palette[6], alpha=alpha,
                             marker="o", linewidth=bp_linewidth, markersize=bp_markersize)
                    self.ax1.plot(self.bodypoints[fr+shadow_i,27:30,0], self.bodypoints[fr+shadow_i,27:30,1], 
                             c=self.ang_palette[7], alpha=alpha,
                             marker="o", linewidth=bp_linewidth, markersize=bp_markersize)
                    
                
                # ax2: angle
                self.ax2.clear()
                self.ax2.set(xlim = [start-file_start_fr, stop-file_start_fr-1], ylim=[-1.1,1.1], 
                             ylabel='Normalized Angle', xlabel=f"frame: {fr-file_start_fr}");
                for i in range(self.num_angles):
                    self.ax2.plot(np.arange(start-file_start_fr, fr-file_start_fr+1), self.angles[start:fr+1,i,0], alpha=0.5, 
                                  linewidth=2, label=f"ang {i}", c=self.ang_palette[i])
                
                # ax3: ethogram
                self.ax3.clear()
                self.ax3.set(yticks=range(-1,num_clusters,3), 
                             xlim=[start-file_start_fr, stop-file_start_fr-1], ylim=[-1.5,num_clusters], 
                             xlabel=f"Cluster: {self.cluster[fr]}", ylabel='Ethogram')
                self.ax3.scatter(np.arange(start-file_start_fr, fr-file_start_fr+1), self.cluster[start:fr+1], c=self.cluster_colors[start:fr+1], 
                            alpha=1, s=3, marker="s")
                
                # ax4: power spectrogram
                xtick_idx = np.arange(1, self.num_freq, 3).astype(int)
                self.ax4.clear()
                self.ax4.imshow(self.power[:,:-1,fr], norm = matplotlib.colors.LogNorm(), 
                              vmin=0.01, vmax=np.max(self.power[:,:-1,:]), cmap='hot')
                self.ax4.set(
                    xticks=xtick_idx, 
                    xticklabels=np.around(self.freq[xtick_idx], 1), 
                    yticks=np.arange(len(self.config['angle_labels'])), 
                    yticklabels=self.config['angle_labels'],
                    aspect=2, 
                    xlabel="Frequency")
                
                # ax5: scatter cluster                  
                self.ax5.clear()       
                self.ax5.scatter(self.embed[self.outlier_pts,0], self.embed[self.outlier_pts,1], 
                           color="gray", edgecolor=None, s=2, alpha=0.05)
                self.ax5.scatter(self.embed[self.labeled_pts,0], self.embed[self.labeled_pts,1], 
                           color=self.cluster_colors[self.labeled_pts], edgecolor=None, s=2, alpha=0.05)

                if self.embed[fr,0] is not np.nan:
                    self.ax5.scatter(self.embed[fr,0], self.embed[fr,1], c='k', s=50, marker="x")
                self.ax5.set(xlabel='Component 1', ylabel='Component 2', xlim=self.ax6.get_xlim(), ylim=self.ax6.get_ylim())
                
                # take snapshot
                writer.grab_frame()
                
            writer.grab_frame()
            plt.close()
def main():
    result_path = "results/round2_legs_antennae"

    # Load Config Files
    with open(f"{result_path}/INFO.yaml") as f:
        INFO = yaml.load(f, Loader=yaml.FullLoader)
        INFO_values = list(INFO.values())
        INFO_values.sort(key=lambda x: x['order'])  
    config_path = "."
    with open(f"{config_path}/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Format Data
    tot_bp, tot_bp_scaled, tot_angles, tot_rotations, tot_power, tot_embed, tot_clusters = [], [], [], [], [], [], []
    for file in tqdm(INFO_values):
        tot_bp.append( np.load(f"{file['directory']}/bodypoints.npy") )
        tot_bp_scaled.append( np.load(f"{file['directory']}/scaled_bodypoints.npy") )
        tot_angles.append( np.load(f"{file['directory']}/angles.npy") )
        tot_power.append( np.load(f"{file['directory']}/power.npy") )
        tot_embed.append( np.load(f"{file['directory']}/embeddings.npy") )
        tot_clusters.append( np.load(f"{file['directory']}/clusters.npy") )
    tot_bp = np.concatenate(tot_bp)
    tot_bp_scaled = np.concatenate(tot_bp_scaled)
    tot_angles = np.concatenate(tot_angles)
    tot_power = np.concatenate(tot_power, axis=2)
    tot_embed = np.concatenate(tot_embed)
    tot_clusters = np.concatenate(tot_clusters)

    # Create Video Class
    video_creator = Video(
        config=config,
        INFO=INFO,
        bodypoints=tot_bp, 
        angles=tot_angles, 
        power=tot_power,
        embed=tot_embed, 
        cluster=tot_clusters[:,0]
    )

    # Setup Video
    video_creator.setup()

    # Determine Which frames to Abstract
    num_clusters = np.max(tot_clusters[:,0]) + 1
    cluster_distribution = []
    for clust_i in range(num_clusters):
        clust_idx = np.where(tot_clusters[:,0] == clust_i)[0]
        cluster_distribution.append(len(clust_idx))

    video_cluster_idx = {}
    for clust_i in range(num_clusters):
        clust_idx = np.where(tot_clusters[:,0] ==clust_i)[0]
        difference = np.diff(clust_idx)
        
        # Find consecutive break
        break_idx = np.where(difference != 1)[0]
        mod_break_idx = np.insert(break_idx, 0, 0)
        break_difference = np.diff(mod_break_idx)
        
        # Find max consecutive
        sorted_idx = np.argsort(break_difference)
        top_idx = sorted_idx[-config['num_sample_videos']:]
        video_idx = [[ clust_idx[mod_break_idx[idx]+1], clust_idx[mod_break_idx[idx+1]+1]] for idx in top_idx]
        video_cluster_idx[clust_i] = video_idx

    # Create Videos
    for clust_i, list_idx in tqdm(video_cluster_idx.items(), desc="Cluster Loop"):
        for (start_fr, stop_fr) in tqdm(list_idx, desc="Video Loop"):
            # Set Frame Range
            fr_length = stop_fr-start_fr 
            if fr_length > config["max_video_length"]:
                start, stop = start_fr, start_fr+config["max_video_length"]
            else:
                pad = int(fr_length/2)
                start, stop = start_fr-pad, stop_fr+pad
            
            # Define Filename
            filepath = f"{save_video_path}/cluster{clust_i}_frame{start}-{stop}.mp4"
            
            # Create Video
            video_creator.create_video_legs(start, stop, filepath)

if __name__ == "__main__":
    main()