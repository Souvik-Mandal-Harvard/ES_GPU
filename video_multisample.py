import numpy as np
import matplotlib, random, yaml, sys, os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tqdm import tqdm
import skvideo.io
from glob import glob
import itertools,operator

from utils.data import Dataset

def main():
    config_name = sys.argv[1]
    
    # load configuration and data
    with open(config_name) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    PROJECT_PATH = config['GPU_project_path']
    config_path = f"{PROJECT_PATH}/{config_name}"
    Data = Dataset(PROJECT_PATH, config_path)
    Data.load_data()

    # configuration
    INFO = Data.info
    INFO_values = Data.info_values
    skeleton = config['skeleton']
    skeleton_color= config['skeleton_color']
    skeleton_fill = config['skeleton_fill']
    num_videos_per_clusters = config['num_sample_videos']
    video_duration =  config['max_video_length']
    video_type = config['video_type']
    
    # bodypoints
    if video_type == 0:
        tot_bp = Data.data_obj['bodypoints']
    elif video_type == 1:
        tot_bp = Data.data_obj['rotated_bodypoints']
    elif video_type == 2:
        tot_bp = Data.data_obj['scaled_bodypoints']
    # embeddings
    tot_embed = Data.data_obj['all_embeddings']
    # cluster
    tot_clusters = Data.data_obj['cluster']
    num_clusters = int(np.max(tot_clusters))+1

    # Determine Which frames to Abstract
    video_cluster_idx = {}
    for clust_i in range(num_clusters):
        sorted_list_idx = sorted((list(y) for (x,y) in itertools.groupby((enumerate(tot_clusters)),operator.itemgetter(1)) if x == clust_i), key=len, reverse=True)
        top_start_stop_idx = map(lambda x: [x[0][0], x[-1][0]], sorted_list_idx[0:num_videos_per_clusters])
        video_cluster_idx[clust_i] = np.array(list(top_start_stop_idx))
    
    global_start_frames = np.array([val['global_start_fr'] for val in INFO_values])
    global_stop_frames = np.array([val['global_stop_fr'] for val in INFO_values])
    global_directories = np.array([val['directory'] for val in INFO_values])
    
    for clust_i in range(0, num_clusters):
        # Create video
        fig, ax = plt.subplots(3,3,figsize=(10,10))
        fig.suptitle(f"Cluster {clust_i} Sample Videos")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # animal video data
        if video_type == 0:
            video_i, file_start_fr = {}, {}
            for i, (start, stop) in enumerate(tqdm(video_cluster_idx[clust_i], desc="Collecting Videos")):
                file_bool = start > global_start_frames
                if any(file_bool):
                    file_start_fr[i] = global_start_frames[file_bool][-1]
                    file_path = global_directories[file_bool][-1]
                    file_key = file_path.split("/")[-1]
                    video_path = glob(f"{config['raw_video_path']}/{file_key}.avi")[0]
                    video = skvideo.io.vread(video_path)

                    video_start = start-file_start_fr[i]
                    video_stop = video_start+video_duration
                    if video_stop < len(video):
                        video_i[i] = video[video_start:video_stop]
                    else:
                        video_i[i] = video[video_start:]
                else:
                    return # don't create a video

        # video format        
        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=25)
        SAVE_PATH=f"{config['save_video_path']}/mutivideo_cluster{clust_i}.mp4"
        if not os.path.exists(f"{config['save_video_path']}"):
            os.makedirs(f"{config['save_video_path']}")
        with writer.saving(fig, SAVE_PATH, dpi=300):
            for fr_i in tqdm(np.arange(0, video_duration), desc=f"Cluster {clust_i} Frame Loop"):
                for i, (start, stop) in enumerate(video_cluster_idx[clust_i]):
                    fr = start+fr_i

                    # configure plot
                    ax[i//3,i%3].clear()
                    ax[i//3,i%3].set_axis_off()
                    ax[i//3,i%3].set(title=f"Cluster {int(tot_clusters[fr])}")

                    if (video_type==1) | (video_type==2):
                        ax[i//3,i%3].set(xlim=(-3,3), ylim=(-3,3))
                    if video_type==0:
                        ax[i//3,i%3].imshow(video_i[i][fr_i])
                    for skeleton_i, color_i in zip(skeleton, skeleton_color):
                        ax[i//3,i%3].plot(tot_bp[fr,skeleton_i,0], tot_bp[fr,skeleton_i,1], marker="o", markersize=2,
                            linewidth=2, alpha=0.6, c=color_i)
                    for fill_obj in skeleton_fill:   
                        ax[i//3,i%3].add_patch(matplotlib.patches.Polygon(xy=tot_bp[fr,fill_obj['trapezoid'],0:2], fill=True, 
                            alpha=0.7, color=fill_obj['fill']))
                writer.grab_frame()
            plt.close()

if __name__ == "__main__":
    main()
