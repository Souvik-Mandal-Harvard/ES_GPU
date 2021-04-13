import numpy as np
import matplotlib, random, yaml, sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tqdm import tqdm
import skvideo.io
from glob import glob

from utils.data import Dataset

def main():
    config_name = sys.argv[1]
    with open(config_name) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    PROJECT_PATH = config['GPU_project_path']
    config_path = f"{PROJECT_PATH}/{config_name}"
    num_videos_per_clusters = 9
    video_duration = 200 # frames

    Data = Dataset(PROJECT_PATH, config_path)
    Data.load_data()

    # configuration
    INFO = Data.info
    INFO_values = Data.info_values

    # config = Data.config
    skeleton = config['skeleton']
    skeleton_color= config['skeleton_color']
    skeleton_fill = config['skeleton_fill']
    
    # features
    #tot_bp = Data.data_obj['rotated_bodypoints']
    tot_bp = Data.data_obj['scaled_bodypoints']
    # embeddings
    tot_embed = Data.data_obj['all_embeddings']
    # cluster
    tot_clusters = Data.data_obj['cluster']
    num_clusters = int(np.max(tot_clusters))+1

    # Determine Which frames to Abstract
    video_cluster_idx = {}
    for clust_i in range(num_clusters):
        clust_idx = np.where(tot_clusters==clust_i)[0]
        difference = np.diff(clust_idx)

        # Find consecutive break
        break_idx = np.where(difference != 1)[0]
        mod_break_idx = np.insert(break_idx, 0, 0)
        break_difference = np.diff(mod_break_idx)

        # Find max consecutive
        sorted_idx = np.argsort(break_difference)
        top_idx = sorted_idx[-num_videos_per_clusters:]
        video_idx = np.array([[ clust_idx[mod_break_idx[idx]+1], clust_idx[mod_break_idx[idx+1]+1]] for idx in top_idx])
        video_cluster_idx[clust_i] = video_idx
    
    global_start_frames = np.array([val['global_start_fr'] for val in INFO_values])
    global_stop_frames = np.array([val['global_stop_fr'] for val in INFO_values])
    global_directories = np.array([val['directory'] for val in INFO_values])
    
    for clust_i in range(0, num_clusters):
        # Create video
        fig, ax = plt.subplots(3,3,figsize=(10,10))
        fig.suptitle(f"Cluster {clust_i} Sample Videos")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # # animal video data
        # video_i, file_start_fr = {}, {}
        # for i, (start, stop) in enumerate(tqdm(video_cluster_idx[clust_i], desc="Collecting Videos")):
        #     file_bool = start > global_start_frames
        #     if any(file_bool):
        #         file_start_fr[i] = global_start_frames[file_bool][-1]
        #         file_path = global_directories[file_bool][-1]
        #         file_key = file_path.split("/")[-1]
        #         video_path = glob(f"{VIDEO_PATH}/{file_key}.avi")[0]
        #         video = skvideo.io.vread(video_path)

        #         video_start = start-file_start_fr[i]
        #         video_stop = video_start+video_duration
        #         if video_stop < len(video):
        #             video_i[i] = video[video_start:video_stop]
        #         else:
        #             video_i[i] = video[video_start:]
        #     else:
        #         return # don't create a video

        # video format        
        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=25)
        SAVE_PATH=f"videos/task1_etho/mutivideo_cluster{clust_i}.mp4"
        with writer.saving(fig, SAVE_PATH, dpi=300):
            for fr_i in tqdm(np.arange(0, video_duration), desc=f"Cluster {clust_i} Frame Loop"):
                for i, (start, stop) in enumerate(video_cluster_idx[clust_i]):
                    fr = start+fr_i

                    # configure plot
                    ax[i//3,i%3].clear()
                    ax[i//3,i%3].set_axis_off()
                    ax[i//3,i%3].set(title=f"Cluster {int(tot_clusters[fr])}")
                    ax[i//3,i%3].set(xlim=(-3,3), ylim=(-3,3))

                    # ax[i//3,i%3].imshow(video_i[i][fr_i])

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
