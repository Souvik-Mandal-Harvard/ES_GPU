import numpy as np
import yaml, pickle, tqdm
from os import path

class Dataset:
    def __init__(self, project_path, config_path):
        self.project_path = project_path
        self.config_path = config_path

        self.data_name = ['bodypoints','scaled_bodypoints', 'rotated_bodypoints', 'angles', 'limbs', 'angle_power', 'limb_power', 
        'all_embeddings', 'all_postural_embeddings', 'maker_postural_embeddings', 'angle_postural_embeddings', 'limb_postural_embeddings',
        'all_kinematic_embeddings', 'marker_kinematic_embeddings', 'limb_kinematic_embeddings', 'angle_kinematic_embeddings', 'cluster', 'kinematic_cluster']
        self.data_obj = {}
        
        self.config = self.load_config()
        self.info, self.info_values = self.load_info()

    def load_info(self):
        print("Loading INFO.yaml ...")
        with open(f"{self.project_path}/{self.config['result_path']}/INFO.yaml") as f:
            INFO = yaml.load(f, Loader=yaml.FullLoader)
            INFO_values = list(INFO.values())
            INFO_values.sort(key=lambda x: x['order'])
        print("Finished loading INFO")
        return INFO, INFO_values
        
    def load_config(self):
        print("Loading config.yaml ...")
        with open(f"{self.config_path}") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print("Finished loading config")
        return config

    def load_data(self):
        for file_name in self.data_name:
            self.data_obj[file_name] = []
        for file in tqdm.tqdm(self.info_values):
            for file_name in self.data_name:
                abs_data_path = f"{self.project_path}/{file['directory']}/{file_name}.npy"
                if path.exists(abs_data_path):
                    self.data_obj[file_name].append( np.load(abs_data_path) )
        for file_name in self.data_name:
            if self.data_obj[file_name]:
                print(len(self.data_obj[file_name]))
                print(len(self.data_obj[file_name][0]))
                print(len(self.data_obj[file_name][0][0]))
                self.data_obj[file_name] = np.concatenate(self.data_obj[file_name])
