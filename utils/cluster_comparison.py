import os
from shutil import copyfile
import pandas as pd
import numpy as np

def create_group_folders(RESULT_PATH, TARGET_DIR_PATH):

    # create TARGET_DIR if it does not already exist
    try:
        os.mkdir(TARGET_DIR_PATH)
    except (FileExistsError):
        print("Unable to create Target directory. Directory already exists.")

    # create a list of all files within given result folder
    all_files = [RESULT_PATH + "/" + f for f in os.listdir(RESULT_PATH)]
    # create an empty list to be filled with only folders (no other files)
    all_folders = []

    # loop through each file, and if it is a directory, add it to the all_folders list
    for file in all_files:
        if os.path.isdir(file):
            all_folders.append(file)
    
    groupings = ["all_embeddings", "all_kinematic_embeddings", "all_postural_embeddings", "angle_power", 
             "angles", "bodypoints", "cluster", "limb_power", "limbs", "rotated_bodypoints", 
             "scaled_bodypoints"]

    # create a directory for each group that will be filled in later
    for group in groupings:
        try:  
            os.mkdir(f"{TARGET_DIR_PATH}/{group}")
        except (FileExistsError):
            print(f"Unable to create {group} directory. Directory already exists.")
    
    # loop through each folder
    for folder in all_folders:
        # save contents of folder to variable contents
        contents = os.listdir(folder)
        # loop through each file within the contents and save only the first part of the filename to group variable
        for file in contents:
            group = file.split(".")[0]
            
            # copy each file to its designated group directory created earlier
            copyfile(f"{folder}/{file}", f"{TARGET_DIR_PATH}/{group}/{group}_{folder.split('/')[-1]}.npy")

def create_normalized_cluster_csv(CLUSTER_PATH, TARGET_DIR_PATH):
    all_clusters = [CLUSTER_PATH + "/" + f for f in os.listdir(CLUSTER_PATH)]

    # find the largest cluster number within cluster data
    def find_max(data):
        max_values = []

        for key, value in data.items():
            max_values.append(max(value))

        return int(max(max_values))
    
    # create a csv of normalized cluster data
    def create_csv(cluster_data, title):
        # create a list of all cluster values within cluster data
        cluster_values = np.arange(0, find_max(cluster_data)+1)

        # create an empty array that will be filled with cluster data
        arr = np.empty((len(cluster_data), len(cluster_values)+1), dtype="object")

        for i, dict in enumerate(cluster_data.items()):
            # key = dict[0]
            # value = dict[1]
            row = np.empty(len(cluster_values)+1, dtype="object")

            # set first row element to be file name
            row[0] = dict[0]

            values = []
        
            for value in dict[1]:
                if value >= 0:
                    values.append(int(value))

            # number of good frames for video = len(values)

            try:
                for j in range(1,len(cluster_values)+1):
                    row[j] = values.count(j-1) / len(values)
            except (ZeroDivisionError):
                print(f"{dict[0]} has 0 good clusters")

            arr[i] = row

        # convert array into dataframe
        DF = pd.DataFrame(arr)
        
        # add mean of each cluster proportion to dataframe
        means = np.empty((1,len(cluster_values)+1), dtype="object")
        means[0][0] = "mean"
        for i in range(1, len(cluster_values)+1):
            means[0][i] = DF[i].mean()  
        means = pd.DataFrame(means)   
        
        # add standard deviation of each cluster proportion to dataframe
        stds = np.empty((1,len(cluster_values)+1), dtype="object")
        stds[0][0] = "standard deviation"
        for i in range(1, len(cluster_values)+1):
            stds[0][i] = DF[i].std()  
        stds = pd.DataFrame(stds)
        
        DF = DF.append(means, ignore_index=True)
        DF = DF.append(stds, ignore_index=True)

        # rename first column to "Video File"
        DF = DF.rename(columns={0: "Video File"})

        # loop through each other column, renaming them to "Cluster {n}"
        for i in range(1,len(cluster_values)+1):
            DF = DF.rename(columns={i: f"cluster{i-1}"})

        # save the dataframe as a csv
        DF.to_csv(f"{TARGET_DIR_PATH}/{title}.csv")

    all_cluster_data = {}

    for file in all_clusters:
        try:
            all_cluster_data[f"{file.split('cluster_')[-1].split('.npy')[0]}"] = np.load(file)
        except (ValueError):
            print("Non-.npy file in directory")

    create_csv(all_cluster_data, title="all_clusters")

    def create_dict_entry(index):
        pre_entry = np.load(all_clusters[index])
        entry = {}
        for index,cluster in enumerate(pre_entry):
            if cluster >= 0:
                entry[index] = cluster
        return entry

    # create an empty dictionary that will contain all cluster data per frame for each video
    cluster_dict = {}

    for index,video_file in enumerate(all_clusters):
        try:
            cluster_dict[video_file.split("cluster_")[-1].split(".npy")[0]] = create_dict_entry(index)
        except (ValueError):
            print("Non-.npy file in directory")
    
    return cluster_dict



