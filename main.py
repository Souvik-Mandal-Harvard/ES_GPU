import random
import numpy as np
import pandas as pd
from glob2 import glob
from tqdm import tqdm

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import umap
from scipy.signal import morlet2, cwt, ricker
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder

# constants
FPS = 50

KEYS = [{"a": 2, "b": 9, "c": 8},
        {"a": 2, "b": 22, "c": 21}]

interp_samp = 1
interp_fps = fps*interp_samp
w = 5

bp_list, angles_list, power_list = [], [], []

for path in tqdm(glob("data/clean_data/**/*.h5")):
	print(path)