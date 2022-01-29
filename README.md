#- import basic python packages
import warnings
import tkinter # to show plot in Pycharm

#- import packages for data manipulations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE

#- import packages for unsupervised machine learning
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
