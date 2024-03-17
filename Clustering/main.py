from dbaccess import get_db_entries
from distance import get_distance_metric
from sklearn.cluster import DBSCAN
import numpy as np

# DBSCAN hyper-parameters
EPS = .5
MIN_SAMPLES = 5
DISTANCE_METRIC = 'base_ibi' # Options: 'weight_distribution', 'band_vlf', 'band_lf', 'band_hf', 'base_ibi'

# get data points from database
data = get_db_entries()

# pre-process data based on metric (TODO)
data_preprocessed = np.array(data)

# get distance metric
distance_metric = get_distance_metric(DISTANCE_METRIC)

# run clustering
clusters = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric=distance_metric).fit(data_preprocessed)

# export results (TODO)
print(clusters.labels_)