from dbaccess import get_db_entries
from distance import get_distance_metric, create_distance_matrix
from sklearn.cluster import DBSCAN
import numpy as np

# DBSCAN hyper-parameters
EPS = 8 # default: 0.5, max distance
MIN_SAMPLES = 50 # default: 5, min cluster size
DISTANCE_METRIC = 'base_ibi' # Options: 'weight_distribution', 'band_vlf', 'band_lf', 'band_hf', 'base_ibi'

# get data points from database
data = get_db_entries()

# get distance metric
distance_metric = get_distance_metric(DISTANCE_METRIC)

# pre calculate distances
distance_matrix = create_distance_matrix(data=data, distance_metric=distance_metric)

# run clustering
dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='precomputed')
labels = dbscan.fit_predict(distance_matrix)

# Count the number of clusters (excluding noise points)
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of clusters found: {num_clusters}")

# Total number of data points
num_data_points = len(data)
print(f"Total number of data points: {num_data_points}")

# Size of each cluster
if num_clusters > 0:
    cluster_sizes = {label: np.sum(labels == label) for label in set(labels) if label != -1}
    for label, size in cluster_sizes.items():
        print(f"Cluster {label}: {size} points")
else:
    print("No clusters found.")

# Dictionary to store data points in each cluster
cluster_data_points = {label: [] for label in set(labels)}
for i, label in enumerate(labels):
    if label != -1:
        cluster_data_points[label].append(data[i])

# Print example cluster
print('\nIBI combinations of cluster 0:')
print([ (d['dyad_parameters']['base_ibi_adult'], d['dyad_parameters']['base_ibi_infant']) for d in cluster_data_points[1] ] )
