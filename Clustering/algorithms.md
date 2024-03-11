# Clustering algorithms for genetic optimization results

## Data characteristics

Each data point consits of hyperparameters und dyad ibi/hrv parameters.

### Hyperparameters

The Hyperparameters are the parameter that control how exactly the genetic optimization algorithm is working. The are:   

- POPULATION_SIZE
- MAX_NUM_GENERATIONS
- DISTANCE_METRIC
- CROSSOVER_METHOD
- MUTATION_RATE
- MUTATION_SCALE
- SELECT_PARENTS_METHOD
- PARENT_RATIO
- TARGET_ZLC
- STOP_ON_CONVERGENCE
- CONVERGENCE_N

These should not be included in the clustering process but should be kept as metainformation for analyzing found clusters in the end. It is possible that clusters only occur due to similar or equal hyperparameters.

### Dyad parameters / individuals

There are 96 dyad parameters in each individual along with a fitness score. The dyad parameters are parameters for creating a dyad IBI sequence using the HRV frequency based IBI generator in this package. The individual parameters are:

- base IBI
- VLF: data for 4 very low frequencies as triples (frequency, phase shift, weight)
- LF: data for 6 low frequencies as triples (frequency, phase shift, weight)
- HF: data for 6 high frequencies as triples (frequency, phase shift, weight)

Each of these parameters is given both for the infant and the adult in the dyad.

The fitness score indicates how close the zero-lag coefficient of the dyad is to the target ZLC when running the dyads IBI data through the RSA algorithm.

## Cluster characteristics

Clusters are groups of data points with low distances between each other. Working in a high-dimensional space (96 dimensions or 108 if fitness and hyperparameters are included), a meaningful definition of distance is cricial. Specifically, points don't need to be close in all dimensions to be considered close, closeness in certain combinations of dimensions is what we are interested in. For example:

- Closeness in any combination of VLFs: similar frequencies, weights, phases in VLFs, independent of frequency indices [same for LF/HF]
- Weight distribution over frequency bands: for example high LF influence, low HF and VLF influence
- Base IBI similarities: the difference between adult and infant IBIs or their absolute values are similar in multiple indivuduals

Combinations of these types of closeness and further variations are possible.

## Algorithm choices

The cluster characteristics and domain knowledge are used to:

- **pre-process** the data and to create different data sets of reduced dimensionality for finding clusters
- create **custom distance metrics** that quantify closeness of data points in a meaningful way

The algorithm **DBSCAN** (density-based spatial clustering of applications with noise) has a couple of advantages:

- noisy data is suported
- custom distance metrics are possible
- the number of resulting clusters does not need to be fixed but is based on density