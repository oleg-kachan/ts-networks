# ts-networks

Connectivity modeling and feature extraction of the time series networks.

The library currently provides the following functionality:

#### Connectivity modeling
- Pearson correlation (via maximum likelihood and the Ledoit-Wolf shrinkage estimators)
- Gaussian kernel
- mutual information (via Kozachenko-Leonenko estimator)

#### Feature extraction

Information-theoretic
- entropy of the adjacency and Laplacian matrices of the network

Graph-theoretic
- degrees (vertex and average of the vertex neighborhood)
- centralities (betwenneess and closeness)
- clustering coefficient

Spectral graph-theoretic
- eigenvalues of the adjacency and the Laplacian matrices of the network

Persistent topological
- persistence diagram
- persistence statistics (total persistence and persistence entropy)

## Installation

Clone the repository and put `tsnetworks` library to the folder with your scripts. Then, the library can be loaded as:

```python
from tsnetworks load Network
```

### Dependendies

- numpy
- scipy
- scikit-learn
- nilearn
- networkx
- dionysus

## Example of usage

```python
import numpy as np
from ts_networks import Network

# generate an n x t time series matrix
n, t = 32, 1000
X = np.random.normal(size=(n, t))

# intialize a network from time series
network = Network(X).correlation()

# compute the entropy of the network's connectivity matrix
entropy = network.entropy()

# compute the network's vertex degrees
degrees = network.degrees()

# compute the persistence diagram of the network
persistence_diagram = network.persistence().diagram 
```
