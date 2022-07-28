# ts-networks

Connectivity modeling and feature extraction of time series networks.

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
