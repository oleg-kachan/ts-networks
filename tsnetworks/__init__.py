import numpy as np
import networkx as nx

from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import rbf_kernel

from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma, digamma
from scipy.spatial import KDTree

from scipy.stats import entropy

import dionysus
from scipy.spatial.distance import pdist, squareform


class Network():
    """
    Time series network class.

    Create a time series network via passing an n x t matrix of time series.

    Parameters
    ----------
    time_series : ndarray of shape (n, t)
        Numpy array of time series with n instances and t time steps.

    Examples
    --------
    >>> import numpy as np
    >>> from ts_networks import Network
    >>>
    >>> n, t = 32, 1000
    >>> X = np.random.normal(size=(n, t))
    >>>
    >>> network = Network(X).correlation()
    """

    def __init__(self, time_series):
        self.time_series = time_series
        self.adjacency = None

    def __repr__(self):
        return np.array_repr(self.adjacency.matrix)

    def __init_graph(self):
        adjacency_no_loops = np.copy(self.adjacency.matrix)
        np.fill_diagonal(adjacency_no_loops, 0)
        self.graph = nx.from_numpy_array(adjacency_no_loops)

    def correlation(self, estimator="maximum_likelihood", assume_centered=True, threshold=None, absolute=True):
        """
        Set network connectivity matrix to 1 - |correlation|.

        Parameters
        ----------
        estimator : str [ 'maximum_likelihood' | 'ledoit_wolf' ]
            Covariance estimator
              - 'maximum_likelihood' : emprical covariance estimator
              - 'ledoit_wolf' : Ledoit-Wolf covariance estimator

        Returns
        -------
        self : Network object
            Returns the Network instance.

        References
        ----------
        "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices",
        Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2,
        February 2004, pages 365-411.
        """

        if estimator=="maximum_likelihood":
            cm = ConnectivityMeasure(kind="correlation", cov_estimator=EmpiricalCovariance(assume_centered=assume_centered))
        elif estimator=="ledoit_wolf":
            cm = ConnectivityMeasure(kind="correlation", cov_estimator=LedoitWolf(assume_centered=assume_centered))
        else:
            raise ValueError("Estimator should be 'maximum_likelihood' or 'ledoit_wolf'")

        R = cm.fit_transform(np.expand_dims(self.time_series.T, axis=0))[0]

        if threshold is not None:
            threshold_idx = R < threshold
            R[threshold_idx] = 0

        if absolute==True:
            R_inv = 1 - np.abs(R)
            np.fill_diagonal(R_inv, 0)
            R = R_inv

        self.adjacency = Matrix(R)
        self.__init_graph()

        return self

    def gaussian_kernel(self, threshold=None):
        """
        Set network connectivity matrix to the Gaussian kernel k = exp(-gamma(|x_i, x_j|^2))
        measuring the similarity between instances.

        Parameters
        ----------
        None

        Returns
        -------
        self : Network object
            Returns the Network instance.

        Examples
        --------
        >>> import numpy as np
        >>> from ts_networks import Network
        >>>
        >>> n, t = 32, 1000
        >>> X = np.random.normal(size=(n, t))
        >>>
        >>> network = Network(X).gaussian_kernel()

        References
        ----------
        Wang, L., Zhang, J., Zhou, L., Tang, C., Li, W.: Beyond covariance: Feature repre-
        sentation with nonlinear kernel matrices. In: Proceedings of the IEEE International
        Conference on Computer Vision. pp. 4570-4578 (2015).
        """
        
        K = rbf_kernel(self.time_series)

        if threshold is not None:
            threshold_idx = K < threshold
            K[threshold_idx] = 0

        np.fill_diagonal(K, 0)
        self.adjacency = Matrix(K)
        self.__init_graph()
        return self

    def mutual_information(self, n_neighbors=15, threshold=None):
        """
        Set network connectivity matrix to the mutual information
        measuring the similarity between instances.

        Parameters
        ----------
        n_neighbors: int, default=15
            Number of k-nearest neighbors for the Kozachenko-Leonenko
            entropy estimator used to estimate mutual information.

        Returns
        -------
        self : Network object
            Returns the Network instance.

        Examples
        --------
        >>> import numpy as np
        >>> from ts_networks import Network
        >>>
        >>> n, t = 32, 1000
        >>> X = np.random.normal(size=(n, t))
        >>>
        >>> network = Network(X).mutual_information(n_neighbors=20)

        References
        ----------
        Linfoot, E.H.: An informational measure of correlation.
        Information and control 1(1), 85-89 (1957)

        Kozachenko, L., Leonenko, N.: Sample estimate of the entropy of a random vector.
        Problemy Peredachi Informatsii 23(2), 9-16 (1987)
        """

        def vol(d):
            numerator = np.pi ** (d/2)
            denominator = gamma((d/2) + 1)
            return numerator / denominator

        def kl(X, n_neighbors=15):
            n, d = X.shape
            distances, _ = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X).kneighbors(X)
            distances_k = distances[:,-1]
            
            return d * np.mean(np.log(distances_k)) + digamma(n) - digamma(n_neighbors) + np.log(vol(d))

        def I(P, n_neighbors=15):

            n, d = P.shape
            
            H_P = kl(P, n_neighbors)
            
            H_marginal = np.zeros(d)
            for i, idx in enumerate(combinations(range(d), 1)):
                P_marginal_i = P[:,idx]
                H_marginal[i] = kl(P_marginal_i, n_neighbors)

            return np.sum(H_marginal) - H_P

        n, _ = self.time_series.shape
        A = np.zeros((n, n))

        for edge in list(combinations(range(n), 2)):
            A[edge] = I(self.time_series.T[:,edge], n_neighbors)

        A = A + A.T

        if threshold is not None:
            threshold_idx = A < threshold
            A[threshold_idx] = 0
            
        self.adjacency = Matrix(A)
        self.__init_graph()

        return self

    def entropy(self):
        """
        Entropy, an information-theoretic feature of the network's connectivity matrix.

        Parameters
        ----------
        None

        Returns
        -------
        entropy : float
            Returns the network's connectivity matrix entropy.

        Examples
        --------
        >>> import numpy as np
        >>> from ts_networks import Network
        >>>
        >>> n, t = 32, 1000
        >>> X = np.random.normal(size=(n, t))
        >>>
        >>> network = Network(X).mutual_information(n_neighbors=20)
        >>> network.entropy()

        References
        ----------
        Cover, T., Thomas, J.: Elements of information theory. Wiley (1991)
        """

        return self.adjacency.entropy()

    def degree(self, kind="vertex"):
        """
        Degree, a graph-theoretic feature of the network's vertices.
        A number of edges adjacent to given vertex.

        Parameters
        ----------
        kind : string [ 'vertex' | 'average_neighbor' ], default='vertex'
            Degree to compute
              - 'vertex' : each vertex degree
              - 'average_neighbor' : each vertex average neighbors degree

        Returns
        -------
        degree : ndarray of shape (n,)
            Returns the network's degree vector of shape n.

        Examples
        --------
        >>> import numpy as np
        >>> from ts_networks import Network
        >>>
        >>> n, t = 32, 1000
        >>> X = np.random.normal(size=(n, t))
        >>>
        >>> network = Network(X).mutual_information(n_neighbors=20)
        >>> network.degree()
        >>> network.degree(kind="average_neighbor")

        References
        ----------
        Newman, M.: Networks. Oxford university press (2018)
        """
        if kind=="vertex":
            degree = np.array(nx.degree(self.graph, weight="weight"))[:,1]
        elif kind=="average_neighbor":
            degree = np.array(list(nx.average_neighbor_degree(self.graph, weight="weight").values()))
        else:
            raise ValueError("Degree kind should be 'vertex' or 'average_neighbor'.")

        return degree

    def centrality(self, kind="betweenness"):
        """
        Centrality, a graph-theoretic feature of the network's vertices.

        Parameters
        ----------
        kind : string [ 'betweenness' | 'closeness' ], default='betweenness'
            Degree to compute
              - 'betweenness' : each vertex betweenness centrality
              - 'closeness' : each vertex closeness centrality

        Returns
        -------
        centrality : ndarray of shape (n,)
            Returns the network's centrality vector of shape n.

        Examples
        --------
        >>> import numpy as np
        >>> from ts_networks import Network
        >>>
        >>> n, t = 32, 1000
        >>> X = np.random.normal(size=(n, t))
        >>>
        >>> network = Network(X).gaussian_kernel()
        >>> network.centrality()
        >>> network.centrality(kind="closeness")

        References
        ----------
        Newman, M.: Networks. Oxford university press (2018)
        """

        if kind=="betweenness":
            centrality = np.array(list(nx.betweenness_centrality(self.graph, weight="weight").values()))
        elif kind=="closeness":
            centrality = np.array(list(nx.closeness_centrality(self.graph).values()))
        else:
            raise ValueError("Centrality kind should be 'betweenness' or 'closeness'.")

        return centrality

    def clustering_coefficient(self):
        """
        Clustering coefficient, a graph-theoretic feature of the network's vertices.

        Parameters
        ----------
        None

        Returns
        -------
        clustering_coeff : ndarray of shape (n,)
            Returns the network's clustering coefficient of shape n.

        Examples
        --------
        >>> import numpy as np
        >>> from ts_networks import Network
        >>>
        >>> n, t = 32, 1000
        >>> X = np.random.normal(size=(n, t))
        >>>
        >>> network = Network(X).gaussian_kernel()
        >>> network.clustering_coefficient()

        References
        ----------
        Newman, M.: Networks. Oxford university press (2018)
        """

        return np.array(list(nx.clustering(self.graph, weight="weight").values()))

    def laplacian(self, normalized=True):
        """
        Laplacian matrix, a matrix derived from the network's connectivity matrix.

        Parameters
        ----------
        normalized : bool, default=True
            Whether to (row-column) normalize the Laplacian matrix.

        Returns
        -------
        laplacian : ndarray of shape (n, n)
            Returns the network's Laplacian matrix of shape (n, n).

        Examples
        --------
        >>> import numpy as np
        >>> from ts_networks import Network
        >>>
        >>> n, t = 32, 1000
        >>> X = np.random.normal(size=(n, t))
        >>>
        >>> network = Network(X).gaussian_kernel()
        >>> laplacian = network.laplacian() # Matrix object, printable
        >>> laplacian_matrix = laplacian.matrix # ndarray
        >>> laplacian.spectra() # Matrix object method

        References
        ----------
        Chung, F.R.: Spectral graph theory. No. 92, American Mathematical Soc. (1997)
        """

        if normalized:
            L = np.array(nx.normalized_laplacian_matrix(self.graph).todense())
        else:
            D = np.diag(self.adjacency.matrix.sum(axis=0))
            L = D - self.adjacency.matrix

        return Matrix(L)

    def persistence(self, remove_inf=True):
        """
        Persistence diagram object of the network.

        Parameters
        ----------
        remove_inf : bool, default=True
            Whether to remove inf's from the persistence diagram.

        Returns
        -------
        persistence_diagram : PersistenceDiagram instance
            Returns the network's PersistenceDiagram instance.

        Examples
        --------
        >>> import numpy as np
        >>> from ts_networks import Network
        >>>
        >>> n, t = 32, 1000
        >>> X = np.random.normal(size=(n, t))
        >>>
        >>> network = Network(X).gaussian_kernel()
        >>> diagram = network.persistence() # PersistenceDiagram object, printable
        >>> diagram = network.persistence().diagram # ndarray

        References
        ----------
        Edelsbrunner, H., Harer, J.: Computational topology: an introduction.
        American Mathematical Soc. (2010)
        """

        def diagram_numpy(dgms, remove_inf=True):
            
            diagram = []
            for p in dgms[0]:
                diagram.append([0, p.birth, p.death])         
            for p in dgms[1]:
                diagram.append([1, p.birth, p.death])
            diagram = np.array(diagram)

            if remove_inf:
                diagram = diagram[~np.isinf(diagram[:,2])]
                                
            return diagram

        distances = squareform((self.adjacency.matrix + self.adjacency.matrix.T) / 2)
        filtration = dionysus.fill_rips(distances, 2, np.inf)

        R = dionysus.homology_persistence(filtration)
        diagram = dionysus.init_diagrams(R, filtration)

        return PeristenceDiagram(diagram_numpy(diagram, remove_inf))


class Matrix():
    """
    Matrix class to represent different matrices
    associated with the network.

    Attributes
    ----------
    matrix: ndarray of shape (n, n)
        Numpy array of shape (n, n)
    """

    def __init__(self, matrix):
        self.matrix = matrix

    def __repr__(self):
        return np.array_repr(self.matrix)

    def entropy(self):
        """
        Entropy, an information-theoretic feature of given network's matrix.

        Parameters
        ----------
        None

        Returns
        -------
        entropy : float
            Returns the network's matrix entropy.

        Examples
        --------
        >>> import numpy as np
        >>> from ts_networks import Network
        >>>
        >>> n, t = 32, 1000
        >>> X = np.random.normal(size=(n, t))
        >>>
        >>> network = Network(X).gaussian_kernel()
        >>> adjacency = network.adjacency # Matrix object, printable
        >>> laplacian = network.laplacian() # Matrix object, printable
        >>> adjacency.entropy()
        >>> laplacian.entropy()

        References
        ----------
        Cover, T., Thomas, J.: Elements of information theory. Wiley (1991)
        """

        triu_idx = np.triu_indices_from(self.matrix, k=1)
        return entropy(self.matrix[triu_idx])

    def spectra(self):
        """
        Spectra (eigenvalues), a spectral graph-theoretic feature of given network's matrix.

        Parameters
        ----------
        None

        Returns
        -------
        eigenvalues : ndarray of shape (n,)
            Returns the network's matrix spectra (eigenvalues).

        Examples
        --------
        >>> import numpy as np
        >>> from ts_networks import Network
        >>>
        >>> n, t = 32, 1000
        >>> X = np.random.normal(size=(n, t))
        >>>
        >>> network = Network(X).gaussian_kernel()
        >>> adjacency = network.adjacency # Matrix object, printable
        >>> laplacian = network.laplacian() # Matrix object, printable
        >>> adjacency.spectra()
        >>> laplacian.spectra()

        References
        ----------
        Chung, F.R.: Spectral graph theory. No. 92, American Mathematical Soc. (1997)
        """

        eigenvalues, _ = np.linalg.eigh(self.matrix)
        return eigenvalues


class PeristenceDiagram():
    """
    PersistenceDiagram class to represent the persistence diagram
    associated with the network.

    Attributes
    ----------
    diagram : ndarray of shape (p, 3)
        Numpy array containing p persistence points (dim, birth, death) in rows.

    References
    ----------
    Edelsbrunner, H., Harer, J.: Computational topology: an introduction.
    American Mathematical Soc. (2010)
    """

    def __init__(self, diagram):
        self.diagram = diagram

    def __repr__(self):
        return np.array_repr(self.diagram)

    def statistics(self, kind="total_persistence"):
        """
        Persistence statistics out of the network's persistence diagram.

        Parameters
        ----------
        kind : string [ 'total_persistence' | 'entropy ]
            Persistence statistic to compute
              - 'total_persistence' : sum of p (death - birth) for 0 and 1 dimensions
              - 'entropy' : persistence entropy for 0 and 1 dimensions

        Returns
        -------
        statistic : ndarray of shape (2,)
            Returns the network's persistence statistic for 0 and 1 dimensions.

        Examples
        --------
        >>> import numpy as np
        >>> from ts_networks import Network
        >>>
        >>> n, t = 32, 1000
        >>> X = np.random.normal(size=(n, t))
        >>>
        >>> network = Network(X).gaussian_kernel()
        >>> network.persistence().statistics(kind="total_persistence")
        >>> persistence = network.persistence()
        >>> persistence.statistics()
        >>> persistence.statistics(kind="entropy")

        References
        ----------
        Atienza, N., Gonzalez-Diaz, R., Soriano-Trigueros, M.:
        A new entropy based summary function for topological data analysis.
        Electronic Notes in Discrete Mathematics 68, 113-118 (2018)
        """

        def log_safe(vec):
            vec[vec<=0] = 1
            return np.log(vec)

        if kind=="total_persistence":
            pd0 = self.diagram[self.diagram[:,0]==0]
            pd1 = self.diagram[self.diagram[:,0]==1]
            statistic = np.array([np.sum(pd0[:,2] - pd0[:,1]), np.sum(pd1[:,2] - pd1[:,1])])
        elif kind=="entropy":
            total_persistence = self.statistics(kind="total_persistence")
            pd0 = self.diagram[self.diagram[:,0]==0]
            pd1 = self.diagram[self.diagram[:,0]==1]
            p0 = (pd0[:,2] - pd0[:,1]) / total_persistence[0]
            p1 = (pd1[:,2] - pd1[:,1]) / total_persistence[1]
            log_p0 = log_safe(p0)
            log_p1 = log_safe(p1)
            statistic = -np.array([p0 @ log_p0, p1 @ log_p1])
        else:
            raise ValueError("Persistence diagram statistic kind should be 'total_persistence' or 'entropy'.")

        return statistic
    