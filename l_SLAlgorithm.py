from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors


class l_SLAlgorithm():
    def __init__(self, threshold=2):
        self.threshold = threshold
        self.tau = threshold / 2
        self.labels = None

    def calculate_maximum_life_time(self, X):
        Clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='single',
                                             compute_distances=True)
        results = Clustering.fit(X)
        maximum_life_time = 0
        actualdistance = 0
        for i in results.distances_:
            if actualdistance == 0:
                actualdistance = i
                if actualdistance != 0:
                    maximum_life_time = actualdistance
            else:
                if maximum_life_time < i - actualdistance:
                    maximum_life_time = i - actualdistance
                actualdistance = i
        self.threshold = maximum_life_time
        self.tau = self.threshold / 2
        return maximum_life_time

    def fit(self, X, Y=None):
        self.labels, self.leaders, self.clustermade = self._fit_predict_process(X)
        return self

    def fit_predict(self, X,
                    Y=None):  # Esto es mas bien fit_predict consultar como solucionar para hacer un fit propiamente
        self.labels, self.leaders, self.clustermade = self._fit_predict_process(X)
        return self.labels, self.leaders, self.clustermade

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self):
        return self

    def _fit_process(self, X):
        resultado, setforleaders = self.leader_algorithm(X, self.tau)
        Clustering = AgglomerativeClustering(None, distance_threshold=self.threshold)
        results = Clustering.fit(resultado)
        return results.labels_, resultado

    def _fit_predict_process(self, X):
        resultado, setforleaders = self.leader_algorithm(X, self.tau)
        Clustering = AgglomerativeClustering(None, linkage='single', distance_threshold=self.threshold)
        results = Clustering.fit(resultado)
        fullfinalCluster = np.empty(len(X))
        for i in range(results.n_clusters_):
            axis = [j for j, x in enumerate(results.labels_) if x == i]
            for j in axis:
                fullfinalCluster[setforleaders[j]]=str(i)
        return results.labels_, resultado, fullfinalCluster

    def _predict_process(self, X):  # ESTO SE TIENE QUE BORRAR NO SE TIENE QUE IMPLEMENTAR.
        leader_for_point = []
        label_for_point = []
        for i in range(len(X)):
            dist = euclidean_distances(self.leaders, [X[i]])
            mascercano = np.where(dist == np.amin(dist))
            if len(mascercano[0]) == 1:
                if dist[mascercano[0]][mascercano[1]] <= self.tau:
                    leader_for_point.append(self.leaders[mascercano[0]])
                    label_for_point.append(self.labels[mascercano[0]])
                else:
                    newLeader = [X[i]]
                    self.leaders.append(newLeader)
                    Clustering = AgglomerativeClustering(None, distance_threshold=self.threshold)
                    results = Clustering.fit(self.leaders)
                    return len(self.leaders), results.labels_
            else:
                if dist[mascercano[0][0]][mascercano[1][0]] <= self.tau:
                    leader_for_point.append(self.leaders[mascercano[0][0]])
                    label_for_point.append(self.labels[mascercano[0][0]])
                else:
                    newLeader = [X[i]]
                    self.leaders.append(newLeader)
                    Clustering = AgglomerativeClustering(None, distance_threshold=self.threshold)
                    results = Clustering.fit(self.leaders)
                    return len(self.leaders)
        return leader_for_point, label_for_point

    @staticmethod
    def leader_algorithm(X, tau):
        leaderList = np.empty((0, X[0].size))
        leaderList = np.vstack((leaderList, X[0]))
        setforleaders = []
        newLeader = [0]
        setforleaders.append(newLeader)
        localcopy = np.delete(X, 0, 0)
        newLeader=[]



        for i in range(len(localcopy)):
            dist = euclidean_distances(leaderList, [localcopy[i]])
            mascercano = np.where(dist == np.amin(dist))
            if len(mascercano[0]) == 1:
                if dist[mascercano[0]][mascercano[1]] <= tau:
                    newpointforleader = i+1
                    setforleaders[mascercano[0].item()].append(newpointforleader)
                else:
                    leaderList = np.vstack((leaderList, localcopy[i]))
                    newLeader = [i+1]
                    setforleaders.append(newLeader)
            else:
                if dist[mascercano[0][0]][mascercano[1][0]] <= tau:
                    newpointforleader = i+1
                    setforleaders[mascercano[0][0].item()].append(newpointforleader)
                else:
                    leaderList = np.vstack((leaderList, localcopy[i]))
                    newLeader = [i+1]
                    setforleaders.append(newLeader)

        return leaderList, setforleaders
