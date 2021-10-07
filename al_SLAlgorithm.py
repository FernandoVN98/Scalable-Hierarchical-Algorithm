import itertools

from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import copy


class al_SLAlgorithm():
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
                maximum_life_time = actualdistance
            else:
                if maximum_life_time < i - actualdistance:
                    maximum_life_time = i - actualdistance
                actualdistance = i
        self.threshold = maximum_life_time
        self.tau = self.threshold / 2
        return maximum_life_time

    def fit(self, X, Y=None):
        self.labels, self.leaders, self.clustermade = self._fit_process(X)
        return self

    def fit_predict(self, X,
                    Y=None):  # Esto es mas bien fit_predict consultar como solucionar para hacer un fit propiamente
        self.labels, self.leaders, self.clustermade = self._fit_process(X)
        return self.labels, self.leaders, self.clustermade

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self):
        return self

    def _fit_process(self, X):
        resultado, setforleaders, idxforleaders = self.leader_algorithm(X, self.tau)
        Clustering = AgglomerativeClustering(None, linkage='single', distance_threshold=self.threshold)
        results = Clustering.fit(resultado)
        S = []
        indexesleaders = []
        combinations = [x for x in
                        list(itertools.product(list(range(results.n_clusters_)), list(range(results.n_clusters_)))) if
                        x[0] != x[1] and x[0] < x[1]]
        for i in combinations:
            firstcluster = [resultado[h] for h in range(len(resultado)) if results.labels_[h] == i[0]]
            secondcluster = [resultado[h] for h in range(len(resultado)) if results.labels_[h] == i[1]]
            dist = euclidean_distances(firstcluster, secondcluster)
            distanciareal = np.where(dist == np.amin(dist))
            if dist[distanciareal[0][0]][distanciareal[1][0]] <= 2 * self.threshold:
                LBi = [[leader] for leader in firstcluster if
                       euclidean_distances([leader], [secondcluster[distanciareal[1][0]]])[0][0] <= 2 * self.threshold]
                LBj = [[leader] for leader in secondcluster if
                       euclidean_distances([firstcluster[distanciareal[0][0]]], [leader])[0][0] <= 2 * self.threshold]
                indexesleaders.append([i[0], i[1]])
                S.append(LBi)
                S.append(LBj)
        suma = sum(len(s) for s in S)
        if len(S) != 0 and suma > 0:
            my_final_sl_clusters = copy.copy(results.labels_)
            dict_to_desambiguate = {}
            for i, j in enumerate(indexesleaders):
                pot_leaders_a = np.array([s[0][:] for s in S[i * 2]])
                pot_leaders_b = np.array([s[0][:] for s in S[(i * 2) + 1]])
                followers_of_leaders_a = [list(setforleaders[i][k]) for i in range(len(resultado)) for j in
                                          range(len(pot_leaders_a)) for k in range(len(setforleaders[i])) if
                                          np.array_equal(pot_leaders_a[j], resultado[i]) and len(
                                              setforleaders[i]) > 0]
                followers_of_leaders_b = [list(setforleaders[i][k]) for i in range(len(resultado)) for j in
                                          range(len(pot_leaders_b)) for k in range(len(setforleaders[i])) if
                                          np.array_equal(pot_leaders_b[j], resultado[i]) and len(
                                              setforleaders[i]) > 0]
                if len(followers_of_leaders_b) > 0 and len(followers_of_leaders_a) > 0:
                    dist = euclidean_distances(followers_of_leaders_a, followers_of_leaders_b)
                    distanciareal = np.where(dist == np.amin(dist))
                    if dist[distanciareal[0][0]][distanciareal[1][0]] <= self.threshold:
                        flat_list = [item for sublist in dict_to_desambiguate.values() for item in sublist]
                        if j[0] in flat_list:
                            if j[1] in flat_list:
                                for key, value in dict_to_desambiguate.items():
                                    if j[0] in value:
                                        dict_to_desambiguate[key].append(j[1])
                                        auxiliarkey1=key
                                    if j[1] in value:
                                        dict_to_desambiguate[key].append(j[0])
                                        auxiliarkey2=key
                                dict_to_desambiguate[auxiliarkey2].append(auxiliarkey1)
                                dict_to_desambiguate[auxiliarkey1].append(auxiliarkey2)
                            else:
                                for key, value in dict_to_desambiguate.items():
                                    if j[0] in value:
                                        dict_to_desambiguate[key].append(j[1])
                        elif j[1] in flat_list:
                            for key, value in dict_to_desambiguate.items():
                                if j[1] in value:
                                    dict_to_desambiguate[key].append(j[0])
                        elif j[0] in dict_to_desambiguate:
                            dict_to_desambiguate[j[0]].append(j[1])
                        else:
                            dict_to_desambiguate[j[0]]=[j[1]]
        for key,value in dict_to_desambiguate.items():
            my_final_sl_clusters=[i if i not in value else key for i in my_final_sl_clusters]
        fullfinalCluster = np.empty(len(X))
        for i in set(my_final_sl_clusters):
            axis = [j for j, x in enumerate(my_final_sl_clusters) if x == i]
            for j in axis:
                fullfinalCluster[idxforleaders[j]] = str(i)
        return my_final_sl_clusters, resultado, fullfinalCluster

    @staticmethod
    # test for equality:
    def arreq_in_list(myarr, list_arrays):
        return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

    @staticmethod
    def leader_algorithm(X, tau):
        leaderList = np.empty((0, X[0].size))
        leaderList = np.vstack((leaderList, X[0]))
        setforleaders = []
        idxforleaders = []
        newLeader = [X[0]]
        setforleaders.append(newLeader)
        idxforleaders.append([0])
        localcopy = np.delete(X, 0, 0)

        for j, i in enumerate(localcopy):
            dist = euclidean_distances(leaderList, [i])
            mascercano = np.where(dist == np.amin(dist))
            if len(mascercano[0]) == 1:
                if dist[mascercano[0]][mascercano[1]] <= tau:
                    newpointforleader = i
                    setforleaders[mascercano[0].item()].append(newpointforleader)
                    idxforleaders[mascercano[0].item()].append(j + 1)
                else:
                    leaderList = np.vstack((leaderList, i))
                    newLeader = [i]
                    setforleaders.append(newLeader)
                    idxforleaders.append([j + 1])
            else:
                if dist[mascercano[0][0]][mascercano[1][0]] <= tau:
                    newpointforleader = i
                    setforleaders[mascercano[0][0].item()].append(newpointforleader)
                    idxforleaders[mascercano[0][0].item()].append(j + 1)
                else:
                    leaderList = np.vstack((leaderList, i))
                    newLeader = [i]
                    setforleaders.append(newLeader)
                    idxforleaders.append([j + 1])

        return leaderList, setforleaders, idxforleaders
