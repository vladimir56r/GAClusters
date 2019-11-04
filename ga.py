import heapq
from collections import defaultdict
import itertools
import time
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import operators


# TODO cool default params
class ga_cluster:
    def __init__(self, n_clusters=5, verbose=True, **params):
        self.verbose = verbose

        self.cluster_params = {'n_clusters': n_clusters}
        self.ga_params = params
        self.ga_params['n_population'] = n_clusters*1 + 10

    ####
    # API
    ####
    def predict(self, X):
        dists = cdist(X, self.cluster_key_points)
        return self.cluster_labels[np.argsort(dists)[:, 0]]

    def fit_predict(self, X):
        population = self._genetic_algorithm(X, self.ga_params)
        cluster_key_points, labels = self._knp_cluster(population, self.cluster_params['n_clusters'])
        self.cluster_key_points = cluster_key_points[:, :-1]
        self.cluster_labels = np.array(labels)
        preds = self.predict(X)
        return preds

    ####
    # show
    ####
    def plot_ds(self, X, population):
        print(X.shape)

        fig, ax = plt.subplots()
        ax.set_xlim((0, 7.5))
        ax.set_ylim((0, 7.5))
        for circle in population:
            # print("sircle", circle)
            ax.add_artist(plt.Circle(tuple(circle[:2]), circle[2]))
        ax.plot(X[:, 0], X[:, 1], 'bx')
        ax.plot(population[:, 0], population[:, 1], 'ro')
        #fig.show()
        plt.savefig("img/" + str(time.time()) + ".jpg")
        return 0

    ####
    # genetic alg
    ####
    def _genetic_algorithm(self, X, params):
        dist_matrix = cdist(X, X)
        all_dist = np.hstack(dist_matrix)
        population = operators.init_population(X, all_dist, n=self.ga_params['n_population'])
        print(population)
        obj_criteria = operators.StopCriteria(verbose=self.verbose)
        fits, population = operators.fitness(population, X)
        while not obj_criteria.stop(fits, population):
            self.plot_ds(X, population)
            new_population = operators.get_population(population, fits)
            population = operators.merge(new_population, population, X)
            fits, population = operators.fitness(population, X)
        print(obj_criteria.record['population'])
        return obj_criteria.record['population']


    ####
    # KNP alg
    ####
    def _knp(self, population):
        g = defaultdict(list)
        weight = 0
        connected = set([])
        pq = []

        dist_matrix = cdist(population, population)
        count_vertices = dist_matrix.shape[0]
        for i in range(count_vertices):
            for j in range(count_vertices):
                a, b, w = i, j, dist_matrix[i, j]
                g[a].append((w, b))
                g[b].append((w, a))

        MCT = []
        start = 0
        connected.add(start)
        for tup in g[start]:
            heapq.heappush(pq, (start, ) + tup)
        while pq:
            a, w, b = heapq.heappop(pq)
            if b not in connected:
                weight += w
                MCT.append([a, b, w])
                connected.add(b)
                for tup in g[b]:
                    heapq.heappush(pq, (b, ) + tup)
        return MCT

    def _knp_cluster(self, population, count_clusters):
        ''' обрезка под нужное количество кластеров '''
        MCT = self._knp(population)
        ordered_MCT = sorted(MCT, key=lambda x: x[-1])
        i_centers = self._cut_mct(ordered_MCT, count_clusters)
        components = [[population[i] for i in comp] for comp in i_centers]
        key_points = np.vstack([x for label, comp in enumerate(components) for x in comp])
        labels = [label for label, comp in enumerate(components) for x in comp]
        return key_points, labels

    def _cut_mct(self, ordered_MCT, count_clusters):
        cuted_MCT = ordered_MCT[:-count_clusters] + \
                    list(itertools.chain(*[[[a, a, 0], [b, b, 0]]for a, b, _ in ordered_MCT[-count_clusters:]]))
        cuted_MCT = list(set(map(tuple, cuted_MCT)))
        components = self._extract_components(cuted_MCT)
        return components


    def _dfs(self, adj_dict, v):
        component = {v}
        st = list()
        [st.append(child) for child in adj_dict[v]]
        while st:
            v = st.pop()
            component.add(v)
            [st.append(child) for child in adj_dict[v] if child not in component]
        return component

    def _extract_components(self, adj_list):
        adj_dict = defaultdict(set)
        vertices = set([])
        for a, b, _ in adj_list:
            adj_dict[a].add(b)
            adj_dict[b].add(a)
            vertices.add(a)
            vertices.add(b)

        placed = set([])
        components = []
        for v in list(vertices):
            if v in placed:
                continue
            new_component = self._dfs(adj_dict, v)
            components.append(new_component)
            placed = placed.union(new_component)
        return components


if __name__ == '__main__':
    import time

    import sklearn
    from sklearn.cluster import k_means
    import sklearn.datasets as ds
    from sklearn.metrics import adjusted_rand_score

    import os
    import shutil

    for root, dirs, files in os.walk('img'):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    data = ds.load_iris()
    X = data.data[:, 1:3]
    y = data.target

    t = time.time()
    clf = ga_cluster(n_clusters=2)
    y_pred = clf.fit_predict(X)
    print(adjusted_rand_score(y, y_pred), time.time() - t)

    t = time.time()
    centers, y_pred, _ = k_means(X, n_clusters=2)
    print(adjusted_rand_score(y, y_pred), time.time() - t)