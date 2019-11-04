import traceback

import numpy as np
from numpy.linalg import norm as euclidean_norm
from scipy.spatial.distance import cdist

# TODO поиск элементов шара через nns, матрица расстояний
# TODO классную эвристику на количество особей в популяции (через грид сеарч и апрокс классом функций)

DO_WARNING = []

####
# Utils
####
# TODO найди норм формулу
def _sphere_measure(r, d):
    return np.pi * r**d

####
# INITs
####
def init_population(X, all_dist, n=100):
    i_centers = np.random.choice(list(range(X.shape[0])), n)
    centers = X[i_centers]
    radiuses = np.random.choice(sorted(all_dist[all_dist > 0.05])[500:2000], n)
    return np.hstack((centers, radiuses.reshape(-1, 1)))

####
# STOPs
####
class StopCriteria:
    def __init__(self, verbose=True, max_iter=100, iter_no_improve=10, eps=0.0001):
        self.history = {'n_iter': 0,
                        'population_fitness':
                            {'fmean':[],
                             'fmax': []
                             }
                        }
        self.verbose = verbose
        self.max_iter = max_iter
        self.iter_no_improve = iter_no_improve
        self.eps = eps
        self.record = {'fmean': 0, 'population': None}

    def stop(self, fits, population):
        print(fits)
        fmean = np.mean(fits)
        fmax = np.max(fits)
        if fmean > self.record['fmean']:
            self.record['fmean'] = fmean
            self.record['population'] = population


        if self.max_iter is not None and self.history['n_iter'] > self.max_iter:
            result = True
        elif self.iter_no_improve is not None and \
                self.no_improve(fmean, 'fmean') and \
                self.no_improve(fmax, 'fmax'):
            result = True
        else:
            result = False
        self.history['population_fitness']['fmean'].append(fmean)
        self.history['population_fitness']['fmax'].append(fmax)
        self.history['n_iter'] += 1
        if self.verbose:
            print('{0}# mean: {1}, max: {2}'.format(self.history['n_iter'], fmean, fmax))
        return result

    def no_improve(self, curr_value, key):
        history_by_key = self.history['population_fitness'][key][-self.iter_no_improve:]
        if self.history['n_iter'] < 2:
            return False
        if np.abs(np.max(history_by_key) - curr_value) < self.eps:
            return True
        return False

####
# FITNESS
####

def _inner_mean_dist(ids, X):
    tmp_X = X[ids]
    n = ids.shape[0]
    dists = cdist(tmp_X, tmp_X)
    return np.sum(dists) / (n**2 - n), dists


def _obj_fitness(obj, X):
    eps = 0.1
    try:
        center, r = obj[:-1], obj[-1]
        # TODO замерить скорость с предпосчитанной матрицей и НН
        dist_center = cdist([center], X)[0]
        ids = np.where(dist_center < r)[0]

        # первый фитнес - плотность сферы (насколько оптимально подобран радиус и положение)
        fit1 = (ids.shape[0] / _sphere_measure(r, center.shape[0])) * 0.02
        
        # TODO проигрывают сферы, которые большие
        # второй фитнес - среднее расстояние до центра(насколько оптимально подобрано положение)
        fit2 = np.mean(dist_center[ids])
        
        #density_low = (ids.shape[0] / _sphere_measure(r * (1 - eps), center.shape[0])) * 0.02
        #density_hi = (ids.shape[0] / _sphere_measure(r * (1 + eps), center.shape[0])) * 0.02        
        #fit4 = (fit1 - density_low) * (density_hi - fit1)
        
        
        if np.isnan(fit2):
            if not DO_WARNING:
                print('Warninng: nan fit')
                DO_WARNING.append(1)
            fit2 = 0
    except:
        # print(f'WARNING: bad fit, {traceback.format_exc()}')
        fit1, fit2 = [0]*2
    return fit1 + fit2


def fitness(population, X):
    # TODO добавить джоб либ, чек буст
    fits = []
    count_bad_obj = 0
    for i, obj in enumerate(population):
        fit = _obj_fitness(obj, X)
        # if fit == 0:
        #     i_center = np.random.choice(list(range(X.shape[0])))
        #     center = X[i_center]
        #     i_radius = np.random.choice(list(range(population.shape[0])))
        #     radius = population[i_radius][-1]
        #     x = np.hstack((center, np.array([radius])))
        #     population[i] = x
        #     count_bad_obj += 1
        fits.append(fit)
    print(f'Count bad obj: {count_bad_obj}')
    return fits, population


####
# MUTATION
####
def _vector_mutation(obj, p=0.05, lambda_=0.1):
    shape = obj.shape
    if np.random.random() <= p:
        return obj + lambda_ * np.random.multivariate_normal(np.zeros(shape), np.eye(shape[0]))
    else:
        return obj


####
# CROSSOVER
####
# TODO по идее должен инбридинг зайти
def _random_linear_comb(x1, x2, bias=0.):
    alpha = np.random.random()*(1 + 2 * bias) - bias
    return alpha*x1 + (1 - alpha)*x2


def _linear_crossover(obj_1, obj_2, bias=0.):
    new_obj1 = _random_linear_comb(obj_1, obj_2, bias=bias)
    return new_obj1

####
# BIG FUNCTIONS
####
def get_population(population, fits):
    print(f'Population var: {np.var(population, axis=0)}')
    pop_size = population.shape[0]
    nns = np.argsort(cdist(population, population))[:, 1]
    ids1 = np.random.choice(list(range(pop_size)), pop_size, p=(fits / np.sum(fits)))
    ids2 = nns[ids1]
    new_population = [_vector_mutation(_linear_crossover(population[i1], population[i2], bias=1), p=0.2, lambda_=0.3)
                      for i1, i2 in zip(ids1, ids2)]
    new_population = np.vstack(new_population)
    return new_population


def merge(new_population, population, X):
    eps = 0.02
    merged_population = np.vstack((new_population, population))
    fits, _ = fitness(merged_population, X)
    sorted_population = merged_population[np.argsort(fits)[::-1], :] # [-population.shape[0]:]
    #print("sorted population: ", sorted_population)
    i = 0
    while i < sorted_population.shape[0]:
        j = i + 1
        while j < sorted_population.shape[0]:
            #print("______________", i, j, sorted_population[i, :], 
            #   sorted_population[j, :], euclidean_norm(sorted_population[i, :] - sorted_population[j, :]))
            if euclidean_norm(sorted_population[i, :] - sorted_population[j, :]) < eps:
                sorted_population[j, :2] += np.random.normal(0, 1, (2,))
                #sorted_population = np.delete(sorted_population, (j, ), axis=0)
            j += 1
        i += 1
    #print("New sorted population without duples: ", sorted_population)
    #input()
    return sorted_population[:population.shape[0], :]
