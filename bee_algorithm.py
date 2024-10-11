import copy
from math import inf, exp, log, pow, sqrt
import random
import logging
from typing import Callable, Any

from tqdm import tqdm
from visualize import draw
import numpy as np
import time
import os
from objective_function import ObjectiveFunction


class BeeAlgorithm:
    def __init__(self, AoI, cell_size, objective_function: ObjectiveFunction, n=7, nb=5, ne=3, nrb=7, nre=7, ngh=20,
                 limit=10, shrink=0.8, num_iter=800, sol_len=100, num_type1=37, num_type2=44, save_dir='./savedir/ba'):
        self.AoI = AoI
        self.cell_size = cell_size
        self._obj_function = objective_function
        self.n = n
        self.nb = nb
        self.ne = ne
        self.nrb = nrb
        self.nre = nre
        self.ngh = ngh
        self.limit = limit
        self.shrink = shrink
        self.num_iter = num_iter
        self.num_type1 = num_type1
        self.num_type2 = num_type2
        self.sol_len = sol_len
        self.patience = 0
        self.patience_patch = [0 for i in range(self.n)]
        self.limit_patch = self.limit
        self.storage = []

        self.root_dir = save_dir
        self.image_dir = os.path.join(self.root_dir, 'plot')
        self.log_dir = os.path.join(self.root_dir, 'log')
        if not os.path.exists(self.root_dir):
            print('Make log dir')
            os.makedirs(self.image_dir)
            os.makedirs(self.log_dir)
        else:
            raise ValueError('Save in another dir')
        self.logger = logging.getLogger(name='best maximum coverage ratio')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.log_dir, 'best_maximum_coverage_ratio.log'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.best_coverage = 0

    def _get_used_node(self, bee):
        used = []
        for s in bee:
            if 0 <= s[0] <= self.AoI[0] and 0 <= s[1] <= self.AoI[1]:
                used.append(s)
        return used

    def _random_init(self):
        sol = []
        for i in range(self.sol_len):
            x = self.AoI[0] * random.random()
            y = self.AoI[1] * random.random()
            sol.append([x, y])
        # for i in range(self.num_type2):
        #     x = self.AoI[0] * random.random()
        #     y = self.AoI[1] * random.random()
        #     sol.append([x, y])
        return sol

    def _initialize_population(self):
        """
        Initialize the population.
        :return: self.n solution vectors
        """
        self.population = []
        for i in range(self.n):
            bee = self._random_init()
            # type_trace = [0 for i in range(self.num_type1)] + [1 for i in range(self.num_type2)]
            used = self._get_used_node(bee)
            type_trace = [random.choice([0, 1]) for j in range(len(used))]
            fitness = self._obj_function.original_fitness(used, type_trace)
            self.population.append([bee, type_trace, fitness])
        key = lambda x: x[2]
        self.best_sol = max(self.population, key=key)
        self.best_sol_patch = [self.population[i] for i in range(self.n)]

    def _find_neigh(self, sol):
        sol_copy = copy.deepcopy(sol)
        x = random.choice(sol_copy[0])
        x[0] += -self.ngh + 2 * self.ngh * random.random()
        x[1] += -self.ngh + 2 * self.ngh * random.random()
        # if 0 <= x[0] <= self.AoI[0] and 0 <= x[1] <= self.AoI[1]:
        #     feasible = True
        # else:
        #     feasible = False
        return sol_copy[0]

    def _neighborhood_shrinking(self):
        key = lambda x: x[2]
        new_best_sol = max(self.population + self.best_sol_patch, key=key)
        if new_best_sol[2] <= self.best_sol[2]:
            self.patience += 1
            if self.patience == self.limit:
                self.ngh *= self.shrink
        else:
            self.patience = 0
            self.best_sol = new_best_sol

    def _site_abandonment(self, i):
        best_neigh = self.population[i]
        if self.best_sol_patch[i][2] >= best_neigh[2]:
            self.patience_patch[i] += 1
            if self.patience_patch[i] == self.limit_patch:
                # self.storage.append(self.best_sol_patch[i])
                bee = self._random_init()
                # type_trace = [0 for i in range(self.num_type1)] + [1 for i in range(self.num_type2)]
                used = self._get_used_node(bee)
                type_trace = [random.choice([0, 1]) for j in range(len(used))]
                fitness = self._obj_function.original_fitness(used, type_trace)
                self.population[i] = [bee, type_trace, fitness]
                self.patience_patch[i] = 0
        else:
            self.patience_patch[i] = 0
            self.best_sol_patch[i] = best_neigh

    def _local_search(self, i):
        neigh_vals = [self.population[i]]
        for j in range(self.nre):
            neigh_val = self._find_neigh(self.population[i])
            # type_trace = [0 for i in range(self.num_type1)] + [1 for i in range(self.num_type2)]
            used = self._get_used_node(neigh_val)
            type_trace = [random.choice([0, 1]) for j in range(len(used))]
            fitness = self._obj_function.original_fitness(used, type_trace)
            neigh_vals.append([neigh_val, type_trace, fitness])
        key = lambda x: x[2]
        best_neigh = max(neigh_vals, key=key)
        self.population[i] = best_neigh

    def _recruitment(self):
        ids = sorted(range(self.n), key=lambda k: self.population[k][2], reverse=True)
        for i in ids[:self.ne]:
            self._local_search(i)
            self._site_abandonment(i)

        for i in ids[self.ne: self.nb]:
            self._local_search(i)
            self._site_abandonment(i)

        for i in ids[self.nb:]:
            bee = self._random_init()
            # type_trace = [0 for i in range(self.num_type1)] + [1 for i in range(self.num_type2)]
            used = self._get_used_node(bee)
            type_trace = [random.choice([0, 1]) for j in range(len(used))]
            fitness = self._obj_function.original_fitness(used, type_trace)
            self.population[i] = [bee, type_trace, fitness]

        self._neighborhood_shrinking()

    def run(self, order):
        start_time = time.time()
        self._initialize_population()
        for loop in tqdm(range(self.num_iter)):
            self._recruitment()
            # print(f"\nBest coverage in iter {loop}: "
            #       f"{self._obj_function.get_coverage_ratio(self._get_used_node(self.best_sol[0]), self.best_sol[1])}")
            # if time.time() - start_time >= 41*60:
            #     break

        best_bee, best_type, best_fitness = self.best_sol[0], self.best_sol[1], self.best_sol[2]
        best_used = self._get_used_node(best_bee)
        best_coverage, target_covered = self._obj_function._coverage_ratio(best_used, best_type)
        best_no_used = len(best_bee)
        best_no_used_convert = sum(best_type) + (len(best_type) - sum(best_type)) / 2

        draw(best_used, best_type, target_covered, os.path.join(self.image_dir, './fig{}.png'.format(str(order))), H=self.AoI[1], W=self.AoI[0], R=self._obj_function.radius, cell_H=self.cell_size[1], cell_W=self.cell_size[0])

        # save the best for 1 run
        self.logger.info(
            f'Best bee: {str(best_bee)}\nType: {str(best_type)}\nBest_fitness: {str(best_fitness)}\nCorresponding '
            f'coverage: {str(best_coverage)} \nCorresponding sensors: {str(best_no_used)} and '
            f'{str(best_no_used_convert)}')
        self.logger.info('------------------------------------------------------------------------------------')

        return best_fitness, best_coverage, best_no_used, best_no_used_convert

    def _reset(self):
        self.patience = 0
        self.patience_patch = [0 for i in range(self.n)]

    def test(self, num_test=10):
        coverage = []
        fitness = []
        no_used = []
        corr = []
        cost = []
        runtime = []
        for i in tqdm(range(num_test)):
            self._reset()
            start = time.time()
            best_fitness, best_coverage, best_no_used, best_no_used_convert = self.run(order=i)
            end = time.time()

            fitness.append(best_fitness)
            no_used.append(best_no_used)
            best_corr = best_no_used_convert / self.sol_len  # (self.num_type1 + self.num_type2)
            corr.append(best_corr)
            cost.append(best_coverage - best_corr)
            coverage.append(best_coverage)
            runtime.append(end - start)

        self.logger.info('------------------------------------------------------------------------------------')
        self.logger.info('------------------------------------------------------------------------------------')
        self.logger.info(f'Coverage mean, std : {str(np.mean(coverage))} and {str(np.std(coverage))}')
        self.logger.info(f'No. used mean, std : {str(np.mean(no_used))} and {str(np.std(no_used))}')
        self.logger.info(f'Corr Used mean, std : {str(np.mean(corr))} and {str(np.std(corr))}')
        self.logger.info(f'Cost mean, std : {str(np.mean(cost))} and {str(np.std(cost))}')
        self.logger.info(f'Fitness mean, std : {str(np.mean(fitness))} and {str(np.std(fitness))}')
        self.logger.info(f'Runtime mean, std : {str(np.mean(runtime))} and {str(np.std(runtime))}')


if __name__ == "__main__":
    pass
