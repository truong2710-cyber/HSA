from math import inf, exp, log, pow, sqrt
import random
import logging
from tqdm import tqdm
from visualize import draw
import numpy as np 
import os 
# from sklearn.cluster import KMeans
from dataset import import_data

class HarmonySearch():
    def __init__(self, objective_function, AoI, cell_size, hms=30, hmv=7, hmcr=0.9, par=0.3, BW=0.2, lower=[], upper=[], min_no = 0, radius=[5,10], savedir = './test/hsa', data='hanoi'):
        """
            param explaination
            
            :param hms: harmony memory size, number of vectors stored in harmony memory
            :param hmv: harmony vector size
            :param hmcr: probability for each node considering
            :param par: pitch adjustment rate
            :param BW: distance bandwidth, used for adjust node position when pich adjustment is applied
            :param lower: list contains coordinates for bottom corners
            :param upper: list contains coordinates for upper corners
        """
        self.root_dir = savedir
        self.image_dir = os.path.join(self.root_dir, 'plot')
        self.log_dir = os.path.join(self.root_dir, 'log')
        if not os.path.exists(self.root_dir):
            print('Make log dir')
            os.makedirs(self.root_dir)
            os.makedirs(self.image_dir)
            os.makedirs(self.log_dir)
        else:
            raise ValueError('Save in another dir')
        
        self._obj_function = objective_function
        self.radius = self._obj_function.get_radius()
        self.hms = hms
        self.hmv = hmv
        self.hmcr = hmcr
        self.par = par
        self.BW = BW
        self.lower = lower
        self.upper = upper
        self.min_no = min_no
        self.AoI = AoI
        self.cell_size = cell_size
        self.bw_min = 0.002
        self.bw_max = 0.2
        self.par_min = 0.01       # for GHS and IHS
        self.par_max = 0.99       # _______________
        self.mutation_prob = 0.1  # for NGHS
        self.radius = radius

        self.logger2 = logging.getLogger(name='best maximum coverage ratio')
        self.logger2.setLevel(logging.INFO)
        handler2 = logging.FileHandler(os.path.join(self.log_dir, 'best_maximum_coverage_ratio.log'))
        handler2.setLevel(logging.INFO)
        formatter2 = logging.Formatter('%(levelname)s: %(message)s')
        handler2.setFormatter(formatter2)
        self.logger2.addHandler(handler2)
        self.best_coverage = 0
        self.z = import_data(data)[0]

    # def _distance(self, x1, x2):
    #     return sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

    # def _prob_loc(self, s, t):
    #     distance = self._distance(s, t)
        
    #     if distance < 5:
    #         return 1
    #     elif distance > 15:
    #         return 0
    #     else:
    #         lambda1 = -5 + distance
    #         lambda2 = 15 - distance
    #         lambda1 = pow(lambda1, 1)
    #         lambda2 = pow(lambda2, 0.5)
    #         return exp(-lambda1/lambda2)

    def _prob_loc_all(self, s, t_set):
        p = 1
        for t in t_set:
            pi = self._obj_function._psm(s, t, type=1)
            if pi >= 0.6:
                p = p * pi
        return p

    def _prob_selection(self):
        num_width_cell = self.AoI[0] // self.cell_size[0]
        num_height_cell = self.AoI[1] // self.cell_size[1]
        #id_valid_cell = list(range(num_height_cell * num_width_cell))
        harmony = []
        xs = np.arange(5, 46, 10)
        ys = np.arange(5, 46, 10)
        targets = []
        for x in xs:
            for y in ys:
                targets.append([x,y])
        for ids in range(self.hmv):
            type_ = random.choice([0, 1])
            if type_ == 0:
                x = self.lower[0][0] + (self.upper[0][0] - self.lower[0][0])*random.random()
                y = self.lower[0][1] + (self.upper[0][1] - self.lower[0][1])*random.random()
            else:
                x = self.lower[1][0] + (self.upper[1][0] - self.lower[1][0])*random.random()
                y = self.lower[1][1] + (self.upper[1][1] - self.lower[1][1])*random.random()
            while True:
                z = np.random.rand()
                if z < self._prob_loc_all([x, y], targets):
                    harmony.append([x, y])
                    break
        return harmony


    def _random_selection(self, min_valid):
        harmony = []
        for each_node in range(self.hmv):
            type_ = random.choice([0, 1])
            if type_ == 0:
                x = self.lower[0][0] + (self.upper[0][0] - self.lower[0][0])*random.random()
                y = self.lower[0][1] + (self.upper[0][1] - self.lower[0][1])*random.random()
            else:
                x = self.lower[1][0] + (self.upper[1][0] - self.lower[1][0])*random.random()
                y = self.lower[1][1] + (self.upper[1][1] - self.lower[1][1])*random.random()
            harmony.append([x, y])
        return harmony      # no pad?
    
    def _centroid_selection(self, min_valid):
        num_width_cell = self.AoI[0] // self.cell_size[0]
        num_height_cell = self.AoI[1] // self.cell_size[1]
        id_valid_cell = list(range(num_height_cell * num_width_cell))
        random.shuffle(id_valid_cell)
        harmony = []
        for ids in range(self.hmv):
            type_ = random.choice([0,1])  # what for?
            width_coor = id_valid_cell[ids] % num_width_cell
            height_coor = id_valid_cell[ids] // num_width_cell
            x = width_coor * self.cell_size[0] + self.cell_size[0] / 2
            y = height_coor * self.cell_size[1] + self.cell_size[1] / 2
            harmony.append([x, y])
        
        return harmony
    
    def _cell_selection(self, min_valid):
        num_width_cell = self.AoI[0] // self.cell_size[0]
        num_height_cell = self.AoI[1] // self.cell_size[1]
        id_valid_cell = list(range(num_height_cell * num_width_cell))
        random.shuffle(id_valid_cell)
        harmony = []
        for ids in range(self.hmv):
            type_ = random.choice([0,1])  # what for?
            width_coor = id_valid_cell[ids] % num_width_cell
            height_coor = id_valid_cell[ids] // num_width_cell
            x = width_coor * self.cell_size[0] + self.cell_size[0]*random.random()
            y = height_coor * self.cell_size[1] + self.cell_size[1]*random.random()
            harmony.append([x, y])
        return harmony

    def _initialize_harmony(self, type = "default", min_valid=14, initial_harmonies=None):  # what is min_valid?
        """
            Initialize harmony_memory, the matrix containing solution vectors (harmonies)
        """
        if initial_harmonies is not None:
            # assert len(initial_harmonies) == self._obj_function.get_hms(),\
            #     "Size of harmony memory and objective function is not compatible"
            # assert len(initial_harmonies[0]) == self._obj_function.get_num_parameters(),\
            #     "Number of params in harmony memory and objective function is not compatible"
            for each_harmony, type_trace in initial_harmonies:
                self._harmony_memory.append((each_harmony, self._obj_function.get_fitness(each_harmony, type_trace)[0]))
        else:
            assert type in ["default", "centroid", "cell", "prob"], "Unknown type of initialization"
            self._harmony_memory = []
            if type == "default":
                for _ in range(0, self._obj_function.get_hms()):
                    harmony = self._random_selection(min_valid)
                    fitness, type_trace = self._obj_function.get_fitness(harmony)
                    self._harmony_memory.append((harmony, type_trace, fitness[0]))
            elif type == "centroid":
                for _ in range(0, self._obj_function.get_hms()):
                    harmony = self._centroid_selection(min_valid)
                    fitness, type_trace = self._obj_function.get_fitness(harmony)
                    self._harmony_memory.append((harmony, type_trace, fitness[0]))
            elif type == "cell":
                for _ in range(0, self._obj_function.get_hms()):
                    harmony = self._cell_selection(min_valid)
                    fitness, type_trace = self._obj_function.get_fitness(harmony)
                    self._harmony_memory.append((harmony, type_trace, fitness[0]))
            elif type == "prob":
                for _ in range(0, self._obj_function.get_hms()):
                    harmony = self._prob_selection()
                    fitness, type_trace = self._obj_function.get_fitness(harmony)
                    self._harmony_memory.append((harmony, type_trace, fitness[0]))
    
    def _memory_consideration(self, n_iter, max_iter, mode = 'HS'):
        """
            Generate one new harmony from previous harmonies in harmony memory
            Apply pitch adjustment with par probability
        """
        harmony = []
        if mode in ['HS', 'GHS', 'IHS']:
            for i in range(self.hmv):
                p_hmcr = random.random()
                if p_hmcr < self.hmcr:
                    id = random.choice(range(self.hms))
                    [x, y] = self._harmony_memory[id][0][i]
                    [x, y] = self._pitch_adjustment([x, y], n_iter, max_iter, mode=mode)
                else:
                    type_ = random.choice([0, 1])
                    if type_ == 0:
                        x = self.lower[0][0] + (self.upper[0][0] - self.lower[0][0])*random.random()
                        y = self.lower[0][1] + (self.upper[0][1] - self.lower[0][1])*random.random()
                    else:
                        x = self.lower[1][0] + (self.upper[1][0] - self.lower[1][0])*random.random()
                        y = self.lower[1][1] + (self.upper[1][1] - self.lower[1][1])*random.random()

                if x > self.upper[1][0] or x < self.lower[0][0]:
                    x = -1
                if y > self.upper[1][1] or y < self.lower[0][1]:
                    y = -1
                harmony.append([x, y])
        if mode == 'NGHS':
            worst_fitness = float("+inf")
            worst_ind = -1
            best_fitness = float("-inf")
            best_ind = -1
            for ind, (_, _x, each_fitness) in enumerate(self._harmony_memory):
                if each_fitness < worst_fitness:
                    worst_fitness = each_fitness
                    worst_ind = ind
                
                if each_fitness > best_fitness:
                    best_fitness = each_fitness
                    best_ind = ind
            for i in range(self.hmv):
                type_ = random.choice([0, 1])
                x_r = self._subtract_set(self._mul_set(self._harmony_memory[best_ind][0][i], 2), self._harmony_memory[worst_ind][0][i])
                if type_ == 0:
                    x_r[0] = min([x_r[0], self.upper[0][0]])
                    x_r[1] = min([x_r[1], self.upper[0][1]])
                if type_ == 1:
                    x_r[0] = min([x_r[0], self.upper[1][0]])
                    x_r[1] = min([x_r[1], self.upper[1][1]])
                x_new = self._add_set(self._harmony_memory[worst_ind][0][i], self._mul_set(self._subtract_set(x_r, self._harmony_memory[worst_ind][0][i]), random.random()))
                if random.random() <= self.mutation_prob:
                    type_ = random.choice([0, 1])
                    x = -1
                    y = -1
                    if type_ == 0:
                        x = self.lower[0][0] + (self.upper[0][0] - self.lower[0][0])*random.random()
                        y = self.lower[0][1] + (self.upper[0][1] - self.lower[0][1])*random.random()
                    else:
                        x = self.lower[1][0] + (self.upper[1][0] - self.lower[1][0])*random.random()
                        y = self.lower[1][1] + (self.upper[1][1] - self.lower[1][1])*random.random()
                    x_new = [x, y]
                harmony.append(x_new)
        
        return harmony

    def _subtract_set(self, s1, s2):
        return [s1[i] - s2[i] for i in range(len(s1))]
    
    def _add_set(self, s1, s2):
        return [s1[i] + s2[i] for i in range(len(s1))]

    def _mul_set(self, s, k):
        return [k * x for x in s]


    def _pitch_adjustment(self, position, n_iter, max_iter, mode = 'HS'):
        """
            Adjustment for generating completely new harmony vectors
        """
        if mode == 'HS':
            p_par = random.random()
            if p_par < self.par:
                bw_rate = random.uniform(-1,1)
                position[0] = self.BW*bw_rate + position[0]
                position[1] = self.BW*bw_rate + position[1]
        elif mode == 'GHS':
            self.par = self.par_min + (self.par_max - self.par_min) * n_iter / max_iter
            p_par = random.random()
            if p_par < self.par:
                best_fitness = float("-inf")
                best_ind = -1
                for ind, (_, _x, each_fitness) in enumerate(self._harmony_memory):
                    if each_fitness > best_fitness:
                        best_fitness = each_fitness
                        best_ind = ind                
                k = random.randint(0, self.hmv - 1)
                position = self._harmony_memory[best_ind][0][k]
        elif mode == 'IHS':
            self.par = self.par_min + (self.par_max - self.par_min) * n_iter / max_iter
            self.BW = self.bw_max * exp(log(self.bw_min / self.bw_max) * n_iter / max_iter)
            p_par = random.random()
            if p_par < self.par:
                bw_rate = random.uniform(-1,1)
                position[0] = self.BW*bw_rate + position[0]
                position[1] = self.BW*bw_rate + position[1]
        return position


    def _new_harmony_consideration(self, harmony, fitness, type_trace):
        """
            Update harmony memory
        """
        # (fitness, _), type_trace = self._obj_function.get_fitness(harmony)
        
        worst_fitness = float("+inf")
        worst_ind = -1
        best_fitness = float("-inf")
        best_ind = -1
        for ind, (_, _x, each_fitness) in enumerate(self._harmony_memory):
            if each_fitness < worst_fitness:
                worst_fitness = each_fitness
                worst_ind = ind
            
            if each_fitness > best_fitness:
                best_fitness = each_fitness
                best_ind = ind
        
        if fitness >= worst_fitness:
            self._harmony_memory[worst_ind] = (harmony, type_trace, fitness)
        
        if fitness > best_fitness:
            best_ind = worst_ind
        return best_ind

    def _get_best_fitness(self):
        """
            Gest best fitness and corresponding harmony vector in harmony memory
        """
        best_fitness = float("-inf")
        best_harmony = []
        for each_harmony, type_trace, each_fitness in self._harmony_memory:
            if each_fitness > best_fitness:
                best_fitness = each_fitness
                best_harmony = each_harmony
                type_ = type_trace
        return best_harmony, type_, best_fitness

    def _get_best_coverage_ratio(self):
        best_harmony, type_trace = self._get_best_fitness()[0:2]
        (_, coverage_ratio), _ = self._obj_function.get_fitness(best_harmony)
        return coverage_ratio

    def _evaluation(self, threshold):
        coverage_ratio = self._get_best_coverage_ratio()
        best_harmony, type_, best_fitness = self._get_best_fitness()
        if coverage_ratio > self.best_coverage and coverage_ratio >= threshold:
            self.logger.info(f"Pos: {str(best_harmony)}\nType: {str(type_)}\nCoverage: {str(coverage_ratio)}")
            self.best_coverage = coverage_ratio
            self.logger.info("This harmony is sastified")
        elif coverage_ratio >= threshold:
            self.logger.info(f"Pos: {str(best_harmony)}\nType: {str(type_)}\nCoverage: {str(coverage_ratio)}")
            self.logger.info("This harmony is sastified")
        elif coverage_ratio > self.best_coverage:
            self.logger.info(f"Pos: {str(best_harmony)}\nType: {str(type_)}\nCoverage: {str(coverage_ratio)}")
            self.best_coverage = coverage_ratio
        return False

    def _count_sensor(self, harmony):
        count_ = 0
        for item in harmony:
            if item[0] >= 0 and item[1] >= 0:
                count_ += 1
        return count_
    
    def _search(self, nSearch, n_iter, max_iter):   # create nSearch new harmonies at a time
        bestharmony = None 
        besttrace = None
        best = float('-inf')
        
        for i in range(nSearch):
            candidate_harmony = self._memory_consideration(n_iter, max_iter)
            (candidatefitness, _), type_trace = self._obj_function.get_fitness(candidate_harmony)
            if candidatefitness > best:
                best=candidatefitness
                bestharmony = candidate_harmony
                besttrace=type_trace
        
        return bestharmony, best, besttrace


    def run(self, type_init="default", min_valid=14, steps=100, threshold=0.9, order=0, logger=None):
        
        print("Start run:")
        self._initialize_harmony(type_init, min_valid)

        best_ind = -1
        for i in tqdm(range(steps)):

            new_harmony, new_fitness, new_trace = self._search(1, i, steps)   # default nSearch = 1 as in the original paper # edit here

            new_best_ind = self._new_harmony_consideration(new_harmony, new_fitness, new_trace)

            #best_harmony, best_type, best_fitness = self._harmony_memory[best_ind]
            
            if new_best_ind != best_ind:
                best_ind = new_best_ind
                best_harmony, best_type, best_fitness = self._harmony_memory[best_ind]
                logger.info(f'Step: {str(i)}\n Best harmony: {str(best_harmony)}\n Type: {str(best_type)}\n Best_fitness: {str(best_fitness)}')
                logger.info('------------------------------------------------------------------------------------')

        best_harmony, best_type, best_fitness = self._harmony_memory[best_ind]
        
   
        used_node = []
        type_trace = best_type
        for ind, node in enumerate(best_harmony):
            if node[0] > 0 and node[1] > 0:
                used_node.append(node)
        coverage, target_covered = self._obj_function._coverage_ratio(used_node, type_trace)
        no_used = len(used_node)
        no_used_convert = sum(type_trace) + (len(type_trace)-sum(type_trace))/2   

        draw(used_node, type_trace, target_covered, os.path.join(self.image_dir, './fig{}.png'.format(str(order))), H=self.AoI[1], W=self.AoI[0], R=self.radius, cell_H=self.cell_size[1], cell_W=self.cell_size[0])

        # save the best for1 runs
        self.logger2.info(f'Best harmony: {str(best_harmony)}\nType: {str(type_trace)}\nBest_fitness: {str(best_fitness)}\nCoressponding coverage: {str(coverage)} \nCoressponding sensors: {str(no_used)} and {str(no_used_convert)}')
        self.logger2.info('------------------------------------------------------------------------------------')
        
        
        
        return best_fitness, coverage, no_used, no_used_convert
    
    def pick_one_node(self, id, i):
        if random.random() < self.hmcr:
            while True:
                id_rand = random.choice(range(self.hms))
                [x, y] = self._harmony_memory[id_rand][0][i]
                x, y = self._pitch_adjustment([x, y])
                if id_rand != id:
                    break
                
        else:
            type =random.randint(0, 1)
            if type == 0:
                x = self.lower[0][0] + (self.upper[0][0] - self.lower[0][0])*random.random()
                y = self.lower[0][1] + (self.upper[0][1] - self.lower[0][1])*random.random()
            else:
                x = self.lower[1][0] + (self.upper[1][0] - self.lower[1][0])*random.random()
                y = self.lower[1][1] + (self.upper[1][1] - self.lower[1][1])*random.random()
        return [x, y]

    def test(self, type_init="default", min_valid=14, steps=60000, threshold=0.9, file='logging.txt', num_run=12):
        coverage = []
        fitness = []
        used = []
        cost = []
        corr = []
        for i in range(num_run):
            logger = logging.getLogger(name='harmony{}'.format(str(i)))
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(os.path.join(self.log_dir, "output{}.log".format(i)))
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            ifitness, icover, iused, iusedconvert = self.run(type_init, min_valid, steps, threshold, i, logger)
            
            coverage.append(icover)
            fitness.append(ifitness)
            used.append(iused)
            corr.append(iusedconvert/self.hmv)
            cost.append(icover - iusedconvert/self.hmv)
        

        self.logger2.info('------------------------------------------------------------------------------------') 
        self.logger2.info('------------------------------------------------------------------------------------') 
        self.logger2.info(f'Coverage mean, std : {str(np.mean(coverage))} and {str(np.std(coverage))}')
        self.logger2.info(f'Used mean, std : {str(np.mean(used))} and {str(np.std(used))}')
        self.logger2.info(f'Corr Used mean, std : {str(np.mean(corr))} and {str(np.std(corr))}')
        self.logger2.info(f'Cost mean, std : {str(np.mean(cost))} and {str(np.std(cost))}')
        self.logger2.info(f'Fitness mean, std : {str(np.mean(fitness))} and {str(np.std(fitness))}')