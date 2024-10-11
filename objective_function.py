import math
import random
import numpy as np
from dataset import import_data

class ObjectiveFunction:
    def __init__(self, hmv, hms, targets, types=2, radius=[0, 0], alpha1=1, alpha2=0, beta1=1, beta2=0.5,
                 threshold=0.9, w=100, h=100, cell_h=10, cell_w=10, data="hanoi"):
        """
            :param hmv: harmony vector size
            :param hms: harmony memory size
            :param targets: position of target points
            :param types: number of different node types
            :param radius: radius for each node type
            :param alpha1, alpha2, beta1, beta2: parameter for calculating Pov
            :param threshold: threshold for Pov
            :param h, w: height and width of AoI
            :param cell_h, cell_w: height and width of cell
        """
        self.hmv = hmv
        self.hms = hms
        self.targets = targets
        self.radius = radius
        self.ue = []
        for r in radius:
            self.ue.append(r / 2)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.threshold = threshold
        self.type_sensor = range(types)
        self.w = w
        self.h = h

        self.cell_h = cell_h
        self.cell_w = cell_w

        self.cell_r = math.sqrt((cell_h / 2) ** 2 + (cell_w / 2) ** 2)
        self.no_cell = (self.w // self.cell_w) * (self.h // self.cell_h)
        self.min_noS = self.w * self.h // ((max(self.radius) ** 2) * 9)
        self.max_noS = self.w * self.h // (min(self.radius) ** 2)
        self.z = import_data(data)[0] if data not in ['aoi1', 'aoi2'] else None
        if self.z is None:
            self.max_diagonal = max(
                [self._distance([self.w, self.h], [self.radius[i] + self.ue[i], self.radius[i] + self.ue[i]]) for i in
                range(len(self.radius))])
        else:
            self.max_diagonal = max(
                [self._distance3d([self.w, self.h, self.z(self.w, self.h)], [self.radius[i] + self.ue[i], self.radius[i] + self.ue[i], self.z(self.radius[i] + self.ue[i], self.radius[i] + self.ue[i])]) for i in
                range(len(self.radius))])
        

    def get_hms(self):
        return self.hms

    def get_radius(self):
        return self.radius

    def _senscost(self, node_list):
        x = (len(node_list) - self.min_noS) / (self.max_noS - self.min_noS)
        return 1 / (x + 0.1)

    def get_coverage_ratio(self, node_list, type_assignment):
        return self._coverage_ratio(node_list, type_assignment)[0]

    def _coverage_ratio(self, node_list, type_assignment):
        """
            Return coverage_ratio and list of covered target
        """
        target_corvered = []
        for target in self.targets:
            Pov = 1
            for index, sensor in enumerate(node_list):
                p = self._psm(sensor, target, type=type_assignment[index])
                if p == 0:
                    continue
                Pov *= (1 - p)
            Pov = 1 - Pov
            if Pov >= self.threshold:
                target_corvered.append(target)

        return len(target_corvered) / self.no_cell, target_corvered

    def _md(self, node_list, type_assignment):
        min_dist_sensor = float('+inf')
        for ia, a in enumerate(node_list):
            for ib, b in enumerate(node_list):
                if a != b:
                    if self.z is None:
                        min_dist_sensor = min(min_dist_sensor, self._distance(a, b))
                    else:
                        z1 = self.z(a[0], a[1])
                        a_3d = [a[0], a[1], z1]

                        z2 = self.z(b[0], b[1])
                        b_3d = [b[0], b[1], z2] 
                        min_dist_sensor = min(min_dist_sensor, self._distance3d(a_3d, b_3d))

        if min_dist_sensor == float('+inf'):
            min_dist_sensor = 0.0
        return min_dist_sensor / (self.max_diagonal)

    def intersection_area(self, d, R, r):
        if d <= abs(R - r):
            # One circle is entirely enclosed in the other.
            return np.pi * min(R, r) ** 2
        if d >= r + R:
            # The circles don't overlap at all.
            return 0

        r2, R2, d2 = r ** 2, R ** 2, d ** 2
        alpha = np.arccos((d2 + r2 - R2) / (2 * d * r))
        beta = np.arccos((d2 + R2 - r2) / (2 * d * R))
        return (r2 * alpha + R2 * beta -
                0.5 * (r2 * np.sin(2 * alpha) + R2 * np.sin(2 * beta))
                )
    
    def intersection_area_3d(self, d, R, r):
        # Check if the spheres are too far apart to overlap
        if d >= R + r:
            return 0
        
        # Check if one sphere is completely inside the other
        if d <= abs(R - r):
            # Return the volume of the smaller sphere
            return (4/3) * math.pi * min(R, r)**3
        
        # Calculate the distance between the centers of the spheres projected onto the x-axis
        x = (d**2 + r**2 - R**2) / (2 * d)
        
        # Calculate the height of the cap of the larger sphere
        h = R - x
        
        # Calculate the volumes of the overlapping portions of the spheres
        overlap_volume_1 = (1/3) * math.pi * h**2 * (3 * R - h)
        overlap_volume_2 = (1/3) * math.pi * (r**2 * (d - h))
        
        return overlap_volume_1 + overlap_volume_2

    def _regularization1(self, node_list, type_assignment):
        """Reg 1

        Args:
            node_list (list): list of sensors
            type_assignment (list(int)): sensor types

        Returns:
            int: function value
        """
        overlap = 0
        total_s = 0
        for i1, node1 in enumerate(node_list):
            for i2, node2 in enumerate(node_list):
                if i1 >= i2:
                    continue
                if self.z is None:
                    d = self._distance(node1, node2)
                    intersection_area = self.intersection_area(d, self.radius[type_assignment[i1]],
                                                            self.radius[type_assignment[i2]])
                    overlap += intersection_area
                    total_s += math.pi * (self.radius[type_assignment[i1]] ** 2 + self.radius[
                        type_assignment[i2]] ** 2)  # - intersection_area
                else:
                    d = self._distance3d(node1, node2)
                    intersection_area = self.intersection_area_3d(d, self.radius[type_assignment[i1]],
                                                            self.radius[type_assignment[i2]])
                    overlap += intersection_area
                    total_s += 4/3 * math.pi * (self.radius[type_assignment[i1]] ** 3 + self.radius[
                        type_assignment[i2]] ** 3)  # - intersection_area

        return 1 - overlap / total_s  # add multiplication?

    ## Keep overlap sensor
    # def _regularization1(self, node_list, type_assignment):
    #     no_interception = 0
    #     for ia, a in enumerate(node_list):
    #         for ib, b in enumerate(node_list):
    #             if  a!= b:
    #                 if self._distance(a, b) < (self.radius[type_assignment[ia]] + self.radius[type_assignment[ib]]):
    #                         no_interception += 1
    #     no_interception = no_interception/2
    #     n = len(node_list)
    #     no_interception = no_interception/(n*(n-1)/2)

    #     return 1-no_interception

    ## Keep every cell has 1 sensor (HARD)
    def _regularization2(self, node_list, type_assignment):
        node_in_cells = []
        for target in self.targets:
            s = 0
            for node in node_list:
                if self._distance(target, node) <= self.cell_r:
                    s += 1
            if s <= 1:
                node_in_cells.append(s)

        return len(node_in_cells) / self.no_cell

    ##  Keep target inside one Sensor range
    def _regularization4(self, node_list, type_assignment):
        node_per_cells = []
        for target in self.targets:
            s = 0
            for inode, node in enumerate(node_list):
                if self._distance(target, node) <= type_assignment[inode]:
                    s += 1
            if s == 1:
                node_per_cells.append(s)

        return len(node_per_cells) / self.no_cell

    ## Keep no sensor but count type assigment
    def _regularization3(self, node_list, type_assignment):
        # no_used = len(node_list)
        no_used_convert = sum(type_assignment) + (len(type_assignment) - sum(type_assignment)) / 2
        return 1 - no_used_convert / self.hmv

    def get_fitness(self, harmony):

        used = []

        for id, sensor in enumerate(harmony):
            if sensor[0] < 0 or sensor[1] < 0:
                continue
            else:
                used.append(sensor)

        if len(used) < self.min_noS:
            return (float('-inf'), 0), []

        type_traces = [[random.choice([0, 1]) for j in range(len(used))] for i in range(1)]  # 1 = #comb

        best_fitness = float('-inf')
        best_coverage_ratio = 0
        best_trace = None

        for type_trace in type_traces:
            coverage_ratio, _ = self._coverage_ratio(used, type_trace)
            # fitness = (self._senscost(used) ** 1) * ((coverage_ratio) ** 1) * (self._md(used, type_trace)  ** 1)

            fitness = (coverage_ratio * self._md(used, type_trace) * self.max_diagonal * 0.5) / len(used)
            if fitness > best_fitness:
                best_fitness = fitness
                best_coverage_ratio = coverage_ratio
                best_trace = type_trace

        return (best_fitness, best_coverage_ratio), best_trace

    def original_fitness(self, node_list, type_assignment):
        return self.get_coverage_ratio(node_list, type_assignment) * self._senscost(node_list) * self._md(node_list, type_assignment)

    def _distance(self, x1, x2):
        return math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
    
    def _distance3d(self, x1, x2):
        return math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2 + (x1[2] - x2[2]) ** 2)

    def _psm(self, x, y, type):
        if self.z is None:
            distance = self._distance(x, y)
        else:
            z1 = self.z(x[0], x[1])
            x_3d = [x[0], x[1], z1]

            z2 = self.z(y[0], y[1])
            y_3d = [y[0], y[1], z2] 
            distance = self._distance3d(x_3d, y_3d)

        if distance < self.radius[type] - self.ue[type]:
            return 1
        elif distance > self.radius[type] + self.ue[type]:
            return 0
        else:
            lambda1 = self.ue[type] - self.radius[type] + distance
            lambda2 = self.ue[type] + self.radius[type] - distance
            lambda1 = math.pow(lambda1, self.beta1)
            lambda2 = math.pow(lambda2, self.beta2)
            return math.exp(-(self.alpha1 * lambda1 / lambda2 + self.alpha2))
