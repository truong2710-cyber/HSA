import math
from extract import extract
import argparse
from dataset import import_data

def distance(x, y, is_3d=False):
    if is_3d is False:
        x1, y1 = x[0], x[1]
        x2, y2 = y[0], y[1]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    else:
        x1, y1, z1 = x[0], x[1], x[2]
        x2, y2, z2 = y[0], y[1], y[2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

interval = 0.75 # s
v_s_base = 5 # V  
i_s_base = 0.001 # A
p_s_base = 0.005 # 0.5
p_s = [p_s_base, 4*p_s_base]
package_size = 4000 # bits

e_f = 10**(-11)
e_m = 13 * 10**(-16)
e_elec = 5 * 10 ** (-8)

d0 = math.sqrt(e_f/e_m)

H, W = (100, 100)
base = [0.4*W, 0.4*H]
cell_W, cell_H = (10, 10)
R = [5, 10]
UE = [x/2 for x in R]

targets = []
for h in range(int(abs(H/cell_H))):
    for w in range(int(abs(W/cell_W))):
        targets.append([w*cell_W + cell_W/2, h*cell_H + cell_H/2])

def calculate_energy(harmony, type_trace, args):
    z = import_data(args.data)[0] if args.data not in ['aoi1', 'aoi2'] else None
    if args.is_3d:
        global base
        base.append(z(base[0], base[1]))
    comm_energy = 0
    sensing_energy = 0
    for i, sensor in enumerate(harmony):
        if args.is_3d:
            sensor.append(z(sensor[0], sensor[1]))
        num_packages = 0
        for target in targets:
            if args.is_3d:
                target.append(z(target[0], target[1]))
            if distance(sensor, target, args.is_3d) <= R[type_trace[i]] + UE[type_trace[i]]:
                num_packages += 1
        d = distance(sensor, base, args.is_3d)
        trans_energy = package_size * (e_elec + e_f * d**2) if d < d0 else package_size * (e_elec + e_m * d**4)
        rec_energy = package_size * e_elec

        comm_energy += (trans_energy + rec_energy) * num_packages

        sensing_energy += p_s[type_trace[i]] * interval 
        

    print(f"Total energy in one interval: {comm_energy + sensing_energy} J")
    return comm_energy + sensing_energy


def main():
    parser = argparse.ArgumentParser(description="Calculate energy")
    parser.add_argument('--path', type=str, help='Path to the log file')
    parser.add_argument('--is_3d', action='store_true', help='is 3d')
    parser.add_argument('--data', type=str, default='aoi1', help='data to use')

    args = parser.parse_args()
    print(args.path)
    best_harmony_list, type_matches = extract(args.path)
    # breakpoint()
    energies = []
    for harmony, type_trace in zip(best_harmony_list, type_matches):
        harmony = [x for x in harmony if x[0] > 0 and x[1] > 0] # W >= x[0] >= 0 and H >=x[1] >= 0 x[0] > 0 and x[1] > 0
        assert len(harmony) == len(type_trace), 'not equal length'
        energy = calculate_energy(harmony, type_trace, args)
        energies.append(energy)
    
    print(f"Average energy consumption in an interval: {sum(energies)/len(energies)} J")

if __name__ == "__main__":
    main()