
def import_data(data='hanoi', size=1000):
    with open(f"data/{data}.asc", "r") as f:
        f.readline()
        f.readline()
        xllcorner = float(f.readline()[9:-1])
        yllcorner = float(f.readline()[9:-1])
        cellsize = float(f.readline()[8:-1])
        NODATA_value = f.readline()
        data_asc = f.readlines()
        data_asc[0] = data_asc[0][13:]
        data_asc[0] = list(map(float, data_asc[0].split()))
        for i in range(1, len(data_asc)):
            data_asc[i] = list(map(float, data_asc[i].split()))
            data_asc[i - 1].append(data_asc[i].pop(0))
        data_asc.pop()
        cell = int(size // 25)
        # expand from the bottom left corner
        data_asc = data_asc[-cell:] 
        for i in range(len(data_asc)):
            data_asc[i] = data_asc[i][:cell]
        def z(x, y):
            x = int(x/25) if 0<=x/25<=cell-1 else cell-1
            y = int(x/25) if 0<=y/25<=cell-1 else cell-1
            return data_asc[x][y]
        return z, data_asc
