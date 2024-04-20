import json
import numpy as np
import pickle

grille, taille_grille = [], 0
with open('type.pkl', 'rb') as f:
    data = pickle.load(f)
    grille, len_x, len_y = data['grille'], data['len_x_grid'], data['len_y_grid']

grid = np.array(grille).reshape(len_x-1, len_y-1)
res = []

def replace_with_big_square(i, j, size, value):
    s1_gps = grid[i][j]['s1_gps']
    s2_gps = grid[i][j + size - 1]['s4_gps']
    s4_gps = grid[i + size - 1][j]['s2_gps']
    s3_gps = grid[i + size - 1][j + size - 1]['s3_gps']
    dbm_somme = 0
    dbm_count = 0
    for row in range(i, i + size):
        for col in range(j, j + size):
            if grid[row][col]['dbm_moy'] != 0:
                dbm_somme += grid[row][col]['dbm_moy']
                dbm_count += len(grid[row][col]['releves'])

    dbm_moy = dbm_somme / dbm_count if dbm_count != 0 else 0.
    res.append({
        's1_gps': s1_gps,
        's2_gps': s2_gps,
        's3_gps': s3_gps,
        's4_gps': s4_gps,
        'dbm_moy': dbm_moy,
        'type': value
    })

types = ['URB', 'PER', 'RUR']
for i in range(0, grid.shape[0]-4, 4):
    for j in range(0, grid.shape[1]-4, 4):
        nb_type = [0, 0, 0, 0]
        for row in range(i, i + 4):
            for col in range(j, j + 4):
                nb_type[0] += 1 if grid[row][col]['type'] == types[0] else 0
                nb_type[1] += 1 if grid[row][col]['type'] == types[1] else 0
                nb_type[2] += 1 if grid[row][col]['type'] == types[2] else 0
                nb_type[3] += 1 if grid[row][col]['type'] == None else 0
        
        max_index = nb_type.index(max(nb_type))
        if max_index == 2:
            replace_with_big_square(i, j, 4, types[2])
        else:
            for row in range(i, i + 4, 2):
                for col in range(j, j + 4, 2):
                    nb_type_2 = 0
                    for row2 in range(row, row + 2):
                        for col2 in range(col, col + 2):
                            nb_type_2 += 1 if grid[row2][col2]['type'] == types[1] else 0
                    if nb_type_2 >= 2:
                        replace_with_big_square(row, col, 2, types[1])
                    else:
                        for row2 in range(row, row + 2):
                            for col2 in range(col, col + 2):
                                if grid[row2][col2]['type'] != None:
                                    res.append({
                                        's1_gps': grid[row2][col2]['s1_gps'],
                                        's2_gps': grid[row2][col2]['s2_gps'],
                                        's3_gps': grid[row2][col2]['s3_gps'],
                                        's4_gps': grid[row2][col2]['s4_gps'],
                                        'dbm_moy': grid[row2][col2]['dbm_moy'],
                                        'type': grid[row2][col2]['type']
                                    })

print(len(res))

with open("etape3.pkl", "wb") as f:
    pickle.dump({'grille': res}, f)
