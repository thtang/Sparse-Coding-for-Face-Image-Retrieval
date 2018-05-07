from l2_distance import l2_distance #l2_distance.py
from sparse_coding import sparse_coding #sparse_coding.py
from calculate_map import calculate_map #calculate_map.py
import pickle

with open(lfw_path, 'rb') as f:
    lfw = pickle.load(f)

#part 1: l2_distance and calculate_map
results_l2 = l2_distance(lfw['query_feature'], lfw['database_feature'])
mean_ap = calculate_map(results_l2, lfw['query_name'], lfw['database_name'])
#part 2: sparse_coding
q_sparse, db_sparse = sparse_coding(lfw['query_feature'], lfw['database_feature'], param) #build sparse dict and lookup using spams library
results_sparse = distance(q_sparse, db_sparse) #you can use l2_distance in part1 or try any distance metric, like cos, l1 

#part 3, 4
