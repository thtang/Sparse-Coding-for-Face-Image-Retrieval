# given two matrices, where each row is the features of one image, you will compute and return the pairwise L2-distance of query images and database images.
import numpy as np
import sys
import pickle

def pairwise_l2_dist(a, b):
    dist = np.linalg.norm(a-b)
    return dist

def search(query_feature,database_feature, identity):
    distance_list = []
    a = query_feature
    for b in database_feature:
        distance_list.append(pairwise_l2_dist(a, b))
    sortIndexByDist = np.argsort(np.array(distance_list))
    return identity[sortIndexByDist]

def l2_distance(query_feature, database_feature):
    distance_list = []
    a = query_feature
    for b in database_feature:
        distance_list.append(pairwise_l2_dist(a, b))
    return distance_list

if __name__ == "__main__":
    lfw_file_path = sys.argv[1]
    with open(lfw_file_path, "rb") as f:
        lfw = pickle.load(f)
    response = search(lfw['query_feature'][1], lfw['database_feature'], lfw["database_identity"])
    print("top ten response:",response[:10])