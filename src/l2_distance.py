# given two matrices, where each row is the features of one image, you will compute and return the pairwise L2-distance of query images and database images.
import numpy as np

def l2_distance2(a, b):
    dist = np.linalg.norm(a-b)
    return dist

def search(query_feature,database_feature, identity):
    distance_list = []
    a = query_feature
    for b in database_feature:
        distance_list.append(l2_distance2(a, b))
    sortIndexByDist = np.argsort(np.array(distance_list))
    return identity[sortIndexByDist]

def pairwise_l2_dist(query_feature, database_feature):
	distance_list = []
    a = query_feature
    for b in database_feature:
        distance_list.append(l2_distance2(a, b))
     return distance_list