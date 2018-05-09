import pickle
from scipy.spatial import distance
from time import time
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from joblib import Parallel, delayed
import spams
import sys
import pandas as pd

def l2_distance1(a, b):
    dst = distance.euclidean(a,b)
    return dst

def l2_distance2(a, b):
    dist = np.linalg.norm(a-b)
    return dist

def search(query_input,database_input, identity, dist_metric="l2"):
	distance_list = []
	a = query_input
	if  dist_metric == "l2":
	    for b in database_input:
	        distance_list.append(l2_distance1(a, b))
	elif dist_metric=="spearmanr":
	    for b in database_input:
	        dist = scipy.stats.spearmanr(a,b)[0]
	        distance_list.append(dist)
	elif dist_metric=="minkowski":
	    for b in database_input:
	        X = np.vstack((a,b))
	        dist = scipy.spatial.distance.pdist(X, dist_metric,p=1)[0]
	        distance_list.append(dist)
	elif dist_metric=="overlap":
	    a[query_input!=0] = 1
	    database_input[database_input!=0] = 1
	    for b in database_input:
	        X = np.vstack((a,b))
	        dist = scipy.spatial.distance.pdist(X, "cosine")[0]
	        distance_list.append(dist)
	else:
	    for b in database_input:
	        X = np.vstack((a,b))
	        dist = scipy.spatial.distance.pdist(X, dist_metric)[0]
	        distance_list.append(dist)
	        
	sortIndexByDist = np.argsort(np.array(distance_list))
	return identity[sortIndexByDist]

def show_image(identity):
    file_name = identity[0][0].replace("\\","/")
    image_dir = "../data/lfw_image/"
    image_path = os.path.join(image_dir,file_name)
    query_image = imageio.imread(image_path)
    plt.figure(figsize=(2,2))
    plt.imshow(query_image)
    plt.show()

# calculate MAP
def p(k, query_identity, response):
#     precision at k
    query_name = query_identity[0][0].split("\\")[0]
    retrieved_name = np.array([response[i][0][0].split("\\")[0] for i in range(len(response))])
    correct = sum(query_name==retrieved_name[:k])
    return correct/k

def AveP(query_identity, response, identity):
    # aompute average precision
    query_name = query_identity[0][0].split("\\")[0]
    database_name = np.array([identity[i][0][0].split("\\")[0] for i in range(len(identity))])
    retrieved_name = np.array([response[i][0][0].split("\\")[0] for i in range(len(response))])
    number_of_relevance = sum(query_name == database_name)
#     print("number of relevance ",number_of_relevance)
    numerator = []
    for k in range(number_of_relevance):
        if retrieved_name[k] == query_name:
            rel = 1
        else:
            rel = 0
        score = p(k+1,query_identity,response)*rel
        numerator.append(score)
    return sum(numerator)/number_of_relevance


def MAP(numberOfQuery,AveP_list):
    return sum(AveP_list)/numberOfQuery

# use all queries for evaluation


def compute_l2_avep(query_index, AveP_list):
    query_feature = lfw["query_feature"][query_index]
    query_identity = lfw["query_identity"][query_index]
    database_feature = lfw["database_feature"]
    identity = lfw["database_identity"]
    response = search(query_feature, database_feature, identity, dist_metric=metric)
    ave_precision = AveP(query_identity, response, identity)

    AveP_list.append(ave_precision)
    return AveP_list



def dictionary_learning(patch_feature, lambda1=1, dictionary_size=100, batchsize=100,
                       posD=True):
    # input shape (feature size, sample size)
    X_patch = np.asfortranarray(patch_feature)
    param = { 'K' : dictionary_size, # learns a dictionary with 400 elements
             "mode":0,
              'lambda1' : lambda1, 'numThreads' : -1,
             "batchsize":batchsize,
             'posD':posD,
             "verbose":False
            }
    D = spams.trainDL(X_patch,**param)
    return D

def sparse_feature_coding(patch_feature, dictionary, lambda1=1, 
                          pos=True,  attri_enhance=False,
                          attri_weight=None):
    # lasso

    param = {
        'lambda1' : lambda1, # not more than 20 non-zeros coefficients
        'numThreads' : -1, 
        'mode' : 0, # penalized formulation
        'pos' : pos
    } 
    X_patch = np.asfortranarray(patch_feature)
    if attri_enhance == True:
    	# attribute enhancement
        attri_weight = np.asfortranarray(attri_weight)
        alpha = spams.lassoWeighted(X_patch, D = dictionary,W=attri_weight,
                                    **param)
    else:
        alpha = spams.lasso(X_patch, D = dictionary, return_reg_path = False, **param)
    dense_alpha = scipy.sparse.csr_matrix.todense(alpha)
    return dense_alpha


def compute_sc_avep(query_index,AveP_list, metric):
    query_feature = sparse_query_feature[query_index]
    query_identity = lfw["query_identity"][query_index]
    database_feature = sparse_database_feature
    identity = lfw["database_identity"]
    response = search(query_feature, database_feature, identity, dist_metric=metric)
    ave_precision = AveP(query_identity, response, identity)
    AveP_list.append(ave_precision)
    return AveP_list

def sparse_coding_with_identity(lfw):
	print("========= Sparse coding =========")
	X = np.array(lfw["database_feature"])
	X_patch = np.asfortranarray(X[:,:59]) # change type to Fortran array
	X_split = np.split(X,80, axis=1) # split to 80 patch

	# Hyper-parameter
	BATCHSIZE = 100
	DICTIONARY_SIZE = 100
	POS_D_CONSTRAINT =  True
	POS_LARS_CONSTRAINT =  True
	LAMBDA_DL = 1
	LAMBDA_LARS = 1

	# train 80 different dictionaries for all parts in the faces
	D_list = []
	alpha_list = []
	patch = 0
	for single_patch in X_split:
	    patch+=1
	    X_patch = single_patch.T
	    if patch%10 == 0:
	        print("patch:",patch)
	    # learn dictionary for single patch
	    D = dictionary_learning(X_patch,
	                            lambda1=LAMBDA_DL, 
	                            dictionary_size=DICTIONARY_SIZE,
	                            batchsize=BATCHSIZE, 
	                            posD=POS_D_CONSTRAINT)
	    alpha = sparse_feature_coding(X_patch, D, 
	                                  lambda1=LAMBDA_LARS,
	                                  pos=POS_LARS_CONSTRAINT).T
	    D_list.append(D)
	    alpha_list.append(alpha)

	sparse_database_feature = np.concatenate(np.array(alpha_list),axis=1) 
	print("sparse database shape:",sparse_database_feature.shape)

	# sparse encode query feature
	X_query = lfw["query_feature"]
	# train 80 different dictionaries for all parts in the faces
	alpha_list = []
	patch = 0
	dictionary_index = 0
	X_query_split = np.split(X_query,80, axis=1)
	for single_patch in X_query_split:
	    patch+=1
	    D = D_list[dictionary_index]
	    dictionary_index += 1
	    X_patch = np.asfortranarray(single_patch).T
	    if patch%10 == 0:
	        print("patch:",patch, "dictionary_index:",dictionary_index)
	    # learn dictionary for single patch
	    
	    alpha = sparse_feature_coding(X_patch, D, lambda1=LAMBDA_LARS,
	                                  pos=POS_LARS_CONSTRAINT).T
	    alpha_list.append(alpha)
	sparse_query_feature = np.concatenate(np.array(alpha_list),axis=1) 
	print("sparse query shape",sparse_query_feature.shape)

	return sparse_query_feature, sparse_database_feature
	


def attribute_weight(attribute_path, K, sigma):
	with open(attribute_path, "r") as f:
		lfw_attributes = f.readlines()


	def replace(string):
	    return string.replace(" ","_")
	def zero_pad(string):
	    return string.zfill(4)

	
	attributes = lfw_attributes[1].strip().split("\t")[1:]
	row = lfw_attributes[2].strip().split("\t")

	column_names = attributes
	attr_feature = []
	for i in lfw_attributes[2:]:
	    row = i.strip().split("\t")
	    attr_feature.append(row)
	attr_feature = np.array(attr_feature)

	attr_df = pd.DataFrame(attr_feature)
	attr_df.columns = column_names
	attri_identity = attr_df["person"].apply(replace) + "_"+ attr_df["imagenum"].apply(zero_pad)


	male_dict = {}
	for i in range(len(attr_df)):
	    male_dict[attri_identity[i]] = attr_df["Male"][i]

	database_id = [lfw["database_identity"][i][0][0].split("\\")[1][:-4] for i in range(len(lfw["database_identity"]))]
	query_id = [lfw["query_identity"][i][0][0].split("\\")[1][:-4] for i in range(len(lfw["query_identity"]))]


	database_attri = []
	query_attri = []
	for i, name in enumerate(database_id):
	    try:
	        database_attri.append(float(male_dict[name]))
	    except:
	        database_attri.append(0.)

	for i, name in enumerate(query_id):
	    try:
	        query_attri.append(float(male_dict[name]))
	    except:
	        query_attri.append(0.)

	# initial a vector
	a = np.array([-1 for _ in range(K//2)] + [1 for _ in range(K//2)])
	Z_database = []
	Z_query = []
	for attri in database_attri:
	    z = np.exp(abs(a-attri)/sigma)
	    Z_database.append(z)
	Z_database = np.array(Z_database)

	for attri in query_attri:
	    z = np.exp(abs(a-attri)/sigma)
	    Z_query.append(z)
	Z_query = np.array(Z_query)

	return Z_database.T, Z_query.T

def attribute_enhancement_sparse_coding(lfw, Z_database, Z_query):
	print("\n========== attribute enhancement sparse coding =============")
	X = np.array(lfw["database_feature"])
	X_patch = np.asfortranarray(X[:,:59]) # change type to Fortran array
	X_split = np.split(X,80, axis=1) # split to 80 patch

	# Hyper-parameter
	BATCHSIZE = 100
	DICTIONARY_SIZE = 100
	POS_D_CONSTRAINT =  True
	POS_LARS_CONSTRAINT =  True
	LAMBDA_DL = 1
	LAMBDA_LARS = 1

	# train 80 different dictionaries for all parts in the faces
	D_list = []
	alpha_list = []
	patch = 0
	for single_patch in X_split:
	    patch+=1
	    X_patch = single_patch.T
	    if patch%10 == 0:
	        print("patch:",patch)
	    # learn dictionary for single patch
	    D = dictionary_learning(X_patch,
	                            lambda1=LAMBDA_DL, 
	                            dictionary_size=DICTIONARY_SIZE,
	                            batchsize=BATCHSIZE, 
	                            posD=POS_D_CONSTRAINT)
	    alpha = sparse_feature_coding(X_patch, D, 
	                                  lambda1=LAMBDA_LARS, 
	                                  attri_enhance=True,
	                                  attri_weight=Z_database,
	                                  pos=POS_LARS_CONSTRAINT).T
	    D_list.append(D)
	    alpha_list.append(alpha)

	sparse_database_feature = np.concatenate(np.array(alpha_list),axis=1) 
	print("sparse database shape:",sparse_database_feature.shape)

	# sparse encode query feature
	X_query = lfw["query_feature"]
	# train 80 different dictionaries for all parts in the faces
	alpha_list = []
	patch = 0
	dictionary_index = 0
	X_query_split = np.split(X_query,80, axis=1)
	for single_patch in X_query_split:
	    patch+=1
	    D = D_list[dictionary_index]
	    dictionary_index += 1
	    X_patch = np.asfortranarray(single_patch).T
	    if patch%10 == 0:
	        print("patch:",patch, "dictionary_index:",dictionary_index)
	    # learn dictionary for single patch
	    
	    alpha = sparse_feature_coding(X_patch, D, lambda1=LAMBDA_LARS,
	                                  attri_enhance=True,
	                                  attri_weight=Z_query,
	                                  pos=POS_LARS_CONSTRAINT).T
	    alpha_list.append(alpha)
	sparse_query_feature = np.concatenate(np.array(alpha_list),axis=1) 
	print("sparse query shape",sparse_query_feature.shape)
	return sparse_query_feature, sparse_database_feature
	

if __name__ == "__main__":

	#####　part 1: l2_distance and calculate_map
	LWF_path = sys.argv[1]

	with open(LWF_path, "rb") as f:
		lfw = pickle.load(f)

	print("========== Baseline performance =========")
	dist_metrics = ["l2", "cosine","dice","correlation","minkowski",
                "sqeuclidean"]
	for metric in dist_metrics:
		print("distance metric:",metric)
		AveP_list = []
		AveP_list = Parallel(n_jobs=20)(delayed(compute_l2_avep)(query_index,AveP_list) for query_index in range(len(lfw["query_identity"])))
		AveP_list = np.array(AveP_list).flatten()
		print("MAP on {} queries {}:\n".format(len(lfw["query_identity"]),
	                                          MAP(len(lfw["query_identity"]),AveP_list)))


   	##### part 2&3: sparse coding with identity information
	sparse_query_feature, sparse_database_feature = sparse_coding_with_identity(lfw)

	### identity embedding
	database_id_list = np.array([lfw["database_identity"][i][0][0].split("\\")[0] for i in range(len(lfw["database_identity"]))])

	database_id_uni = np.array(sorted(list(set(database_id_list))))
	print("have {} different names in database".format(len(database_id_uni)))
	weight = 0.
	for identity in database_id_uni:
	    id_mask = database_id_list == identity
	    sub_feature = sparse_database_feature[id_mask]
	    id_mean_feature = sub_feature.mean(axis=0)
	    sparse_database_feature[id_mask,:] = sparse_database_feature[id_mask,:]*weight + id_mean_feature*(1-weight) # re-embed the features

	# use all queries for evaluation
	dist_metrics = ["overlap","cosine"]


	# experiment
	for metric in dist_metrics:
	    print("distance metric:",metric)
	    AveP_list = []
	    AveP_list = Parallel(n_jobs=20)(delayed(compute_sc_avep)(query_index,AveP_list,metric) for query_index in range(len(lfw["query_identity"])))
	    AveP_list = np.array(AveP_list).flatten()
	    print("MAP on {} queries {}".format(len(lfw["query_identity"]),
	                                              MAP(len(lfw["query_identity"]),AveP_list)))

	##### part 4: attribute enhancement
	attribute_path = sys.argv[2]
	Z_database, Z_query = attribute_weight(attribute_path, K=100, sigma=120)
	sparse_query_feature, sparse_database_feature = attribute_enhancement_sparse_coding(lfw, Z_database, Z_query)
	# use all queries for evaluation
	dist_metrics = ["overlap","cosine"]

	# experiment
	for metric in dist_metrics:
	    print("distance metric:",metric)
	    AveP_list = []
	    AveP_list = Parallel(n_jobs=20)(delayed(compute_sc_avep)(query_index,AveP_list,metric) for query_index in range(len(lfw["query_identity"])))
	    AveP_list = np.array(AveP_list).flatten()
	    print("MAP on {} queries {}".format(len(lfw["query_identity"]),
	                                              MAP(len(lfw["query_identity"]),AveP_list)))