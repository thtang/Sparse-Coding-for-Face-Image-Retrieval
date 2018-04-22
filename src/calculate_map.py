# compute and return the mean average precision based on the distance computed by l2_distance.py

import numpy as np

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