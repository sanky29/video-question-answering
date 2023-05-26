import numpy as np
import cv2 as cv
import os
'''
matrix ---- predicted_label
|
|
true_label
|
|
'''
def get_matrix(true_label, predicted_label, n):
    matrix = np.zeros((n,n))
    for i,j in zip(true_label, predicted_label):
        matrix[i][j] += 1
    for i in range(n):
        if(matrix[i,i] == 0):
            matrix[i,i] += 0.0000000001
    return matrix

def f1_macro(matrix, n):

    tp_fp = matrix.sum(axis = 0)
    tp_fn = matrix.sum(axis = 1)
    
    precision = np.array([matrix[i][i] / tp_fp[i] for i in range(n)])
    recall = np.array([matrix[i][i]/ tp_fn[i] for i in range(n)])
    
    f1 = 2*precision*recall/(precision + recall)

    return sum(f1)/len(f1), f1, precision, recall
    
def f1_micro(matrix, n):
    
    tp_fp = matrix.sum(axis = 0)
    tp_fn = matrix.sum(axis = 1)
    
    precision = sum(matrix[i,i] for i in range(n))/ sum(tp_fp)
    recall = sum(matrix[i,i] for i in range(n))/ sum(tp_fn)

    return 2*precision*recall/(precision + recall)

def get_metric(true_label, predicted_label, n, indv = False):
    matrix = get_matrix(true_label, predicted_label, n)
    result = {}
    result['f1_macro'], f1, precision, recall = f1_macro(matrix, n)
    result['f1_micro'] = f1_micro(matrix, n)

    accuracy = sum([matrix[i,i] for i in range(n)])/np.sum(matrix)
    result['accuracy'] = accuracy
    if(indv):
        for i in range(n):
            result[i] = {'f1':f1[i], 'prec': precision[i], 'recall': recall[i]}
    return result

# true_label = [0,0,0,1,1,1,2,2,2]
# predicted_label = [0,0,0,1,1,1,2,2,2]
# print(get_metric(true_label, predicted_label,3))
