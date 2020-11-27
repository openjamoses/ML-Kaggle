import numpy as np
import pandas as pd
import multiprocessing
import itertools
from functools import partial
import operator
import math
import functools
from concurrent.futures import ProcessPoolExecutor
import re
import string
from collections import Counter
from collections import defaultdict
import random
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import f1_score
import copy


def read_from_csv(path):
    data = pd.read_csv(path)
    data = data.dropna(axis=1,how='all')
    return (data.to_numpy().T).tolist()

def get_range(min_value, max_value, step):
    """ 
    return a list of values ranging from start to max_value according to step
    min_value: The minimum value
    max_value: The max value in the range
    step: the step to increment the start value
    """
    if min_value > max_value:
        raise ValueError("max_value can not be inferior to min_value!")
    if step <= 0:
        raise ValueError("step can not be negative or zero!")
    result = []
    while min_value <= max_value:
        result.append(min_value)
        min_value += step
    return result


train_np = np.load('Dataset/train_images.npy', allow_pickle=True, encoding="bytes")
test_np = np.load('Dataset/test_images.npy', allow_pickle=True, encoding="bytes")
train_labels = read_from_csv('Dataset/train_labels.csv')
Y_train = train_labels[1]

train_labels_dict = dict(zip(train_labels[0], train_labels[1])) 
train_list = train_np.tolist()

test_list = test_np.tolist()
#print('')

def get_inner_np_array(img_list_item):
    """
    This function takes a list of images and returns the list of the inner list
    img_list: a list  [1, nparra(...)]
    return a list []
    """
    return img_list_item[1].tolist() if (img_list_item and len(img_list_item) > 1 ) else []

def get_inner_X(img_list_item):
    """
    This function takes a list of images and returns the list of the inner list
    img_list: a list  [1, nparra(...)]
    return 1
    """
    return img_list_item[0]

def get_inner_np_array_all(img_list, func):
    """
    This function takes a list of list of images and returns the list of the inner list
    img_list: a list [[0, nparra(...)], [1, nparra(...)]]
    return a list [[], []]
    """
    p = multiprocessing.Pool()
    result = p.map(func, img_list)
    p.close()
    p.join()
    p.terminate()
    return result



def get_decision_tree_classifier(X_data, Y_data, criterion_par=7):
    """
    this function creates an instance of decision tree model and train it
    X_data: The training data [[a line of data of size k features],...]
    Y_data: The corresponding classes in a list [0,1,0,1,0,1]
    criterion_par: The function to measure the quality of a split 'gini' or 'entropy'
    returns a tuple(decision_tree_classifier, trained_model)
    """
    #decision_tree_classifier = DecisionTreeClassifier(criterion=criterion_par,max_depth=53)
    decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=criterion_par)
    decision_tree_classifier.fit(X_data, Y_data)
    return decision_tree_classifier


#   Linear SVM classifier                  A finir vendredi

def get_linear_SVM_classifier(X_data, Y_data, C_par=1.0):
    """
    this function creates an instance of linear svm model and train it
    X_data: The training data [[a line of data of size k features]]
    Y_data: The corresponding classes in a list [0,1,0,1,0,1]
    C_par: Regularization parameter. The strength of the regularization is inversely proportional to C.
    Must be strictly positive. The penalty is a squared l2 penalty
    returns a tuple(linear_SVM_classifier, trained_model)
    """
    linear_SVM_classifier = svm.LinearSVC(C = C_par, max_iter=15000)
    linear_SVM_classifier.fit(X_data, Y_data)
    return linear_SVM_classifier

def classification_applier(hyper_paramater, func_classifier, X_data, Y_data):
    """
    this function creates an instance of func_classfier model and train it
    hyper_paramater: the hyperparameter for this classifier
    func_classifier: the function of the classifier to use
    X_data: The training data [[a line of data of size k features]]
    Y_data: The corresponding classes in a list [0,1,0,1,0,1]
    returns a tuple (hyper_parameter, (func_classifier, trained_model))
    """
    return hyper_paramater, func_classifier(X_data, Y_data, hyper_paramater)

def classification_applier_pool(hyper_paramater_list, func_classifier, X_data, Y_data):
    """
    this function creates an instance of func_classfier model and train it
    hyper_paramater_list: the hyperparameter for this classifier
    func_classifier: the function of the classifier to use
    X_data: The training data [[a line of data of size k features]]
    Y_data: The corresponding classes in a list [0,1,0,1,0,1]
    returns a list of tuples [(hyper_parameter, (func_classifier, trained_model)),...]
    """
    p = multiprocessing.Pool(5)
    result = p.map(partial(classification_applier, func_classifier=func_classifier, X_data=X_data, Y_data=Y_data), hyper_paramater_list)
    p.close()
    p.join()
    p.terminate()
    return result

def prediction_metrics(tuple_var, X_data, Y_data, average_par):
    """
    this function creates an instance of func_classfier model and train it
    tuple_var: a tuple (hyper_parameter, (func_classifier, trained_model))
    X_data: The test or validation data [[a line of data of size k features]]
    Y_data: The corresponding classes in a list [0,1,0,1,0,1]
    average_par:This parameter is required for multiclass/multilabel targets 'binary' 'macro' 'micro' 'weighted' 'samples'
    return a tuple (hyper_parameter, (prediction, f1_score))
    """
    return tuple_var[0], f1_score(Y_data, tuple_var[1].predict(X_data), average=average_par)

def prediction_metrics_pool(tuple_var_list, X_data, Y_data, average_par='macro'):
    """
    this function creates an instance of func_classfier model and train it
    hyper_paramater_list: the hyperparameter for this classifier
    tuple_var_list: a list of tuples (hyper_parameter, (func_classifier, trained_model))
    X_data: The test or validation data [[a line of data of size k features]]
    Y_data: The corresponding classes in a list [0,1,0,1,0,1]
    average_par:This parameter is required for multiclass/multilabel targets 'binary' 'macro' 'micro' 'weighted' 'samples'
    return a list of tuples [(hyper_parameter, (prediction, f1_score)),...]
    """
    p = multiprocessing.Pool(10)
    result = p.map(partial(prediction_metrics, X_data=X_data, Y_data=Y_data, average_par=average_par), tuple_var_list)
    p.close()
    p.join()
    p.terminate()
    return result


def get_best_train_classifier(tup_var_list, tup_var_list_trained):
    """
    This function takes:
    tup_var_list: a list of tuples [(hyper_parameter, f1_score)...]
    tup_var_list_trained: a list of tuples [(hyper_parameter, (func_classifier, trained_model)),...]
    hyper_param: the hyper_param
    returns a tuple (the tuple with the best f1_score, the corresponding trained model)
    """
    hyper_param = max(tup_var_list, key=operator.itemgetter(1))
    return hyper_param, (dict(tup_var_list_trained))[hyper_param[0]]

X_train = get_inner_np_array_all(train_list, get_inner_np_array)

X_test = get_inner_np_array_all(test_list, get_inner_np_array)

Y_test_id = get_inner_np_array_all(test_list, get_inner_X)

def get_from_dict(Y_raw_item, dict_Y_par):
    """
    This value takes an id of Y
    Y_raw: a list id Ids 0
    Y_dict: a dictionnary{key=id:Values='snail'}
    and returns the corresponding Value of none if not in dictionnary
    """
    return dict_Y_par.get(Y_raw_item, None)

def map_Y_id_to_value_pool(Y_raw, dict_Y):
    """
    This value takes a list of ids of Y
    Y_raw: a list id Ids [0,1,2,3]
    Y_dict: a dictionnary{key=id:Values='snail'}
    and returns a list of corresponding Values ['', '', '', '']
    """
    p = multiprocessing.Pool()
    result = p.map(partial(get_from_dict, dict_Y_par=dict_Y), Y_raw)
    p.close()
    p.join()
    p.terminate()
    return result


#knn classifier:

def get_tp_fp_tn_fn(tup_var):
    """ 
    this function takes a tuple
    tup[0] = a prediction
    tup[1] = a real value
    and returns a tuple:
    (1,0,0,0) for true positive
    (0,1,0,0) for true negative
    (0,0,1,0) for false positive
    (0,0,0,1) for false negative
    """
    y_line_est, y_line = tup_var
    #if true positive-0
    if y_line_est == y_line and y_line == 1:
        return (1,0,0,0)
    #if true negative-1
    elif y_line_est == y_line and y_line == 0:
        return (0,1,0,0)
    #if false positive-2
    elif y_line_est != y_line and y_line_est == 1:
        return (0,0,1,0)
    #if false negative-3
    elif y_line_est != y_line and y_line_est == 0:
        return (0,0,0,1)

def get_acc_prec_rec_f1_metrics(Y_est_var, Y_var):
    """
    this function takes
    Y_est_var: a vector of prediction
    Y_var: a vector of real value
    and returns a tuple (accuracy, precision, recall, f1_measure)
    """
    p = multiprocessing.Pool()
    result = p.map(get_tp_fp_tn_fn, zip(Y_est_var, Y_var))
    p.close()
    p.join()
    p.terminate()
    result_temp = functools.reduce(lambda x, y,: (x[0] + y[0] , x[1] + y[1], x[2] + y[2], x[3] + y[3]), result)
    accuracy = (result_temp[0] + result_temp[1]) / (result_temp[0] + result_temp[1] + result_temp[2] + result_temp[3])
    precision = result_temp[0] / (result_temp[0] + result_temp[2])
    recall = result_temp[0] / (result_temp[0] + result_temp[3])
    f1_measure = (2 * precision * recall) / (precision + recall)
    #return accuracy, precision, recall, f1_measure, (result_temp[0] + result_temp[1] + result_temp[2] + result_temp[3]) == len(Y_var)
    return accuracy, precision, recall, f1_measure


def euclidean_distance(vector1, vector2):
    """
    this function returs the euclidian distance of vector1 and vector2
    """
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    return math.sqrt(sum(dist))
    

def distance(tup_var, line_q, func_distance):
    """ 
    This function takes:
    line_q : [line of features in query]
    tup_var :  a tuple (index of the line X, [a line of features in X])
    func_distance : the function to calculate the distance
    and returns (index of the line X, distance)
    """
    i = tup_var[0]
    train_line = tup_var[1]
    return i, func_distance(train_line, line_q)


def knn_(query_line, data_var, Y_var, func_distance_var, k_var):
    """
    This function takes:
    query_line: [line of features in query]
    data_var: train data
    Y_var: the list of class of train data
    func_distance_var: tthe function to calculate the distance
    k_var: number k of neighbors
    
    returns (predicted class, index of query_line)
    """
    p = multiprocessing.Pool()
    result = p.map(partial(distance, line_q=query_line[1], func_distance=func_distance_var), list(enumerate(data_var)))
    p.close()
    p.join()
    p.terminate()
    k_nearest_distances_and_indices = sorted(result, key=lambda x: x[1])[:k_var]
    k_nearest_labels = [Y_var[i] for i, _ in k_nearest_distances_and_indices]
    return 1 if sum(k_nearest_labels)/k_var > 0.5 else 0, query_line[0]


def knn(k_par, query_var, data_par, Y_par, func_distance_par):
    """
    This function takes:
    query_var: the data to be classified validation, test
    data_par: train data
    Y_par: the list of class of train data
    func_distance_par: tthe function to calculate the distance
    k_par: number k of neighbors
    returns (predicted class, index of query_line)
    """
    resultats = []
    with ProcessPoolExecutor(max_workers = None) as executor:
        results = executor.map(partial(knn_, data_var=data_par, Y_var=Y_par, func_distance_var=func_distance_par, k_var=k_par), list(enumerate(query_var)))
        for e in results:
            resultats.append(e)
    return resultats

def knn_performance_report(k_par, query_var, data_par, Y_par, func_distance_par, Y_val_par):
    """
    This function takes:
    query_var: the data to be classified validation, test
    data_par: train data
    Y_par: the list of class of train data
    func_distance_par: tthe function to calculate the distance
    k_par: number k of neighbors
    Y_val_par: Y_val_par
    returns a tuple (k, (accuracy, precision, recall, f1_measure))
    """
    temp_result = knn(k_par, query_var, data_par, Y_par, func_distance_par)
    return k_par, get_acc_prec_rec_f1_metrics(list(zip(*temp_result))[0], Y_val_par)

def knn_performance_report_range_k(k_range_par, query_var, data_par, Y_par, func_distance_par, Y_val_par):
    """
    This function takes:\n
    k_range_par: list of ks (k number of neighbors)\n
    query_var: the data to be classified validation, test\n
    data_par: train data\n
    Y_par: the list of class of train data\n
    func_distance_par: the function to calculate the distance\n
    Y_val_par: Y_val_par\n
    returns a dict {k:(accuracy, precision, recall, f1_measure)}
    """
    resultats = []
    with ProcessPoolExecutor(max_workers = None) as executor:
        results = executor.map(partial(knn_performance_report, query_var=query_var, data_par=data_par, Y_par=Y_par, func_distance_par=func_distance_par, Y_val_par=Y_val_par), k_range_par)
        for e in results:
            resultats.append(e)
    return dict(resultats)

def get_best_k(performance_dict, criterion_key):
    """ 
    This function takes in:
    performance_dict: dictionnaries where the key is the k number and values are tuples (accuracy, precision, recall, f1_measure)
    {1: (0.23625, 0.2358974358974359, 0.2271604938271605, 0.23144654088050315), 2: (0.37125, 0.21176470588235294, 0.08888888888888889, 0.12521739130434784)}
    criterion: the performance criterion considered
    0: accuracy
    1: precision
    2: recall
    3: f1_measure
    this function returns the best item in the dictionnaries according to the criterion (k, (accuracy, precision, recall, f1_measure))
    """
    if criterion_key<=3 and criterion_key>=0:
        return max(list(performance_dict.items()),key=lambda item:item[1][criterion_key])
    else:
        return None

Y_test = map_Y_id_to_value_pool(Y_test_id, train_labels_dict)

#['gini','entropy'] 18 => 0.338
decision_tree_class_train = classification_applier_pool(list(range(15, 28, 1)), get_decision_tree_classifier, X_train, Y_train)
decision_tree_class_metric_valid = prediction_metrics_pool(decision_tree_class_train, X_test, Y_test, 'macro')

decision_tree_class_best_metric = get_best_train_classifier(decision_tree_class_metric_valid, decision_tree_class_train)
#decision_tree_class_metric_test = prediction_metrics((decision_tree_class_best_metric[0][0], decision_tree_class_best_metric[1]), X_test, Y_test, 'macro')
decision_tree_class_metric_train = prediction_metrics((decision_tree_class_best_metric[0][0], decision_tree_class_best_metric[1]), X_train, Y_train, 'macro')

#list(range(90, 111, 1))list(range(8, 11, 1))
#[x/10 for x in range(90, 111,1)] best 10 => 0.4285
#[x/100 for x in range(945, 956,1)]get_range(9.5, 10.5, 0.05) best 9.5 => 0.4290 best 0.2 => 0.438 0.26 => 0.4385
linear_svm_class_train = classification_applier_pool(get_range(0.2, 0.3, 0.01), get_linear_SVM_classifier, X_train, Y_train)
linear_svm_class_metric_valid = prediction_metrics_pool(linear_svm_class_train, X_test, Y_test, 'macro')

linear_svm_class_best_metric = get_best_train_classifier(linear_svm_class_metric_valid, linear_svm_class_train)
#linear_svm_class_metric_test = prediction_metrics((linear_svm_class_best_metric[0][0], linear_svm_class_best_metric[1]), X_test,Y_test,'macro')
linear_svm_class_metric_train = prediction_metrics((linear_svm_class_best_metric[0][0], linear_svm_class_best_metric[1]), X_train, Y_train,'macro')

print('')