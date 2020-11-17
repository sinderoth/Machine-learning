import random

random.seed(1234)  # fix randomness

import math

train=[]
test=[]

file = open("task1_train.txt","r")
data= file.readlines()
for line in data:
    item = line.split(",")
    
    train.append(item)
file.close()

file2 = open("task1_test.txt","r")
data2= file2.readlines()
for line in data2:
    item = line.split(",")
    test.append(item)
file2.close()

def calculateDistance(l1,l2):
    result = 0
    for i in range(len(l1)-1):
        result += math.pow(float(l1[i])-float(l2[i]),2)
    return math.sqrt(result)


def accuracy(l,test):
    length = len(l)
    result=0
    for i in range(length):
        if l[i]==test[i][-1]:
            result+=1
    return (result/length)*100
    


def kNN(k, train_set, test_set):
    """
    the unweighted k-NN algorithm using Euclidean distance as the metric

    :param k: the k value, i.e, how many neighbors to consider
    :param train_set: training set, a list of lists where each nested list is a training instance
    :param test_set: test set, a list of lists where each nested list is a test instance
    :return: percent accuracy for the test set, e.g., 78.42
    """
    
    labels = []
    for index2, x_i in enumerate(test_set):
        closest_k = []
        for index, example in enumerate(train_set):
            distance = calculateDistance(example,x_i)
            closest_k.append((distance,index))
        closest_k.sort()
            

        
        neigh = []
        for i in range(k):
            neigh.append(train_set[closest_k[i][1]][-1])

        

        popularvote = max(neigh,key=neigh.count)
        labels.append(popularvote)
       

    return accuracy(labels,test_set)


        


def find_best_k(train_set, test_set, num_folds):
    """
    finds the best k value by using K-fold cross validation. Try at least 10 different k values. Possible choices
    can be: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19. Besides the return value, as a side effect, print each k value and
    the corresponding validation accuracy to the screen as a tuple. As an example,
    (1, 78.65)
    (3, 79.12)
    ...
    (19, 76.99)

    :param train_set: training set, a list of lists where each nested list is a training instance
    :param test_set: test set, a list of lists where each nested list is a test instance
    :param num_folds: the K value in K-fold cross validation
    :return: a tuple, best k value and percent accuracy for the test set using the best k value, e.g., (3, 80.06)
    """

    random.shuffle(train_set)
    length = len(train_set)
    gap = length//num_folds
    best_k = []
    key=1
    while key<20:
        average = 0
        for i in range(num_folds):
            single = train_set[i*gap:gap*(i+1)]
            others = train_set[:i*gap]+train_set[(i+1)*gap:]
            average += kNN(key,others,single)
        average /= num_folds
        print((key,average))
        best_k.append((average,key))
        key+=2

    result = max(best_k)
    print ("best")
    print ((result[1],result[0]))
    return (result[1],result[0])

find_best_k(train,test,5)