#! /usr/bin/env python
# -*- coding:utf-8 -*-


from numpy import *
import operator

def createDataSet():
    """"""
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(intX, dataSet, labels, k):
    """k-NN algorithm"""
    #  number of rows of the training set.
    dataSetSize = dataSet.shape[0]
    #  tile() func creates a matrix whose values are inX and
    #  whose size is identical to dataSet.
    #  b = time(a, (m,n)),将a复制n次存入c中，再将c复制m次存入b中，如此构造出b。
    #  Then do the minus calculation.
    diffMat = tile(intX, (dataSetSize,1)) - dataSet
    #  Do the power
    sqDiffMat = diffMat ** 2
    #  Do the sum, on the row direction.
    sqDistances = sqDiffMat.sum(axis=1)
    #  Do the square root.
    distances = sqDistances ** 0.5
    #  Sort and return the index value of the sorted elements.
    #  For example, distances = [1.487, 1.414, 0, 0.1]
    #  sortedDistIndicies := [2, 3, 1, 0] whose elements' values
    #  are indexes of the distances elements.
    #  0   0.1   1.414   1.487
    #  d2  d3    d1      d0
    sortedDistIndicies = distances.argsort()
    #  A dict
    classCount = {}
    #  对k个点的每一个。
    for i in range(k):
        #  取该点label
        #  i=0, sd[0]= 2(见sortedDistIndicies注释)
        #  labels[2] = 'B'
        #  voteIlabel := 'B'
        voteIlabel = labels[sortedDistIndicies[i]]
        #  计数，出现一次该label就增1
        #  计算k个点中，各标签出现次数。
        #  出现次数最多的标签，就是测试点的分类标签。
        #  classCount.get('B', 0), dict.get() 功能如下： classCount中有key'B'，
        #  就取其相应value。如果没有，取默认值，本例为0。 这样完成对标签计数。
        #  dict.get('name', defaultValue)与dict['name']异同处：
        #  "It allows you to provide a default vlaue if the key is missing.
        #  dict.get('name') is same as writing,
        #      dict['name'] or None
        #  so it implicitly handles keyError exception."
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    #  将classCount按第二列(key参数指定)，即value列排序，
    #  大值在前（reverse参数指定）。
    #  最终sortedClassCount形如 [('B', 2), ('A', 1)]， 是一个list。
    #  In python 3, dict.iteritems() is replaced with dict.items.
    #  So classCount.items() is correct.
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    #  sortedClassCount[0], ('B', 2)
    #  sortedClasscount[1], ('A', 1)
    #  而返回值sortedClassCount[0][0], 为'B'，为该测试记录通过knn计算出的标签
    return sortedClassCount[0][0]


def file2matrix(filename):
    """"""
    with open(filename) as fr:
        arrayOLines = fr.readlines()
        #  记录总条数
        numberOfLines = len(arrayOLines)
        #  initialize the return matrix as a zeros matrix.
        returnMat = zeros((numberOfLines, 3))
        #  initialize the label vector as empty list.
        classLabelVector = []
        index = 0

        #  取每一条记录
        for line in arrayOLines:
            #  去除行前后空格
            line = line.strip()
            #  以tab为分隔符，切分列。
            listFromLine = line.split('\t')
            #  0 through 2nd column of the training example are features
            returnMat[index,:] = listFromLine[0:3]
            #  3rd (or -1) column of the training example is
            #  the classification label
            classLabelVector.append(int(listFromLine[-1]))
            index += 1

    return returnMat, classLabelVector

def autoNorm(dataSet):
    """归一化数据，保证特征权值等同。避免值大的features，比值小的features更有
    影响力。
    newValue = (oldValue - min) / (max - min)"""

    #  0 表明按列取最小、最大值。
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #  分母
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    #  分子，取为矩阵，计算按element-wise方式进行。
    normDataSet = dataSet - tile(minVals, (m, 1))
    #  求归一值
    normDataSet = normDataSet / tile(ranges, (m, 1))

    return normDataSet, ranges, minVals

def datingClassTest():
    """测试knn算法"""
    #  Hold ration保留率10%， 数据中90%条记录取为训练集，余10%留为测试集。
    hoRatio = 0.10
    #  自文件读入数据。并将数据列，分为features, label。
    #  最后一列是label，放入dl中， 前面若干列是features，放入ddm中。
    ddm, dl, = file2matrix('datingTestSet2.txt')
    #  将features做归一化处理。
    nm, r , minv = autoNorm(ddm)
    #  训练集矩阵的“行”数。本例1000
    m = nm.shape[0]
    #  测试集数目。本例100
    numTestVecs = int(m * hoRatio)
    #  置错误数初值为0.0
    errorCount = 0.0
    
    #  对每一条测试数据。本例中i取值从0:99
    for i in range(numTestVecs):
        #  比对每一条数据与所有(numTestVecs:m,本例100:1000)训练集记录的knn距离
        #  nm是训练集features, dl是训练集label。 k取3。
        classifierResult = classify0(nm[i,:], nm[numTestVecs:m,:], \
                                     dl[numTestVecs:m], 3)
        #  比较计算的分类标签与真实的分类标签。
        if (classifierResult != dl[i]):
            #  不一致，分类错误，错误计数增1。
            errorCount += 1.0
            #  并格式化输出
            print(" TestExample #%d : the label is misclassified as : [%d], the real answer is: [%d]" \
                  % (i, classifierResult, dl[i]))
        else:
            #  一致，分类正确，格式化输出
            print("Test Example #%d : the classifier came back with: %d, the real answer is: %d" \
              % (i, classifierResult, dl[i]))

    #  总错误计数 / 总测试集记录条数， 即为错误率。
    print("The total error rate is: %f" % (errorCount/float(numTestVecs)))

def main():
    datingClassTest()
    
if __name__ == "__main__":
    main()

#  ============================================================================
#  After reading this knn code, I came up with the idea that maybe something
#  can be done to improve or extend this example.
#  1. The calculation of distances can be refactored. There are multiple
#  definition variants concerning distances between two points.
#  To be specific, any form of Minkowski distances will do the job.
#  2. The error rate can be improved using F-score based on precision and
#  recall.
#  3. autoNorm can be replaced with standard score known as Z-score, due to
#  "Z-score will be our primary method of normalization."
