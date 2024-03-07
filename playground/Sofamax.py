import math
import numpy
def Sofamax(inMatrix):
    m, n = numpy.shape(inMatrix)
    outMatrix = numpy.mat(numpy.zeros(m, n))
    soft_num = 0
    for idx in range(0, n):
        outMatrix[0, idx] = math.exp(inMatrix[0, idx])
        soft_num += outMatrix[0, idx]
    for idx in range(0, n):
        outMatrix[0, idx] = outMatrix[0, idx] / soft_num
    return outMatrix

#
# a = numpy.array([[1, 2, 1, 2, 1, 1, 3], [4, 5, 6, 7, 8, 9, 10, 11]])
# print(Sofamax(a))
