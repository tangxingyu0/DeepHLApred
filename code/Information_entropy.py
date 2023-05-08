import numpy as np
import warnings
warnings.filterwarnings("ignore")
import numpy as np


def Calculate_information(seq,list=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']):
    #读一段seq 返回一个数组 记录频数
    """
        Calculate_information('ATCGGGG')->[1, 1, 1, 4]
    """
    labelCounts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #Record Number
    # length = len(seq)
    for i in seq:
        for j in range(20):
            if(i == list[j]):
                labelCounts[j] = labelCounts[j]+1
    return labelCounts


def Calculate_entropy(labelCounts):
    """
        TestCounts=Calculate_information('ATCG')
        Calculate_entropy(Calculate_entropy([2,3,1,0]))
        Output:1.459

    """
    entropy = 0
    sum = np.sum(labelCounts)
    for x in labelCounts:
        x = x / sum
        logx = np.log2(x)
        if (x != 0):
            entropy -= x * logx
    return entropy


def CalculateMatrix_SingleNucleobase(data, order):
    # 目标是 4*166

    matrix = np.zeros((20, 15))
    for j in range(len(data)):
        for i in range(len(data[j])):  # position

            matrix[order[data[j][i]]][i] += 1
    return matrix


# order = {}
# nucleotides = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
#
# for i in range(len(nucleotides)):
#     order[nucleotides[i]] = i


def Calculate_ent_Position(Matrix0):
    # 计算序列位置信息熵
    entropy_list = []
    for i in range(len(Matrix0[0])):
        entropy_list.append(Calculate_entropy(Matrix0[:,i]))
    return entropy_list


def Calculate_ent_Category(Matrix0):
    # 计算每种氨基酸的信息熵
    entropy_list=[]
    for i in range(len(Matrix0)):
        entropy_list.append(Calculate_entropy(Matrix0[i]))
    return entropy_list

def read_data(seq):
    # 读取序列文件
    seq_list = []
    f = open(seq).readlines()
    for i in range(0, len(f)-1, 2):
        sequence = f[i+1].strip('\n')
        seq_list.append(sequence)
    return seq_list
