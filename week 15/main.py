import math
from collections import Counter
import numpy as npy


debug = 1


def calE():
    while 1:
        try:
            line = input()
            line = line.split(' ')
            a = int(line[0])
            b = int(line[1])
            s = a + b
            result = - (a / s) * math.log2(a / s) - (b / s) * math.log2(b / s)
            print(result)
        except Exception:
            pass


def read(file_path):
    _data = npy.loadtxt(file_path)
    if debug:
        print(_data)
    return _data


def get_p_matrix_laplace(arr_data):
    # 获得y中分类标签的唯一值
    y_labels = npy.unique(arr_data[:, -1])
    # y_labels = set(arr_data[:,-1])  # 同上，两种写法均可
    lambda1 = 1  # λ=1 拉普拉斯平滑
    k = len(y_labels)  # y分类个数k，用于拉普拉斯平滑

    y_counts = len(arr_data)  # y总数据条数
    y_p = {}  # y中每一个分类的概率，字典初始化为空，y分类数是不定的，按字典存储更方便取值
    for y_label in y_labels:
        y_p[y_label] = (len(arr_data[arr_data[:, -1] == y_label]) + lambda1) / (
                    y_counts + k * lambda1)  # y中每一个分类的概率（其实就是频率）

    yx_cnt = []  # 固定y以后的，x中每一种特征出现的次数，此数据量并不大，y分类数 * x维度列数，按list存储即可
    for y_label in y_p.keys():  # 先固定y，遍历y中每一个分类
        y_label_cnt = len(arr_data[arr_data[:, -1] == y_label])  # 此y分类数据条数,N
        for x_j in range(0, arr_data.shape[1] - 1):  # 在固定x特征列，遍历每列x中的特征
            x_j_count = Counter(
                arr_data[arr_data[:, -1] == y_label][:, x_j])  # 按列统计每种特征出现的次数，因为某一列的特征数是不固定的，所以按dict类型存储
            yx_cnt.append([y_label, y_label_cnt, x_j, dict(x_j_count)])

    yx_p = []  # 将统计次数处理为概率
    for i in range(0, len(yx_cnt)):
        # print(yx_cnt[i])
        # print(yx_cnt[i][3])
        p = {}  # 将每列x特征出现的次数转换为概率
        s = len(yx_cnt[i][3].keys())
        for key in yx_cnt[i][3].keys():
            p[key] = (yx_cnt[i][3][key] + lambda1) / (yx_cnt[i][1] + s * lambda1)
        yx_p.append([yx_cnt[i][0], yx_cnt[i][1], yx_cnt[i][2], p])
    return y_p, yx_p


if __name__ == '__main__':
    # calE()
    data = read('./practice2_input.txt')
    p1, p2 = get_p_matrix_laplace(data)
    if debug:
        print(p1)
        for row in p2:
            print(row)
    pass