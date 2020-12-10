import pandas as pd
import numpy as np
from pulp import *
import re
import copy


def brgd(n):
    '''
    递归生成n位的二进制反格雷码
    :param n:
    :return:
    '''
    if n==1:
        return ["0","1"]
    L1 = brgd(n-1)
    L2 = copy.deepcopy(L1)
    L2.reverse()
    L1 = ["0" + l for l in L1]
    L2 = ["1" + l for l in L2]
    L = L1 + L2
    return L


def tsp(cost, basic_info):
    # cost: 距离矩阵
    # basic_info:基本信息

    table = cost
    duration = list(basic_info['duration'])
    entertain = list(basic_info['entertain'])

    # 实例化问题
    row = table.shape[0]
    col = table.shape[1]
    prob = LpProblem('Transportation_Problem', sense=LpMinimize)

    # 路线决策变量，0-1
    var = [[LpVariable(f'x{i}_{j}', lowBound=0, upBound=1, cat=LpInteger) for j in range(col)] for i in range(row)]
    # 节点顺序变量
    ui_tag = 'u_%s'
    ui = np.array([LpVariable(ui_tag % (i), lowBound=0, cat='Integer') for i in range(col)])

    # 目标函数
    t = np.array(table)
    prob += lpSum(var[i][j] * (t[i][j] + duration[j]) for i in range(row) for j in range(col) if i != j)

    # 约束条件
    # 遍历每一个点
    prob += lpSum([var[i][j] for i in range(row) for j in range(col)]) == col
    # 不允许两点间折返
    for i in range(row):
        for j in range(col):
            prob += var[i][j] + var[j][i] <= 1
    # 起点约束：以最后一个点为起点
    prob += lpSum([var[col - 1][j] for j in range(row)]) == 1
    # 终点约束：以最后一个点为起终点
    prob += lpSum([var[j][col - 1] for j in range(row)]) == 1
    # 出度约束，每点出度不大于2
    for i in range(row):
        prob += lpSum([var[i][j] for j in range(col)]) <= 1
    # 入度约束，每点入度不大于2
    for j in range(row):
        prob += lpSum([var[i][j] for i in range(row)]) <= 1
    # 防止生成子回路
    for i in range(1, col):
        for j in range(1, col):
            if i != j:
                prob += (
                        ui[i] - ui[j] + col * var[i][j] <= col - 1
                )

    # cluster1子圈
    # prob += lpSum(var[i][j] for i in range(col) for j in negitave) == 0
    # prob += lpSum(var[i][j] for i in negitave for j in range(col)) == 0
    prob.solve()
    result = value(prob.objective)
    path = pd.DataFrame([[value(var[i][j]) for j in range(col)] for i in range(row)])
    print('最短时间: ', result)

    # 生成顺序路线
    display = False
    cycle_ods = {}
    for varb in prob.variables():
        if varb.name.startswith(ui_tag[0]):
            continue
        if varb.varValue > 0:
            if display:
                print("%s: %s" % (varb.name, varb.varValue))
            od = varb.name.split("_")
            o, d = od[0], od[1]
            cycle_ods[int(re.findall('\d+', o)[0])] = int(d)
    if display:
        print("Status: %s" % LpStatus[prob.status])
    print(cycle_ods)

    tour_pairs = []
    for origin in range(col):
        tour_pairs.append([])
        if origin == 0:
            next_origin = cycle_ods[origin]
            tour_pairs[origin].append(origin)
            tour_pairs[origin].append(next_origin)
            continue
        tour_pairs[origin].append(next_origin)
        next_origin = cycle_ods[next_origin]
        tour_pairs[origin].append(next_origin)
    tour_pairs = {idx: tp for idx, tp in enumerate(tour_pairs)}
    print(tour_pairs)

    for pairs in range(len(tour_pairs)):
        print(basic_info['name'][tour_pairs[pairs][0]], '->', basic_info['name'][tour_pairs[pairs][1]])
    return result


if __name__ == '__main__':
    df = pd.read_csv('spotsF.csv')
    table = pd.read_csv('mix_time.csv')

    for index, item in enumerate(list(df.name)):
        print('景点代码:', index, '景点名称', item)
    print('\n------------------------------------')
    cluster4 = []
    a = input('请输入出发点, -1表示输入完成。')
    while a != '-1':
        if int(a) in list(range(len(table))):
            cluster4.append(int(a))
        else:
            print('错误代码，景点代码从0~' ,len(table)-1)
        a = input('请输入景点代码, -1表示输入完成(至少输入3个景点)。')
    print('输入的景点代码为:', cluster4)
    # cluster4 = [12, 13, 14, 15]
    t4 = table.iloc[cluster4, cluster4].copy()
    df_t4 = df.iloc[cluster4].copy()
    df_t4.reset_index(inplace=True)
    time_cost = tsp(t4, df_t4)
    print('最短时间为', time_cost, '分钟，共', time_cost/60, '小时')