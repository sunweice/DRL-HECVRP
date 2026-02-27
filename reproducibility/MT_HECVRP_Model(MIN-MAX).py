import time

from gurobipy import *
import numpy as np
import json
from collections import defaultdict
def solution_to_chain(solution,cs_mapping,depot=0):
    """
    solution:[[(0, 3), (0, 8), (0, 9), (1, 5), (2, 0), (3, 7), (4, 10), (5, 0), (6, 0), (7, 6), (8, 12), (9, 1), (10, 13), (12, 4), (13, 2)]]
    """
    if not solution:
        return []
    # 构建邻接表
    graph = defaultdict(list)
    for u, v in solution:
        graph[u].append(v)

    # 从每一个 depot 出发，寻找路径直到返回 depot
    start=graph[depot][0]
    graph[depot].remove(start)

    path = [depot, start]
    if start in cs_mapping:
        path[1]=cs_mapping[start]
    current = start
    while current in graph:
        # 找第一个未访问的后继
        if current == depot:
            if graph[current]:
                next_nodes = graph[current][0]
            else:
                break
            graph[current].remove(next_nodes)
            current = next_nodes
        else:
            next_nodes = graph[current][0]
            current = next_nodes

        if next_nodes in cs_mapping:
            next_nodes=cs_mapping[next_nodes]
        path.append(next_nodes)
    return path
def Reconstruct_solution(model,cs_mapping,Q,n_total):
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        print("\nOptimal Solution Found!\n")
        #线路变量
        tours=[]
        for v in range(len(Q)):
            tour=[]
            for i in range(n_total):
                for j in range(n_total):
                    var_name = f"x[{v},{i},{j}]"
                    var = model.getVarByName(var_name)
                    if var is not None and var.X > 0.5:
                        tour.append((i,j))  #i0->j0
            #solution_to_chain
            tour=solution_to_chain(tour,cs_mapping)
            tours.append(tour)
        #容量变量
        Qs=[]
        for v in range(len(Q)):
            q=[]
            for i in range(n_total):
                    i0=cs_mapping[i] if i in cs_mapping else i
                    var_name = f"q[{v},{i}]"
                    var = model.getVarByName(var_name)
                    q.append((i0,var.X))
            Qs.append(q)
        #电量变量
        Bs=[]
        for v in range(len(Q)):
            b=[]
            for i in range(n_total):
                    i0=cs_mapping[i] if i in cs_mapping else i
                    var_name = f"b[{v},{i}]"
                    var = model.getVarByName(var_name)
                    b.append((i0,var.X))
            Bs.append(b)
        return tours,Qs,Bs
    else:
        return None

def MT_HECVRP(Matrix,Q,B,S,A,D,timelimit,n_charger,n_customer):
    """
    Q:容量;B:电量;S:速度;A:每公里耗电量
    Matrix:[depot;n_customer;n_charger]
    D:需求,[depot;n_customer]
    """
    #定义模型
    model = Model("MT_HECVRP")
    # model.setParam('OutputFlag', 0)
    start_time = time.time()
    TIME_LIMIT = timelimit  # 10分钟


    #定义集合
    n_veh=len(Q)
    V = range(n_veh)  # 车辆集合 V
    C = range(1, n_customer+1)  # 客户点集合 C: 节点1~5
    R=range(n_customer+ 1, n_customer+n_charger+1)
    R_primed = [] # 充电站副本节点 R'
    cs_mapping = {}  # 副本编号 → 原始充电站编号映射
    next_id = n_customer+ 1  # 从此编号起分配副本编号
    replication=n_customer*2
    for r in R:
        for k in range(replication):
            R_primed.append(next_id)
            cs_mapping[next_id] = r  # 记录副本对应的原始充电站
            next_id += 1
    N = list(C) + list(R_primed)  # 中间节点集合 N，不含仓库
    depot = 0
    nodes = [depot] + N
    n_total=len(nodes)

    def get_distance(i, j):
        # 如果 i 是副本，就映射到原始
        i0 = cs_mapping[i] if i in cs_mapping else i
        j0 = cs_mapping[j] if j in cs_mapping else j
        return Matrix[i0][j0]

    #定义变量
    xs=[]
    for v in V:
        for i in nodes:
            for j in nodes:
                i0 = cs_mapping[i] if i in cs_mapping else i
                j0 = cs_mapping[j] if j in cs_mapping else j
                if i0!=j0:  #剔除充电站到充电站
                    xs.append((v,i,j))
    x = model.addVars(xs,vtype=GRB.BINARY, name="x")
    q = model.addVars(V, nodes, lb=0, name="q")  # 剩余载重
    b = model.addVars(V, nodes, lb=0, name="b")  # 剩余电量
    z = model.addVar(name="max_travel_time", vtype=GRB.CONTINUOUS) #辅助变量
    #定义约束

    #目标函数 (2)
    model.setObjective(z, GRB.MINIMIZE)
    #为每辆车添加约束
    for v in V:  # V 为车辆集合
        travel_time_v = quicksum((get_distance(i, j) / S[v]) * x[v,i,j] for i in nodes for j in nodes if (v,i,j) in x)
        model.addConstr(travel_time_v <= z, name=f"max_time_constr_v{v}")

    # 约束 (3): 每个客户点只能被访问一次
    for i in C:
        model.addConstr(quicksum(x[v, i, j] for v in V for j in nodes if (v,i,j) in xs) == 1, name=f"visit_once_{i}")

    # 约束 (4): 每个充电站副本最多访问一次
    for i in R_primed:
        model.addConstr(quicksum(x[v, i, j] for v in V for j in nodes if (v,i,j) in xs) <= 1, name=f"repl_limit_{i}")

    # 约束 (5): 流量守恒约束，每个节点进等于出
    for v in V:
        for i in nodes:
            model.addConstr(
                quicksum(x[v, i, j] for j in nodes if (v,i,j) in xs) -
                quicksum(x[v, j, i] for j in nodes if (v,j,i) in xs) == 0,
                name=f"flow_balance_{i}"
            )

    # 约束 (6): 剩余载重转移(不含depot)
    for v in V:
        for i in nodes:
            for j in N:
                if (v,i,j) in xs:
                    delta_q = D[i] if i in C else 0  # 只在客户点交付货物
                    model.addConstr(
                        q[v, j] <= q[v, i] - delta_q * x[v, i, j] + Q[v] * (1 - x[v, i, j]),
                        name=f"load_transfer_{v}_{i}_{j}"
                    )
    for v in V:
        for i in nodes:
            if i!=0:
                delta_q = D[i] if i in C else 0  # 只在客户点交付货物
                model.addConstr(
                    0 <= q[v, i] - delta_q * x[v, i, 0] + Q[v] * (1 - x[v, i, 0]),
                    name=f"load_transfer_{v}_{i}_{0}"
                )
    # 约束 (7): 初始载重限制
    for v in V:
        model.addConstr(q[v, depot] <= Q[v], name=f"init_q_ub_{v}")

    # 约束 (8): 电量转移（考虑单位距离耗电）
    for v in V:
        for i in C:
            for j in nodes:
                if i != j:
                    a_ij = A[v] * get_distance(i, j)
                    model.addConstr(
                        b[v, j] <= b[v, i] - a_ij * x[v, i, j] + B[v] * (1 - x[v, i, j]),
                        name=f"battery_transfer_{v}_{i}_{j}"
                    )

    # 约束 (9): 初始电量限制
    for v in V:
        for i in [depot]+R_primed:
            for j in nodes:
                if (v,i,j) in xs:
                    a_ij = A[v] * get_distance(i, j)
                    model.addConstr(b[v, j] <= B[v]-a_ij*x[v,i,j], name=f"init_b_ub_{v}")

    model.update()
    # 设置求解时间限制，减去建模已经花费的时间
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time <= 0:
        return None,None,None
    model.setParam('TimeLimit', remaining_time)
    model.optimize()

    end_time = time.time()
    t=end_time-start_time
    tours,Qs,Bs=Reconstruct_solution(model,cs_mapping,Q,n_total)

    # 输出结果

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        return model.ObjVal,tours,t
    else:
        return 0



if __name__ == '__main__':
    import random
    import os
    np.random.seed(42)
    random.seed(42)
    path='instance/V5/C50_RC6_V5/'
    obj='min-max'
    timelimit=60
    results = {
        'name': [],
        'cost': [],
        'time': [],
        'tours':[],
    }
    count=0
    for p in os.listdir(path):
        # count+=1
        # if count<=6:
        #     continue
        filepath=path+p
        results['name'].append(p)

        with open(filepath,'r') as f:
            instance=json.load(f)

            n_customer =len(instance['Customer'])
            n_charger = len(instance['Charger'])
            n_veh=len(instance['Capacity'])
            n_node = 1 + n_customer + n_charger  # depot + customers + chargers

            coords = np.concatenate((np.array(instance['Depot'])[None,:],np.array(instance['Customer']),np.array(instance['Charger'])),0)  # 坐标范围 [0,100]
            Matrix = np.zeros((n_node, n_node))

            # 欧几里得距离 + 三角不等式对称
            for i in range(n_node):
                for j in range(n_node):
                    if i != j:
                        dist = np.linalg.norm(coords[i] - coords[j])
                        Matrix[i, j] = dist
                    else:
                        Matrix[i, j] = 0.0

            # 车辆参数（4辆异质车）
            Q = np.array(instance['Capacity'])          # 载重上限
            B = np.array(instance['Battery'])         # 电量上限
            S = np.array(instance['Speed'])
            if n_veh==3:
                A = [1, 1, 1]  # 耗电量 kWh/km
            else:
                A = [1, 1, 1,1,1]  # 耗电量 kWh/km


            # 客户需求（depot为0）
            D = [0] + instance['Demand']  # depot + customers

            # 调用模型

            ObjVal,tours,t = MT_HECVRP(Matrix, Q, B, S, A, D, timelimit,n_charger=n_charger, n_customer=n_customer)
            if ObjVal==None:
                results['time'].append(0)
                results['cost'].append(0)
                results['tours'].append([0])
            else:
                results['time'].append(t)
                results['cost'].append(ObjVal)
                results['tours'].append(tours)
    savepath='./results/minmax/V{}/C{}_RC{}_V{}/'.format(n_veh,n_customer,n_charger,n_veh)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    import pandas as pd

    pd.DataFrame(results).to_csv(savepath+f'{obj}_{timelimit}.csv'.format(obj,n_veh,n_customer),index=False)
