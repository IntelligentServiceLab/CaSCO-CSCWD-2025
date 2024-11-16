import math
import random
import numpy as np
import copy
import pandas as pd

"""
    PSO for Industrial Internet
"""
class PSO():
    """粒子群算法"""
    def __init__(self, abstract_service, task_number, population_size, iteration_number, bounds):

        self.w = 0.9 # w为惯性因子
        self.c1 = 1.6
        self.c2 = 1.6  # c1, c2为学习因子，一般取2
        self.bounds = bounds  # 位置的边界
        self.abstract_service = abstract_service  # 抽象服务组合链
        self.task_number = task_number  # 任务数
        self.population_size = population_size  # 种群规模(粒子数量)
        self.iteration_number = iteration_number  # 迭代次数

    def initialization(self):  # (已测试）
        """初始化阶段:根据种群规模，生成相应个数的个体（服务组合解）;
        通过为每个任务随机选择对应要求的资源来初始化一个组合服务"""
        population_x = []  # 种群
        population_num_x = []
        for i in range(0, self.population_size):
            temp = []
            num = []
            if_services = []
            for j in range(0, self.task_number):
                else_services_j = []
                if j == 0:
                    for k in range(0, 3):
                        high = self.abstract_service.candidate_service_number
                        r = random.randint(0, high - 1)
                        time_candidate = self.abstract_service.Time_candidates[r][k]  # 从第K个资源的时间候选服务集选择
                        cost_candidate = self.abstract_service.Cost_candidates[r][k]  # 添加对应的成本属性
                        # temp.extend([time_candidate, cost_candidate])
                        # service_combination = [time_candidate, cost_candidate]
                        if_services.extend([time_candidate, cost_candidate])
                        num.append(r)
                    temp.append(if_services)
                else:
                    for k in range(2 * j + 1, 2 * j + 3):
                        high = self.abstract_service.candidate_service_number
                        r = random.randint(0, high - 1)
                        # print(r)
                        time_candidate = self.abstract_service.Time_candidates[r][k]  # 从第K个资源的时间候选服务集选择
                        cost_candidate = self.abstract_service.Cost_candidates[r][k]  # 添加对应的成本属性
                        # temp.extend([time_candidate, cost_candidate])
                        service_combination = [time_candidate, cost_candidate]
                        else_services_j.extend(service_combination)
                        num.append(r)
                    temp.append(else_services_j)
            population_num_x.append(num)
            population_x.append(temp)
        return population_x, population_num_x

    def initialization_V(self, Vmin, Vmax):
        """
            初始化解的 速度
        """
        population_V = []  # 速度
        for i in range(0, self.population_size):
            i_task = []
            for j in range(0, self.task_number):
                if j==0:
                    temp = [0, 0, 0, 0, 0]
                    temp[0] = random.uniform(Vmin[j][0], Vmax[j][0])
                    temp[1] = random.uniform(Vmin[j][1], Vmax[j][1])
                    temp[2] = random.uniform(Vmin[j][2], Vmax[j][2])
                    temp[3] = random.uniform(Vmin[j][3], Vmax[j][3])
                    temp[4] = random.uniform(Vmin[j][4], Vmax[j][4])
                    i_task.append(temp)
                else:
                    temp = [0, 0, 0, 0]
                    temp[0] = random.uniform(Vmin[j][0], Vmax[j][0])
                    temp[1] = random.uniform(Vmin[j][1], Vmax[j][1])
                    temp[2] = random.uniform(Vmin[j][2], Vmax[j][2])
                    temp[3] = random.uniform(Vmin[j][3], Vmax[j][3])
                    i_task.append(temp)
            population_V.append(i_task)
        return population_V

    def get_Vmax(self):
        """获取速度的上下界"""
        Vmax = []  # 每个任务的速度上界
        # 速度的上界
        for i in range(self.task_number):
            if i==0:
                temp = [0, 0, 0, 0, 0]
                temp[0] = 0.3 * (self.bounds[0][1][1])
                temp[1] = 0.3 * (self.bounds[0][0][1])
                temp[2] = 0.3 * (self.bounds[0][1][1])
                temp[3] = 0.3 * (self.bounds[0][0][1])
                temp[4] = 0.3* (self.bounds[0][1][1])
                Vmax.append(temp)
            else:
                temp = [0, 0, 0, 0]
                temp[0] = 0.3 * (self.bounds[0][0][1])
                temp[1] = 0.3 * (self.bounds[0][1][1])
                temp[2] = 0.3 * (self.bounds[0][0][1])
                temp[3] = 0.3 * (self.bounds[0][1][1])
                Vmax.append(temp)

        return Vmax

    def get_Vmin(self):
        """获取速度的上下界"""
        Vmin = []  # 每个任务的速度下界
        for i in range(self.task_number):
            if i == 0:
                temp = [0, 0, 0, 0, 0]
                temp[0] = (-0.3) * (self.bounds[0][1][1])
                temp[1] = (-0.3)* (self.bounds[0][0][1])
                temp[2] = (-0.3) * (self.bounds[0][1][1])
                temp[3] = (-0.3) * (self.bounds[0][0][1])
                temp[4] = (-0.3)* (self.bounds[0][1][1])
                Vmin.append(temp)
            else:
                temp = [0, 0, 0, 0]
                temp[0] = (-0.3) * (self.bounds[0][0][1])
                temp[1] = (-0.3) * (self.bounds[0][1][1])
                temp[2] = (-0.3)* (self.bounds[0][0][1])
                temp[3] = (-0.3) * (self.bounds[0][1][1])
                Vmin.append(temp)

        return Vmin

    def update_X(self, pop_X, pop_V):
        """更新位置"""
        new_pop_X = []  # 种群更新后的位置
        for i in range(0, self.population_size):
            temp = []
            for j in range(0, self.task_number):
                if j == 0:
                    new_X = [0,0, 0, 0, 0, 0]
                    new_X[1] = pop_X[i][j][1] + pop_V[i][j][0]
                    new_X[2] = pop_X[i][j][2] + pop_V[i][j][1]
                    new_X[3] = pop_X[i][j][3] + pop_V[i][j][2]
                    new_X[4] = pop_X[i][j][4] + pop_V[i][j][3]
                    new_X[5] = pop_X[i][j][5] + pop_V[i][j][4]

                    # 判断是否越上界
                    if new_X[1] > self.bounds[0][1][1]:
                        new_X[1] = self.bounds[0][1][1]
                    if new_X[2] > self.bounds[0][0][1]:
                        new_X[2] = self.bounds[0][0][1]
                    if new_X[3] > self.bounds[0][1][1]:
                        new_X[3] = self.bounds[0][1][1]
                    if new_X[4] > self.bounds[0][0][1]:
                        new_X[4] = self.bounds[0][0][1]
                    if new_X[5] > self.bounds[0][1][1]:
                        new_X[5] = self.bounds[0][1][1]

                    # 判断是否越下界
                    if new_X[1] < self.bounds[0][1][0]:
                        new_X[1] = self.bounds[0][1][0]
                    if new_X[2] < self.bounds[0][0][0]:
                        new_X[2] = self.bounds[0][0][0]
                    if new_X[3] < self.bounds[0][1][0]:
                        new_X[3] = self.bounds[0][1][0]
                    if new_X[4] < self.bounds[0][0][0]:
                        new_X[4] = self.bounds[0][0][0]
                    if new_X[5] < self.bounds[0][1][0]:
                        new_X[5] = self.bounds[0][1][0]
                    temp.append(new_X)
                else:
                    new_X = [0, 0, 0, 0]
                    new_X[0] = pop_X[i][j][0] + pop_V[i][j][0]
                    new_X[1] = pop_X[i][j][1] + pop_V[i][j][1]
                    new_X[2] = pop_X[i][j][2] + pop_V[i][j][2]
                    new_X[3] = pop_X[i][j][3] + pop_V[i][j][3]

                    # 判断是否越上界
                    if new_X[0] > self.bounds[0][0][1]:
                        new_X[0] = self.bounds[0][0][1]
                    if new_X[1] > self.bounds[0][1][1]:
                        new_X[1] = self.bounds[0][1][1]
                    if new_X[2] > self.bounds[0][0][1]:
                        new_X[2] = self.bounds[0][0][1]
                    if new_X[3] > self.bounds[0][1][1]:
                        new_X[3] = self.bounds[0][1][1]

                    # 判断是否越下界
                    if new_X[0] < self.bounds[0][0][0]:
                        new_X[0] = self.bounds[0][0][0]
                    if new_X[1] < self.bounds[0][1][0]:
                        new_X[1] = self.bounds[0][1][0]
                    if new_X[2] < self.bounds[0][0][0]:
                        new_X[2] = self.bounds[0][0][0]
                    if new_X[3] < self.bounds[0][1][0]:
                        new_X[3] = self.bounds[0][1][0]
                    temp.append(new_X)
            new_pop_X.append(temp)
        return new_pop_X

    def update_V(self, pop_X, pop_V, pbest, gbest, Vmin, Vmax):
        """更新速度"""
        new_pop_V = []  # 种群更新后的速度
        for i in range(0, self.population_size):
            temp = []
            for j in range(0, self.task_number):
                if j==0:
                    speed=[0,0,0,0,0]
                    r1 = random.random()
                    r2 = random.random()
                    speed[0] = self.w * pop_V[i][j][0] + self.c1 * r1 * (
                                pbest[i][j][0] - pop_X[i][j][0]) + self.c2 * r2 * (
                                       gbest[j][0] - pop_X[i][j][0])
                    speed[1] = self.w * pop_V[i][j][1] + self.c1 * r1 * (
                                pbest[i][j][1] - pop_X[i][j][1]) + self.c2 * r2 * (
                                       gbest[j][1] - pop_X[i][j][1])
                    speed[2] = self.w * pop_V[i][j][2] + self.c1 * r1 * (
                                pbest[i][j][2] - pop_X[i][j][2]) + self.c2 * r2 * (
                                       gbest[j][2] - pop_X[i][j][2])
                    speed[3] = self.w * pop_V[i][j][3] + self.c1 * r1 * (
                                pbest[i][j][3] - pop_X[i][j][3]) + self.c2 * r2 * (
                                       gbest[j][3] - pop_X[i][j][3])
                    speed[4]= self.w * pop_V[i][j][4] + self.c1 * r1 * (
                                pbest[i][j][4] - pop_X[i][j][4]) + self.c2 * r2 * (
                                       gbest[j][4] - pop_X[i][j][4])
                    # 判断是否越上界
                    if speed[0] > Vmax[j][0]:
                        speed[0] = Vmax[j][0]

                    if speed[1] > Vmax[j][1]:
                        speed[1] = Vmax[j][1]

                    if speed[2] > Vmax[j][2]:
                        speed[2] = Vmax[j][2]

                    if speed[3] > Vmax[j][3]:
                        speed[3] = Vmax[j][3]

                    if speed[4] > Vmax[j][4]:
                        speed[4] = Vmax[j][4]

                    # 判断是否越下界
                    if speed[0] < Vmin[j][0]:
                        speed[0] = Vmin[j][0]
                    if speed[1] < Vmin[j][1]:
                        speed[1] = Vmin[j][1]
                    if speed[2] < Vmin[j][2]:
                        speed[2] = Vmin[j][2]
                    if speed[3] < Vmin[j][3]:
                        speed[3] = Vmin[j][3]
                    if speed[4] < Vmin[j][4]:
                        speed[4] = Vmin[j][4]
                    temp.append(speed)
                else:
                    speed = [0, 0, 0, 0]
                    r1 = random.random()
                    r2 = random.random()
                    speed[0] = self.w * pop_V[i][j][0] + self.c1 * r1 * (
                            pbest[i][j][0] - pop_X[i][j][0]) + self.c2 * r2 * (
                                       gbest[j][0] - pop_X[i][j][0])
                    speed[1] = self.w * pop_V[i][j][1] + self.c1 * r1 * (
                            pbest[i][j][1] - pop_X[i][j][1]) + self.c2 * r2 * (
                                       gbest[j][1] - pop_X[i][j][1])
                    speed[2] = self.w * pop_V[i][j][2] + self.c1 * r1 * (
                            pbest[i][j][2] - pop_X[i][j][2]) + self.c2 * r2 * (
                                       gbest[j][2] - pop_X[i][j][2])
                    speed[3] = self.w * pop_V[i][j][3] + self.c1 * r1 * (
                            pbest[i][j][3] - pop_X[i][j][3]) + self.c2 * r2 * (
                                       gbest[j][3] - pop_X[i][j][3])

                    # 判断是否越上界
                    if speed[0] > Vmax[j][0]:
                        speed[0] = Vmax[j][0]

                    if speed[1] > Vmax[j][1]:
                        speed[1] = Vmax[j][1]

                    if speed[2] > Vmax[j][2]:
                        speed[2] = Vmax[j][2]

                    if speed[3] > Vmax[j][3]:
                        speed[3] = Vmax[j][3]

                    # 判断是否越下界
                    if speed[0] < Vmin[j][0]:
                        speed[0] = Vmin[j][0]
                    if speed[1] < Vmin[j][1]:
                        speed[1] = Vmin[j][1]
                    if speed[2] < Vmin[j][2]:
                        speed[2] = Vmin[j][2]
                    if speed[3] < Vmin[j][3]:
                        speed[3] = Vmin[j][3]
                    temp.append(speed)
            new_pop_V.append(temp)
        return new_pop_V

    def save_pbest(self, pbest,pbest_num, pop_X,pop_num):
        """更新个体历史最优"""
        updated_pbest = []
        updated_pbest_num = []
        for i in range(self.population_size):
            # 如果新解支配旧解
            if (self.TimeFitness_1(pop_X[i], pop_num[i]) + self.CostFitness_1(pop_X[i], pop_num[i])) < (self.TimeFitness_1(pbest[i], pbest_num[i]) + self.CostFitness_1(pbest[i], pbest_num[i])):
                updated_pbest.append(pop_X[i])
                updated_pbest_num.append(pop_num[i])
            else:
                updated_pbest.append(pbest[i])
                updated_pbest_num.append(pbest_num[i])
        return updated_pbest,updated_pbest_num

    def save_gbest(self, gbest,gbest_num, Pop_X,Pop_num):
        """更新种群历史最优"""
        [Pareto,Pareto_num] = copy.deepcopy(self.ParetoSearch(Pop_X,Pop_num))

        for i in range(0, len(Pareto)):
            # if ((self.TimeFitness(Pareto[i],Pareto_num[i]) < self.TimeFitness(gbest,gbest_num)) and (
            #         self.CostFitness(Pareto[i],Pareto_num[i]) < self.CostFitness(gbest,gbest_num))) or (
            #         (round(self.TimeFitness(Pareto[i],Pareto_num[i])) == round(self.TimeFitness(gbest,gbest_num))) and (
            #         self.CostFitness(Pareto[i],Pareto_num[i]) < self.CostFitness(gbest,gbest_num))) or (
            #         (self.TimeFitness(Pareto[i],Pareto_num[i]) < self.TimeFitness(gbest,gbest_num)) and (
            #         round(self.CostFitness(Pareto[i],Pareto_num[i])) == round(self.CostFitness(gbest,gbest_num)))):
            if (self.TimeFitness_1(Pareto[i], Pareto_num[i]) + self.CostFitness_1(Pareto[i], Pareto_num[i])) < (self.TimeFitness_1(gbest, gbest_num) + self.CostFitness_1(gbest, gbest_num)):
                gbest = copy.deepcopy(Pareto[i])
                gbest_num = copy.deepcopy(Pareto_num[i])

        return gbest,gbest_num

    def ParetoSearch(self, population, population_num):  # (已测试）
        """
            Pareto前沿面搜索
            功能：找出种群中的非支配解集
            参数：种群
            返回值：Pareto解集
        """
        # Pareto非支配解集
        ParetoSet = []
        ParetoSet1 = []

        # 种群的适应值列表
        Fitness_List = []
        # 计算出所有解的两个适应值
        for i in range(0, self.population_size):
            temp = []
            # 添加种群中第i个个体的时间适应值
            TimeFit = copy.deepcopy(self.TimeFitness_1(population[i], population_num[i]))
            # 添加种群中第i个个体的成本适应值
            CostFit = copy.deepcopy(self.CostFitness_1(population[i], population_num[i]))
            temp.append(TimeFit)
            temp.append(CostFit)
            Fitness_List.append(temp)

        # 将适应值列表的第三位视作判断是否为pareto解的依据
        for i in range(0, self.population_size):
            Fitness_List[i].append(0)

        # 寻找Pareto解集
        for i in range(0, self.population_size):
            for j in range(0, self.population_size):
                if i != j:
                    if (Fitness_List[i][0] > Fitness_List[j][0] and Fitness_List[i][1] > Fitness_List[j][1]) \
                            or (Fitness_List[i][0] > Fitness_List[j][0] and round(Fitness_List[i][1]) == round(
                        Fitness_List[j][1])) \
                            or (round(Fitness_List[i][0]) == round(Fitness_List[j][0]) and Fitness_List[i][1] >
                                Fitness_List[j][1]):
                        Fitness_List[i][2] = 1
                    else:
                        Fitness_List[i][2] = 0
                else:
                    continue
            if Fitness_List[i][2] == 0:
                ParetoSet.append(population_num[i])
                ParetoSet1.append(population[i])

        # 若不存在非支配解则选出两个适应值相加最小的作为唯一的非支配解
        if len(ParetoSet) == 0:
            pareto = copy.deepcopy(population_num[0])  # 初始化pareto解为第一个
            pareto1 = copy.deepcopy(population[0])
            pareto_fit = sum(Fitness_List[0])  # 初始化pareto解的适应值

            for i in range(1, self.population_size):
                my_fit = sum(Fitness_List[i])
                if my_fit < pareto_fit:
                    pareto_fit = my_fit
                    pareto = copy.deepcopy(population_num[i])
                    pareto1 = copy.deepcopy(population[i])
            ParetoSet.append(pareto)
            ParetoSet1.append(pareto1)

        # 将适应值列表的第三位移除
        for i in range(0, self.population_size):
            Fitness_List[i].pop(2)

        return ParetoSet1, ParetoSet

    def TimeFitness(self, solution1, solution2):  # (已测试）
        """
            时间消耗适应函数
        """
        Time_fit = 0
        radio = [0.0] * self.task_number
        radio1 = [0.0] * self.task_number
        df_a = pd.read_csv('lab/history_records.csv')
        for i in range(0, self.task_number):
            if i == 0:
                # 假设 solution2 是一个列表，包含三个值
                # 例如: solution2 = [value1, value2, value3]
                # 使用列的索引来访问数据
                count1 = df_a[(df_a.iloc[:, 0] == solution2[0]) & (df_a.iloc[:, 1] == solution2[1])].shape[0]
                count2 = df_a[(df_a.iloc[:, 1] == solution2[1]) & (df_a.iloc[:, 2] == solution2[2])].shape[0]
                count3 = df_a[(df_a.iloc[:, 0] == solution2[0]) & (df_a.iloc[:, 2] == solution2[2])].shape[0]
                radio[i] = (count1 + count2 + count3) / (3 * 400)
                # print(radio[i])

            else:
                count1 = df_a[(df_a[df_a.columns[2 * i + 1]] == solution2[2 * i + 1]) & (
                            df_a[df_a.columns[2 * i + 2]] == solution2[2 * i + 2])].shape[0]
                radio[i] = count1 / 400
                # print(radio[i])
        for i in range(0, self.task_number - 1):
            if i == 0:
                count1 = df_a[(df_a[df_a.columns[0]] == solution2[0]) & (df_a[df_a.columns[3]] == solution2[3])].shape[
                    0]
                count2 = df_a[(df_a[df_a.columns[0]] == solution2[0]) & (df_a[df_a.columns[4]] == solution2[4])].shape[
                    0]
                count3 = df_a[(df_a[df_a.columns[1]] == solution2[1]) & (df_a[df_a.columns[3]] == solution2[3])].shape[
                    0]
                count4 = df_a[(df_a[df_a.columns[1]] == solution2[1]) & (df_a[df_a.columns[4]] == solution2[4])].shape[
                    0]
                count5 = df_a[(df_a[df_a.columns[2]] == solution2[2]) & (df_a[df_a.columns[3]] == solution2[3])].shape[
                    0]
                count6 = df_a[(df_a[df_a.columns[2]] == solution2[2]) & (df_a[df_a.columns[4]] == solution2[4])].shape[
                    0]
                radio1[i] = (count1 + count2 + count3 + count4 + count5 + count6) / (6 * 400)
                # print(radio1[i])
            else:
                count1 = df_a[(df_a[df_a.columns[2 * i + 1]] == solution2[2 * i + 1]) & (
                            df_a[df_a.columns[2 * i + 3]] == solution2[2 * i + 3])].shape[0]
                count2 = df_a[(df_a[df_a.columns[2 * i + 1]] == solution2[2 * i + 1]) & (
                            df_a[df_a.columns[2 * i + 4]] == solution2[2 * i + 4])].shape[0]
                count3 = df_a[(df_a[df_a.columns[2 * i + 2]] == solution2[2 * i + 2]) & (
                            df_a[df_a.columns[2 * i + 3]] == solution2[2 * i + 3])].shape[0]
                count4 = df_a[(df_a[df_a.columns[2 * i + 2]] == solution2[2 * i + 2]) & (
                            df_a[df_a.columns[2 * i + 4]] == solution2[2 * i + 4])].shape[0]
                radio1[i] = (count1 + count2 + count3 + count4) / (4 * 400)
                # print(radio1[i])
        for i in range(0, self.task_number):
            if i == 0:
                time = solution1[i][2] + solution1[i][4]
                Time_fit += time * (1 - radio1[i]) * (1 - radio[i])
            else:
                time = solution1[i][0] + solution1[i][2]
                if i != 4:
                    Time_fit += time * (1 - radio1[i]) * (1 - radio[i])
                else:
                    Time_fit += time * (1 - radio[i])
        return Time_fit

    def CostFitness(self, solution1, solution2):  # (已测试）
        """
            成本消耗适应函数
        """
        df_a = pd.read_csv('lab/Manu_label.csv', header=None)
        Cost_Fit = 0  # 解的总体成本消耗
        discount = {}
        for i in range(1, 11):
            column_index = i
            row_index = solution2[i]
            value = df_a.iloc[row_index, column_index - 1]

            if value in discount:
                discount[value] += 1
            else:
                discount[value] = 1
            # discount[df_a[df_a.columns[i-1]][solution2[i]]]+=1
        Cost_Fit = Cost_Fit + solution1[0][1]
        for i in range(1, 11):
            if i < 3:
                price = solution1[0][2 * i + 1]
                cnt = self.Discount(discount[df_a[df_a.columns[i - 1]][solution2[i]]])
                Cost_Fit += price * cnt
            else:
                p = math.floor((i - 1) / 2)
                price = solution1[p][2 * (i - 2 * p - 1) + 1]
                cnt = self.Discount(discount[df_a[df_a.columns[i - 1]][solution2[i]]])
                Cost_Fit += price * cnt
        return Cost_Fit

    def Discount(self, x):  # （已测试）
        """折扣函数"""
        count = 0.5 + 1 / (1 + np.exp(0.1 * (x - 1)))
        return count

    def refine(self, population, bounds):  # (已测试）
        """refine操作符:防止越界"""
        # 第一步
        # for i in range(0, self.population_size):
        #     for j in range(0, self.task_number):
        #         population[i][j] = round(population[i][j])

        # 第二步
        for i in range(0, self.population_size):
            for j in range(0, self.task_number):
                # 如果超过上界,等于上界
                if j == 0:
                    if population[i][j][1] > bounds[0][1][1]:
                        population[i][j][1] = bounds[0][1][1]

                    if population[i][j][2] > bounds[0][0][1]:
                        population[i][j][2] = bounds[0][0][1]

                    if population[i][j][3] > bounds[0][1][1]:
                        population[i][j][3] = bounds[0][1][1]
                    # 成本上界
                    if population[i][j][4] > bounds[0][0][1]:
                        population[i][j][4] = bounds[0][0][1]

                    if population[i][j][5] > bounds[0][1][1]:
                        population[i][j][5] = bounds[0][1][1]

                        # 如果小于下界，等于下界
                    if population[i][j][1] < bounds[0][1][0]:
                        population[i][j][1] = bounds[0][1][0]

                    if population[i][j][2] < bounds[0][0][0]:
                        population[i][j][2] = bounds[0][0][0]

                    if population[i][j][3] < bounds[0][1][0]:
                        population[i][j][3] = bounds[0][1][0]
                    # 成本上界
                    if population[i][j][4] < bounds[0][0][0]:
                        population[i][j][4] = bounds[0][0][0]

                    if population[i][j][5] < bounds[0][1][0]:
                        population[i][j][5] = bounds[0][1][0]

                else:
                    if population[i][j][0] > bounds[0][0][1]:
                        population[i][j][0] = bounds[0][0][1]

                    if population[i][j][1] > bounds[0][1][1]:
                        population[i][j][1] = bounds[0][1][1]
                    # 成本上界
                    if population[i][j][2] > bounds[0][0][1]:
                        population[i][j][2] = bounds[0][0][1]

                    if population[i][j][3] > bounds[0][1][1]:
                        population[i][j][3] = bounds[0][1][1]

                        # 如果小于下界，等于下界
                    if population[i][j][0] < bounds[0][0][0]:
                        population[i][j][0] = bounds[0][0][0]

                    if population[i][j][1] < bounds[0][1][0]:
                        population[i][j][1] = bounds[0][1][0]
                    # 成本上界
                    if population[i][j][2] < bounds[0][0][0]:
                        population[i][j][2] = bounds[0][0][0]

                    if population[i][j][3] < bounds[0][1][0]:
                        population[i][j][3] = bounds[0][1][0]
        return population

    def update(self, old_group, old_group_num, new_group, new_group_num):  # （已测试）
        """这个函数用来更新种群:若新解支配旧解，则替换;否则保留"""

        updated_group = []
        updated_group_num = []
        for i in range(self.population_size):
            # 如果新解支配旧解
            # if ((self.TimeFitness(new_group[i], new_group_num[i]) < self.TimeFitness(old_group[i],
            #                                                                          old_group_num[i])) and (
            #             self.CostFitness(new_group[i], new_group_num[i]) < self.CostFitness(old_group[i],
            #                                                                                 old_group_num[i]))) or (
            #         (round(self.TimeFitness(new_group[i], new_group_num[i])) == round(
            #             self.TimeFitness(old_group[i], old_group_num[i]))) and (
            #                 self.CostFitness(new_group[i], new_group_num[i]) < self.CostFitness(old_group[i],
            #                                                                                     old_group_num[i]))) or (
            #         (self.TimeFitness(new_group[i], new_group_num[i]) < self.TimeFitness(old_group[i],
            #                                                                              old_group_num[i])) and (
            #                 round(self.CostFitness(new_group[i], new_group_num[i])) == round(
            #             self.CostFitness(old_group[i], old_group_num[i])))):
            if (self.TimeFitness_1(new_group[i], new_group_num[i]) + self.CostFitness_1(new_group[i], new_group_num[i])) < (
                    self.TimeFitness_1(old_group[i], old_group_num[i]) + self.CostFitness_1(old_group[i],old_group_num[i])):
                updated_group.append(new_group[i])
                updated_group_num.append(new_group_num[i])
            else:
                updated_group.append(old_group[i])
                updated_group_num.append(old_group_num[i])

        return updated_group, updated_group_num

    def FuzzingMatch(self, population):  # (已测试)
        """
            用最小欧氏距离，在候选服务集中寻找种群中与每个个体的单个服务值最接近的真实值
            参数：种群
        """
        new_population = []  # 初始化新种群
        new_population_num = []
        # 对于种群中的每个个体
        for i in range(0, self.population_size):
            temp_list = []
            num = []
            # 对于每个个体的每个任务
            for j in range(0, self.task_number):

                map_service = []  # 初始化匹配的服务

                # 对于每个任务的候选服务集
                difference = 1000000
                E_distance = 0  # 初始化欧氏距离列表
                # 第j个任务的候选服务集
                Time_candidates = copy.deepcopy(self.abstract_service.Time_candidates)
                Cost_candidates = copy.deepcopy(self.abstract_service.Cost_candidates)
                # 欧氏距离中被替换个体的参数
                if j == 0:
                    s_i_res = population[i][j][1]
                    s_i_exe = population[i][j][2]
                    s_i_cost = population[i][j][3]
                    s_i_del = population[i][j][4]
                    s_i_mash = population[i][j][5]
                    a, b, c = 0, 0, 0
                    for k in range(0, self.abstract_service.candidate_service_number):
                        for t in range(0, self.abstract_service.candidate_service_number):
                            for r in range(0, self.abstract_service.candidate_service_number):
                                s_j_res = Cost_candidates[k][0]
                                s_j_exe = Time_candidates[t][1]
                                s_j_cost = Cost_candidates[t][1]
                                s_j_del = Time_candidates[r][2]
                                s_j_mash = Cost_candidates[r][2]
                                # 计算种群第i个个体任务j的服务与第j个任务候选服务集中第k个候选服务的 欧氏距离
                                E_distance = abs(s_i_res - s_j_res) + abs(s_i_del - s_j_del) + abs(
                                    s_i_exe - s_j_exe) + abs(
                                    s_i_cost - s_j_cost + abs(s_i_mash - s_j_mash))
                                if E_distance < difference:
                                    map_service = [0, s_j_res, s_j_exe, s_j_cost, s_j_del, s_j_mash]
                                    difference = E_distance
                                    a = k
                                    b = t
                                    c = r
                    num.append(a)
                    num.append(b)
                    num.append(c)
                else:
                    s_i_exe = population[i][j][0]
                    s_i_cost = population[i][j][1]
                    s_i_del = population[i][j][2]
                    s_i_mash = population[i][j][3]
                    a, b = 0, 0
                    for k in range(0, self.abstract_service.candidate_service_number):
                        for t in range(0, self.abstract_service.candidate_service_number):
                            s_j_exe = Time_candidates[k][2 * j + 1]
                            s_j_cost = Cost_candidates[k][2 * j + 1]
                            s_j_del = Time_candidates[t][2 * j + 2]
                            s_j_mash = Cost_candidates[t][2 * j + 2]
                            # 计算种群第i个个体任务j的服务与第j个任务候选服务集中第k个候选服务的 欧氏距离
                            E_distance = abs(s_i_del - s_j_del) + abs(s_i_exe - s_j_exe) + abs(
                                s_i_cost - s_j_cost + abs(s_i_mash - s_j_mash))

                            # 为种群第i个个体任务j的服务 匹配欧氏距离最小的真实服务
                            if E_distance < difference:
                                map_service = [s_j_exe, s_j_cost, s_j_del, s_j_mash]
                                difference = E_distance
                                a = k
                                b = t
                    num.append(a)
                    num.append(b)
                # 将第i个个体第j个任务的真实服务添加进来
                temp_list.append(map_service)
            new_population_num.append(num)
            # 将第i个个体所有任务的真实服务添加进来
            new_population.append(temp_list)

        return new_population, new_population_num

    def find_gbest(self, population,population_num):#（已检测)
        """找到种群中的老师(Pareto解集)"""
        gbest = []
        gbest_num=[]
        [ParetoSet1,ParetoSet2] = copy.deepcopy(self.ParetoSearch(population,population_num))
        # 若pareto解集里只有一个解
        if len(ParetoSet1) == 1:
            gbest = copy.deepcopy(ParetoSet1[0])
            gbest_num = copy.deepcopy(ParetoSet2[0])
        # 若pareto解集里有多个解
        else:
            r = np.random.randint(0, len(ParetoSet1) - 1)
            gbest = copy.deepcopy(ParetoSet1[r])
            gbest_num=copy.deepcopy(ParetoSet2[r])

        return gbest,gbest_num