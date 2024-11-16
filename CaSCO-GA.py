import math
import random
import numpy as np
import copy

import pandas as pd


"""
    Genetic algorithm for Industrial Internet
"""
class GA():
    """遗传算法"""
    def __init__(self, abstract_service, task_number, population_size, iteration_number, bounds):
        self.crossover_probability = 0.40  # 交叉率
        self.mutation_probability = 0.011  # 变异率
        self.abstract_service = abstract_service  # 抽象服务组合链
        self.task_number = task_number  # 任务数
        self.population_size = population_size  # 种群规模
        self.iteration_number = iteration_number  # 迭代次数
        self.bounds = bounds  # 候选服务集的上下界列表

    def initialization(self):#(已修改）
        """初始化阶段:根据种群规模，生成相应个数的个体（服务组合解）;
        通过为每个任务随机挑选候选服务来初始化一个组合服务"""
        population = []  # 种群
        population_num = []
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
            population_num.append(num)
            population.append(temp)
        return population, population_num

    def Selection(self, population, population_num):#(已测试）
        """选择操作：采用锦标赛选择算法(随机选择)（ps：由于本场景下，个体的适应值越小表示越好，故不宜使用轮盘赌选择算法）"""
        new_population = []
        new_population_num = []
        tournament_size = 2  # 锦标赛规模

        # 锦标赛
        for i in range(0, self.population_size):
            temp = copy.deepcopy(population)  # 临时列表，供锦标赛抽取
            temp_num =copy.deepcopy(population_num)
            a=random.randint(0, len(temp) - 1)
            b=random.randint(0, len(temp) - 1)
            competitor_a = temp[a]  # 随机抽取选手a
            competitor_a_num=temp_num[a]
            competitor_b = temp[b] # 随机抽取选手b
            competitor_b_num=temp_num[b]
            # 若a支配b
            if (self.TimeFitness_1(competitor_a, competitor_a_num) + self.CostFitness_1(competitor_a, competitor_a_num)) < self.TimeFitness_1(competitor_b, competitor_b_num)+self.CostFitness_1(competitor_b, competitor_b_num):
                    new_population.append(competitor_a)
                    new_population_num.append(competitor_a_num)
            # 若b支配a
            elif (self.TimeFitness_1(competitor_a, competitor_a_num) + self.CostFitness_1(competitor_a, competitor_a_num)) > self.TimeFitness_1(competitor_b, competitor_b_num)+self.CostFitness_1(competitor_b, competitor_b_num):
                new_population.append(competitor_b)
                new_population_num.append(competitor_b_num)
            # 若互相不支配
            else:
                Fitness_a = self.TimeFitness_1(competitor_a,competitor_a_num) + self.CostFitness_1(competitor_a,competitor_a_num)  # a的适应值之和
                Fitness_b = self.TimeFitness_1(competitor_b,competitor_b_num) + self.CostFitness_1(competitor_b,competitor_b_num)  # b的适应值之和
                if Fitness_a < Fitness_b:
                    new_population.append(competitor_a)
                    new_population_num.append(competitor_a_num)
                else:
                    new_population.append(competitor_b)
                    new_population_num.append(competitor_b_num)
        return new_population, new_population_num

    def Crossover(self, population,population_num):#(已测试）
        """交叉操作"""
        cp = self.crossover_probability  # 交叉概率
        new_population = []  # 初始化交叉完毕的种群
        new_population_num = []
        crossover_population = []  # 初始化需要交叉的种群
        crossover_population_num=[]
        # 根据交叉概率选出需要交叉的个体
        # for c in population:
        #     r = random.random()
        #     if r <= cp:
        #         crossover_population.append(c)
        #     else:
        #         new_population.append(c)
        for i in range(0, self.population_size):
            r = random.random()
            if r <= cp:
                crossover_population.append(population[i])
                crossover_population_num.append(population_num[i])
            else:
                new_population.append(population[i])
                new_population_num.append(population_num[i])

        # 需保证交叉的个体是偶数,若不是偶数，则删掉需交叉列表的最后一个元素
        if len(crossover_population) % 2 != 0:
            new_population.append(crossover_population[len(crossover_population) - 1])
            new_population_num.append(crossover_population_num[len(crossover_population) - 1])
            del crossover_population[len(crossover_population) - 1]
            del crossover_population_num[len(crossover_population) - 1]

        # crossover——单点交叉
        for i in range(0, len(crossover_population), 2):
            i_solution = crossover_population[i]
            i_num = crossover_population_num[i]
            j_solution = crossover_population[i + 1]
            j_num = crossover_population_num[i + 1]
            crossover_position = random.randint(1, self.task_number - 2)  # 随机生成一个交叉位
            left_i = copy.deepcopy(i_solution[0:crossover_position])
            left_i_num = copy.deepcopy(i_num[0:crossover_position*2+3])
            right_i = copy.deepcopy(i_solution[crossover_position:self.task_number])
            right_i_num = copy.deepcopy(i_num[crossover_position*2+3:11])
            left_j = copy.deepcopy(j_solution[0:crossover_position])
            left_j_num = copy.deepcopy(j_num[0:crossover_position*2+3])
            right_j = copy.deepcopy(j_solution[crossover_position:self.task_number])
            right_j_num = copy.deepcopy(j_num[crossover_position*2+3:11])
            # 生成新个体
            new_i = copy.deepcopy(left_i + right_j)
            new_i_num = copy.deepcopy(left_i_num + right_j_num)
            new_j = copy.deepcopy(left_j + right_i)
            new_j_num = copy.deepcopy(left_j_num + right_i_num)
            new_population.append(new_i)
            new_population_num.append(new_i_num)
            new_population.append(new_j)
            new_population_num.append(new_j_num)

            if (i + 1) == (len(crossover_population) - 1):
                break

        return new_population, new_population_num

    def Mutation(self, population,population_num):
        """变异操作"""
        mp = self.mutation_probability  # 变异率
        new_population = []  # 初始化变异后的种群
        new_population_num = []
        for i in range(0, self.population_size):
            r = random.random()
            if r <= mp:
                mutation_position = random.randint(0, self.task_number - 1)  # 变异位置
                temp = copy.deepcopy(population[i])
                temp_num = copy.deepcopy(population_num[i])
                if mutation_position == 0:
                    a=random.randint(0, 9)
                    b=random.randint(0, 9)
                    c=random.randint(0, 9)
                    replaced_Res=self.abstract_service.Cost_candidates[mutation_position][a]
                    # 更新第i个解的第j个任务的人力成本
                    replaced_Exe =self.abstract_service.Time_candidates[mutation_position][b]
                    # 更新第i个解的第j个任务的加工时间
                    replaced_Cost =self.abstract_service.Cost_candidates[mutation_position][b]
                    # 更新第i个解的第j个任务的物料成本
                    replaced_Del =self.abstract_service.Time_candidates[mutation_position][c]
                    # 更新第i个解的第j个任务的运输时间
                    replaced_Mash =self.abstract_service.Cost_candidates[mutation_position][c]
                    replaced_service=[0,replaced_Res,replaced_Exe,replaced_Cost,replaced_Del,replaced_Mash]
                    temp[mutation_position] = replaced_service
                    temp_num[0] = a
                    temp_num[1] = b
                    temp_num[2] = c
                    new_population.append(temp)
                    new_population_num.append(temp_num)
                else:
                    a = random.randint(0, 9)
                    b = random.randint(0, 9)
                    replaced_Exe = self.abstract_service.Time_candidates[mutation_position][a]
                    # 更新第i个解的第j个任务的加工时间
                    replaced_Cost = self.abstract_service.Cost_candidates[mutation_position][a]
                    # 更新第i个解的第j个任务的物料成本
                    replaced_Del = self.abstract_service.Time_candidates[mutation_position][b]
                    # 更新第i个解的第j个任务的运输时间
                    replaced_Mash = self.abstract_service.Cost_candidates[mutation_position][b]
                    replaced_service = [replaced_Exe, replaced_Cost,replaced_Del,replaced_Mash]
                    temp[mutation_position] = replaced_service
                    temp_num[3+2*(mutation_position-1)] = a
                    temp_num[4+2*(mutation_position-1)] = b
                    new_population.append(temp)
                    new_population_num.append(temp_num)
            else:
                new_population.append(copy.deepcopy(population[i]))
                new_population_num.append(copy.deepcopy(population_num[i]))

        # # 调用refine方法，确保不越界
        # new_population = self.refine(new_population, bounds)

        return new_population,new_population_num

    def ParetoSearch(self, population,population_num):#（已修改）
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

    def TimeFitness(self, solution1,solution2):#(已测试）
        """
            时间消耗适应函数
        """
        Time_fit = 0
        radio = [0.0] * self.task_number
        radio1 = [0.0] * self.task_number
        df_a = pd.read_csv('lab/history_records.csv')
        for i in range(0, self.task_number):
            if i==0:
                # 假设 solution2 是一个列表，包含三个值
                # 例如: solution2 = [value1, value2, value3]
                # 使用列的索引来访问数据
                count1 = df_a[(df_a.iloc[:, 0] == solution2[0]) & (df_a.iloc[:, 1] == solution2[1])].shape[0]
                count2 = df_a[(df_a.iloc[:, 1] == solution2[1]) & (df_a.iloc[:, 2] == solution2[2])].shape[0]
                count3 = df_a[(df_a.iloc[:, 0] == solution2[0]) & (df_a.iloc[:, 2] == solution2[2])].shape[0]
                radio[i] = (count1 + count2 + count3) / (3 * 400)
                #print(radio[i])

            else:
                count1 = df_a[(df_a[df_a.columns[2*i+1]] == solution2[2*i+1] )& (df_a[df_a.columns[2*i+2]] == solution2[2*i+2])].shape[0]
                radio[i]=count1/400
                #print(radio[i])
        for i in range(0, self.task_number-1):
            if i==0:
                count1 = df_a[(df_a[df_a.columns[0]] == solution2[0]) & (df_a[df_a.columns[3]] == solution2[3])].shape[0]
                count2 = df_a[(df_a[df_a.columns[0]] == solution2[0]) & (df_a[df_a.columns[4]] == solution2[4])].shape[0]
                count3 = df_a[(df_a[df_a.columns[1]] == solution2[1]) & (df_a[df_a.columns[3]] == solution2[3])].shape[0]
                count4 = df_a[(df_a[df_a.columns[1]] == solution2[1]) & (df_a[df_a.columns[4]] == solution2[4])].shape[0]
                count5 = df_a[(df_a[df_a.columns[2]] == solution2[2]) & (df_a[df_a.columns[3]] == solution2[3])].shape[0]
                count6 = df_a[(df_a[df_a.columns[2]] == solution2[2]) & (df_a[df_a.columns[4]] == solution2[4])].shape[0]
                radio1[i]=(count1+count2+count3+count4+count5+count6)/(6*400)
                #print(radio1[i])
            else:
                count1 = df_a[(df_a[df_a.columns[2*i+1]] == solution2[2*i+1]) & (df_a[df_a.columns[2*i+3]] == solution2[2*i+3])].shape[0]
                count2 = df_a[(df_a[df_a.columns[2*i+1]] == solution2[2*i+1]) & (df_a[df_a.columns[2*i+4]] == solution2[2*i+4])].shape[0]
                count3 = df_a[(df_a[df_a.columns[2*i+2]] == solution2[2*i+2]) & (df_a[df_a.columns[2*i+3]] == solution2[2*i+3])].shape[0]
                count4 = df_a[(df_a[df_a.columns[2*i+2]] == solution2[2*i+2]) & (df_a[df_a.columns[2*i+4]] == solution2[2*i+4])].shape[0]
                radio1[i]=(count1+count2+count3+count4)/(4*400)
                #print(radio1[i])
        for i in range(0, self.task_number):
            if i==0:
                time=solution1[i][2]+solution1[i][4]
                Time_fit+=time*(1-radio1[i])*(1-radio[i])
            else:
                time=solution1[i][0]+solution1[i][2]
                if i!=4:
                    Time_fit+=time*(1-radio1[i])*(1-radio[i])
                else :
                    Time_fit+=time*(1-radio[i])
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

    def Discount(self,x):#（已测试）
        """折扣函数"""
        count=0.5 + 1 / (1 + np.exp(0.1 * (x - 1)))
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
            if (self.TimeFitness_1(new_group[i], new_group_num[i]) + self.CostFitness_1(new_group[i],
                                                                                        new_group_num[i])) < (
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