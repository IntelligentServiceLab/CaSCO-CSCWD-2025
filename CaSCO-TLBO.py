
import random
import numpy as np
import copy
import math

import pandas as pd

"""
    TLBO algorithm for Industrial Internet
"""
class TLBO:
    """教学优化算法"""

    def __init__(self, abstract_service, task_number, population_size, iteration_number, bounds):
        self.abstract_service = abstract_service  # 抽象服务组合链
        self.task_number = task_number  # 子任务数
        self.population_size = population_size  # 种群规模
        self.iteration_number = iteration_number  # 迭代次数
        self.bounds = bounds  # 候选服务集的上下界

    def initialization(self):#(已测试）
        """初始化阶段:根据种群规模，生成相应个数的个体（服务组合解）;
        通过为每个任务随机选择对应要求的资源来初始化一个组合服务"""
        population = []  # 种群
        population_num = []
        for i in range(0, self.population_size):
            temp = []
            num=[]
            if_services = []
            for j in range(0, self.task_number):
                else_services_j = []
                if j==0:
                    for k in range(0, 3):
                        high = self.abstract_service.candidate_service_number
                        r = random.randint(0, high - 1)
                        time_candidate=self.abstract_service.Time_candidates[r][k]  # 从第K个资源的时间候选服务集选择
                        cost_candidate=self.abstract_service.Cost_candidates[r][k]  # 添加对应的成本属性
                        #temp.extend([time_candidate, cost_candidate])
                        #service_combination = [time_candidate, cost_candidate]
                        if_services.extend([time_candidate, cost_candidate])
                        num.append(r)
                    temp.append(if_services)
                else:
                    for k in range(2*j+1,2*j+3):
                        high = self.abstract_service.candidate_service_number
                        r = random.randint(0, high - 1)
                        # print(r)
                        time_candidate = self.abstract_service.Time_candidates[r][k]  # 从第K个资源的时间候选服务集选择
                        cost_candidate = self.abstract_service.Cost_candidates[r][k]  # 添加对应的成本属性
                        #temp.extend([time_candidate, cost_candidate])
                        service_combination = [time_candidate, cost_candidate]
                        else_services_j.extend(service_combination)
                        num.append(r)
                    temp.append(else_services_j)
            population_num.append(num)
            population.append(temp)
        return population,population_num

    def teacher_phase(self, population,population_num, teacher):#（已测试）
        """教师阶段:所有个体通过老师和个体平均值的差值像老师;
        学习参数是 种群列表 和 候选服务集的上下界列表"""

        Mean = self.get_Mean(population)  # 每个任务的平均值列表
        old_population = copy.deepcopy(population)  # 保存算法开始前的种群
        old_population_num = copy.deepcopy(population_num)
        # old_population_fitness = self.fitness_evaluation(old_population)  # 保存旧种群的适应值

        # 这个循环遍历每个个体
        for i in range(0, self.population_size):
            for j in range(0, self.task_number):
                TF = round(1 + random.random())  # 教学因素 = round[1 + rand(0, 1)]
                r = random.random()  # ri=rand(0,1), 学习步长
                # 更新第i个解的第j个任务的响应时间
                if j==0:
                    difference_Res = r * (teacher[j][1] - TF * Mean[j][0])
                    population[i][j][0] += difference_Res
                    # 更新第i个解的第j个任务的人力成本
                    difference_Exe = r * (teacher[j][2] - TF * Mean[j][1])
                    population[i][j][2] += difference_Exe
                    # 更新第i个解的第j个任务的加工时间
                    difference_Cost = r * (teacher[j][3] - TF * Mean[j][2])
                    population[i][j][3] += difference_Cost
                    # 更新第i个解的第j个任务的物料成本
                    difference_Del = r * (teacher[j][4] - TF * Mean[j][3])
                    population[i][j][1] += difference_Del
                    # 更新第i个解的第j个任务的运输时间
                    difference_Mash = r * (teacher[j][5] - TF * Mean[j][4])
                    population[i][j][3] += difference_Mash
                    # 更新第i个解的第j个任务的机械成本
                else:
                    difference_Exe = r * (teacher[j][0] - TF * Mean[j][0])
                    population[i][j][2] += difference_Exe
                    # 更新第i个解的第j个任务的加工时间
                    difference_Cost = r * (teacher[j][1] - TF * Mean[j][1])
                    population[i][j][3] += difference_Cost
                    # 更新第i个解的第j个任务的物料成本
                    difference_Del = r * (teacher[j][2] - TF * Mean[j][2])
                    population[i][j][1] += difference_Del
                    # 更新第i个解的第j个任务的运输时间
                    difference_Mash = r * (teacher[j][3] - TF * Mean[j][3])
                    population[i][j][3] += difference_Mash
        # 在教师阶段方法内直接调用refine方法
        new_population = copy.deepcopy(self.refine(population, self.bounds))

        # 匹配运算结束以后的服务真实值——这个操作导致算法的时间复杂度随着候选服务集的数量增加而增加
        [new_population,new_population_num]= copy.deepcopy(self.FuzzingMatch(new_population))

        # 在教师阶段方法内直接调用update方法
        new_population,new_population_num = copy.deepcopy(self.update(old_population,old_population_num, new_population,new_population_num))

        return new_population,new_population_num

    def student_phase(self, population,population_num):#(已测试）
        """学生阶段"""
        old_population = copy.deepcopy(population)  # 保存算法开始前的旧种群
        old_population_num = copy.deepcopy(population_num)
        new_population = []  # 初始化新种群
        for i in range(0, self.population_size):
            num_list = self.get_list()  # 获得一个种群大小的数字列表
            num_list.remove(i)
            index = random.choice(num_list)  # 这两步获得一个除了自身以外的随机索引

            # print("第"+str(i)+"个选择了"+"第"+str(index)+"个")
            X1 = copy.deepcopy(population[i])
            X2=copy.deepcopy(population_num[i])
            Y1 = copy.deepcopy(population[index])  # 被选中与X交叉的个体
            Y2 = copy.deepcopy(population_num[index])
            # 如果X支配Y, X比Y好
            if (self.TimeFitness_1(X1, X2) + self.CostFitness_1(X1, X2)) < (self.TimeFitness_1(Y1, Y2) + self.CostFitness_1(Y1, Y2)):
                # 学习步长ri=rand(0,1)
                for j in range(0, self.task_number):
                    r = random.random()
                    # 更新第Y的第j个任务的响应时间
                    if j==0:
                        X1[j][1] += r * (X1[j][1] - Y1[j][1])
                        X1[j][2] += r * (X1[j][2] - Y1[j][2])
                        X1[j][3] += r * (X1[j][3] - Y1[j][3])
                        X1[j][4] += r * (X1[j][4] - Y1[j][4])
                        X1[j][5] += r * (X1[j][5] - Y1[j][5])
                    else:
                        X1[j][0] += r * (X1[j][0] - Y1[j][0])
                        X1[j][1] += r * (X1[j][1] - Y1[j][1])
                        X1[j][2] += r * (X1[j][2] - Y1[j][2])
                        X1[j][3] += r * (X1[j][3] - Y1[j][3])
            # 如果Y支配X, Y比X好
            elif (self.TimeFitness_1(X1, X2) + self.CostFitness_1(X1, X2)) > (self.TimeFitness_1(Y1, Y2) + self.CostFitness_1(Y1, Y2)):
                  # 学习步长ri=rand(0,1)
                for j in range(0, self.task_number):
                    r = random.random()
                    if j == 0:
                        X1[j][1] += r * (Y1[j][1]- X1[j][1])
                        X1[j][2] += r * (Y1[j][2]- X1[j][2])
                        X1[j][3] += r * (Y1[j][3] - X1[j][3])
                        X1[j][4] += r * (Y1[j][4] - X1[j][4])
                        X1[j][5] += r * (Y1[j][5] - X1[j][5])
                    else:
                        X1[j][0] += r * (Y1[j][0] - X1[j][0])
                        X1[j][1] += r * (Y1[j][1] - X1[j][1])
                        X1[j][2] += r * (Y1[j][2] - X1[j][2])
                        X1[j][3] += r * (Y1[j][3] - X1[j][3])

            # 若互相不支配，则两个目标函数分别学习
            else:
                # 若X的时间目标强于Y，成本目标弱于Y
                if (self.TimeFitness_1(X1,X2) < self.TimeFitness_1(Y1,Y2)) & (self.CostFitness_1(X1,X2) > self.CostFitness_1(Y1,Y2)):
                      # 学习步长ri=rand(0,1)
                    for j in range(0, self.task_number):
                        r = random.random()
                        if j == 0:
                            X1[j][1] += r * (X1[j][1] - Y1[j][1])
                            X1[j][2] += r * (X1[j][2] - Y1[j][2])
                            X1[j][3] += r * (X1[j][3] - Y1[j][3])
                            X1[j][4] += r * (X1[j][4] - Y1[j][4])
                            X1[j][5] += r * (X1[j][5] - Y1[j][5])
                        else:
                            X1[j][0] += r * (X1[j][0] - Y1[j][0])
                            X1[j][1] += r * (X1[j][1] - Y1[j][1])
                            X1[j][2] += r * (X1[j][2] - Y1[j][2])
                            X1[j][3] += r * (X1[j][3] - Y1[j][3])

                # 若X的时间目标弱于Y，成本目标强于Y
                else:
                     # 学习步长ri=rand(0,1)
                    for j in range(0, self.task_number):
                        r = random.random()
                        if j == 0:
                            X1[j][1] += r * (Y1[j][1] - X1[j][1])
                            X1[j][2] += r * (Y1[j][2] - X1[j][2])
                            X1[j][3] += r * (Y1[j][3] - X1[j][3])
                            X1[j][4] += r * (Y1[j][4] - X1[j][4])
                            X1[j][5] += r * (Y1[j][5] - X1[j][5])
                        else:
                            X1[j][0] += r * (Y1[j][0] - X1[j][0])
                            X1[j][1] += r * (Y1[j][1] - X1[j][1])
                            X1[j][2] += r * (Y1[j][2] - X1[j][2])
                            X1[j][3] += r * (Y1[j][3] - X1[j][3])
            new_population.append(X1)

        # 在教师阶段方法内直接调用refine方法
        new_population = copy.deepcopy(self.refine(population, self.bounds))

        # 匹配运算结束以后的服务真实值——这个操作导致算法的时间复杂度随着候选服务集的数量增加而增加
        [new_population,new_population_num] = copy.deepcopy(self.FuzzingMatch(new_population))

        # 在教师阶段方法内直接调用update方法
        [new_population,new_population_num] = copy.deepcopy(self.update(old_population,old_population_num, new_population,new_population_num))

        return new_population,new_population_num

    def find_teacher(self, population,population_num):#（已检测)
        """找到种群中的老师(Pareto解集)"""
        teacher = []
        [ParetoSet1,ParetoSet2] = copy.deepcopy(self.ParetoSearch(population,population_num))
        # 若pareto解集里只有一个解
        if len(ParetoSet1) == 1:
            teacher = copy.deepcopy(ParetoSet1[0])
        # 若pareto解集里有多个解
        else:
            r = np.random.randint(0, len(ParetoSet1) - 1)
            teacher = copy.deepcopy(ParetoSet1[r])

        return teacher

    def refine(self, population, bounds):#(已测试）
        """refine操作符:防止越界"""
        # 第一步
        # for i in range(0, self.population_size):
        #     for j in range(0, self.task_number):
        #         population[i][j] = round(population[i][j])

        # 第二步
        for i in range(0, self.population_size):
            for j in range(0, self.task_number):
                # 如果超过上界,等于上界
                if j==0:
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

    def ParetoSearch(self, population,population_num):#(已测试）
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
            TimeFit = copy.deepcopy(self.TimeFitness_1(population[i],population_num[i]))
            # 添加种群中第i个个体的成本适应值
            CostFit = copy.deepcopy(self.CostFitness_1(population[i],population_num[i]))
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

    def CostFitness(self, solution1,solution2):#(已测试）
        """
            成本消耗适应函数
        """
        df_a = pd.read_csv('lab/Manu_label.csv',header=None)
        Cost_Fit = 0  # 解的总体成本消耗
        discount = {}
        for i in range(1, 11):
            column_index = i
            row_index = solution2[i]
            value = df_a.iloc[row_index,column_index-1]

            if value in discount:
                discount[value] += 1
            else:
                discount[value] = 1
            #discount[df_a[df_a.columns[i-1]][solution2[i]]]+=1
        Cost_Fit = Cost_Fit + solution1[0][1]
        for i in range(1, 11) :
            if i<3:
                price=solution1[0][2*i+1]
                cnt=self.Discount(discount[df_a[df_a.columns[i-1]][solution2[i]]])
                Cost_Fit+=price*cnt
            else:
                p=math.floor((i-1)/2)
                price = solution1[p][2*(i-2*p-1)+1]
                cnt = self.Discount(discount[df_a[df_a.columns[i-1]][solution2[i]]])
                Cost_Fit += price * cnt
        return Cost_Fit
    def Discount(self,x):#（已测试）
        """折扣函数"""
        count=0.5 + 1 / (1 + np.exp(0.1 * (x - 1)))
        return count

    def update(self, old_group,old_group_num, new_group,new_group_num):#（已测试）
        """这个函数用来更新种群:若新解支配旧解，则替换;否则保留"""

        updated_group = []
        updated_group_num = []
        for i in range(self.population_size):
            # 如果新解支配旧解
            # if ((self.TimeFitness(new_group[i],new_group_num[i]) < self.TimeFitness(old_group[i],old_group_num[i])) and (
            #         self.CostFitness(new_group[i],new_group_num[i]) < self.CostFitness(old_group[i],old_group_num[i]))) or (
            #         (round(self.TimeFitness(new_group[i],new_group_num[i])) == round(self.TimeFitness(old_group[i],old_group_num[i]))) and (
            #         self.CostFitness(new_group[i],new_group_num[i]) < self.CostFitness(old_group[i],old_group_num[i]))) or (
            #         (self.TimeFitness(new_group[i],new_group_num[i]) < self.TimeFitness(old_group[i],old_group_num[i])) and (
            #         round(self.CostFitness(new_group[i],new_group_num[i])) == round(self.CostFitness(old_group[i],old_group_num[i])))):
            if(self.TimeFitness_1(new_group[i],new_group_num[i])+self.CostFitness_1(new_group[i],new_group_num[i])) <(self.TimeFitness_1(old_group[i],old_group_num[i])+self.CostFitness_1(old_group[i],old_group_num[i])):
                updated_group.append(new_group[i])
                updated_group_num.append(new_group_num[i])
            else:
                updated_group.append(old_group[i])
                updated_group_num.append(old_group_num[i])

        return updated_group,updated_group_num

    def get_list(self):#（已测试）
        """"为了学生阶段获得一个种群大小的数字列表"""
        nums_list = []
        for i in range(0, self.population_size):
            nums_list.append(i)
        return nums_list

    def get_Mean(self, population):#(已测试）
        """获得种群中 每个任务 的平均值;
           参数为种群;
           返回值为每个任务平均值的列表
        """
        Mean = []
        for i in range(0, self.task_number):

            # 第i个任务的人力成本之和
            Sum_Res = 0
            # 第i个任务的运输时间之和
            Sum_Del = 0
            # 第i个任务的执行时间之和
            Sum_Exe = 0
            # 第i个任务的物料成本之和
            Sum_Cost = 0
            ## 第i个任务的机械成本之和
            Sum_MASH = 0
            # 第i个任务的平均值列表
            Mean_i = []
            if i==0:
                for j in range(0, self.population_size):
                    # 人力成本
                    Sum_Res += population[j][i][1]
                    # 加工时间
                    Sum_Exe += population[j][i][2]
                    # 物料成本
                    Sum_Cost += population[j][i][3]
                    # 运输时间
                    Sum_Del += population[j][i][4]
                    #机械成本
                    Sum_MASH += population[j][i][5]
                Mean_i.append(Sum_Res / self.population_size)
                Mean_i.append(Sum_Exe / self.population_size)
                Mean_i.append(Sum_Cost / self.population_size)
                Mean_i.append(Sum_Del / self.population_size)
                Mean_i.append(Sum_MASH / self.population_size)
                Mean.append(Mean_i)
            else:
                for j in range(0, self.population_size):

                    # 加工时间
                    Sum_Exe += population[j][i][0]
                    # 物料成本
                    Sum_Cost += population[j][i][1]
                    # 运输时间
                    Sum_Del += population[j][i][2]
                    # 机械成本
                    Sum_MASH += population[j][i][3]
                Mean_i.append(Sum_Exe / self.population_size)
                Mean_i.append(Sum_Cost / self.population_size)
                Mean_i.append(Sum_Del / self.population_size)
                Mean_i.append(Sum_MASH / self.population_size)
                Mean.append(Mean_i)
        return Mean

    def FuzzingMatch(self, population):#(已测试)
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
                Time_candidates= copy.deepcopy(self.abstract_service.Time_candidates)
                Cost_candidates= copy.deepcopy(self.abstract_service.Cost_candidates)
                # 欧氏距离中被替换个体的参数
                if j==0:
                    s_i_res = population[i][j][1]
                    s_i_exe = population[i][j][2]
                    s_i_cost = population[i][j][3]
                    s_i_del = population[i][j][4]
                    s_i_mash = population[i][j][5]
                    a,b,c=0,0,0
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
                                    map_service = [0,s_j_res, s_j_exe, s_j_cost, s_j_del, s_j_mash]
                                    difference = E_distance
                                    a=k
                                    b=t
                                    c=r
                    num.append(a)
                    num.append(b)
                    num.append(c)
                else:
                    s_i_exe = population[i][j][0]
                    s_i_cost = population[i][j][1]
                    s_i_del = population[i][j][2]
                    s_i_mash = population[i][j][3]
                    a,b=0,0
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
                                    a=k
                                    b=t
                    num.append(a)
                    num.append(b)
                # 将第i个个体第j个任务的真实服务添加进来
                temp_list.append(map_service)
            new_population_num.append(num)
            # 将第i个个体所有任务的真实服务添加进来
            new_population.append(temp_list)

        return new_population, new_population_num