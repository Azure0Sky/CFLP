import numpy as np


class Greedy:
    def __init__(self, data_path=''):
        self.demand = []
        self.cost = []
        self.open_cost = []
        self.capacity = []
        self.facility_num = 0
        self.customer_num = 0
        self.fcy_rate = []
        self.data_path = data_path

    def __read_data(self):
        with open(self.data_path, 'r') as instance:
            # first line
            self.facility_num, self.customer_num = list(map(int, instance.readline().split()))
            self.capacity = [-1] * self.facility_num
            self.open_cost = [-1] * self.facility_num
            self.cost = np.zeros((self.facility_num, self.customer_num))
            self.demand = np.zeros(self.customer_num)

            # facility
            for i in range(self.facility_num):
                self.capacity[i], self.open_cost[i] = list(map(float, instance.readline().split()))
                self.capacity[i] = (i, self.capacity[i])
                self.open_cost[i] = (i, self.open_cost[i])

            # customer demand
            cnt = 0
            while cnt < self.customer_num:
                line = list(map(float, instance.readline().split()))
                for dem in line:
                    self.demand[cnt] = dem
                    cnt += 1

            # assignment cost
            for i in range(self.facility_num):
                cost_arr = []
                while len(cost_arr) < self.customer_num:
                    cost_arr = cost_arr + list(map(float, instance.readline().split()))

                self.cost[i] = np.array(cost_arr)

        for i in range(self.facility_num):
            one = []
            for j in range(self.customer_num):
                one.append((j, self.cost[i, j] / self.demand[j]))
            one.sort(key=lambda t: t[1])
            self.fcy_rate.append(one)

    def exec(self):
        self.__read_data()

        self.open_cost.sort(key=lambda t: t[1])
        assign_num = 0
        total_cost = 0
        assignment = [-1] * self.customer_num

        i = 0
        while assign_num < self.customer_num:
            fcy_idx, oc = self.open_cost[i]
            total_cost += oc

            load = 0
            for j in range(self.customer_num):
                csr_idx = self.fcy_rate[fcy_idx][j][0]
                if assignment[csr_idx] != -1:
                    continue

                if load + self.demand[csr_idx] > self.capacity[fcy_idx][1]:
                    break

                assignment[csr_idx] = fcy_idx
                total_cost += self.cost[fcy_idx, csr_idx]
                load += self.demand[csr_idx]
                assign_num += 1

            i += 1

        print(assignment)
        print(total_cost)

