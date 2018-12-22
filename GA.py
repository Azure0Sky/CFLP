import numpy as np
import timeit
from random import choice, randint, random, sample
from Greedy import Greedy
from io import StringIO

sol = []


class Genetic(object):
    def __init__(self, data_path='', alpha=1.):
        self.demand = np.zeros(1)
        self.cost = np.zeros((1, 1))
        self.open_cost = np.zeros(1)
        self.capacity = np.zeros(1)
        self.facility_num = 0
        self.customer_num = 0
        self.fcy_rate = []
        self.csr_rate = []
        self.alpha = alpha
        self.pc = 0.95
        self.pm = 0.1
        self.size = 0
        self.data_path = data_path

        self.__read_data()

    def __read_data(self):
        with open(self.data_path, 'r') as instance:
            # first line
            self.facility_num, self.customer_num = list(map(int, instance.readline().split()))
            self.capacity = np.zeros(self.facility_num)
            self.open_cost = np.zeros(self.facility_num)
            self.cost = np.zeros((self.facility_num, self.customer_num))
            self.demand = np.zeros(self.customer_num)

            # facility
            for i in range(self.facility_num):
                self.capacity[i], self.open_cost[i] = list(map(float, instance.readline().split()))

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

    def __init_population(self):
        population = []
        total_demand = sum(self.demand)

        # size represents the number of population
        self.size = self.facility_num * 2

        for i in range(self.facility_num):
            one = []
            for j in range(self.customer_num):
                one.append((j, self.cost[i, j] / self.demand[j]))
            one.sort(key=lambda t: t[1])
            self.fcy_rate.append(one)

        for i in range(self.customer_num):
            one = []
            for j in range(self.facility_num):
                one.append((j, self.cost[j, i] / self.demand[i]))
            one.sort(key=lambda t: t[1])
            self.csr_rate.append(one)

        for k in range(self.size):
            # assign a facility for each customer

            individual = [-1] * self.customer_num
            load = {}

            # randomly pick one from top 5
            first_facility = k // 2
            customer_idx = choice(self.fcy_rate[first_facility][:5])[0]
            individual[customer_idx] = first_facility
            load[first_facility] = self.demand[customer_idx]
            curr_cap = self.capacity[first_facility]

            for i in range(self.customer_num):
                if individual[i] != -1:
                    continue

                # approximate probability that open a new facility
                p = 0.4 if curr_cap < total_demand else 0.1

                # randomly pick one from the open facilities
                # or randomly pick one from all the facilities
                fcy_candidate = list(load.keys()) if (random() > p) else range(self.facility_num)
                for j in range(self.facility_num):
                    if self.csr_rate[i][j][0] not in fcy_candidate:
                        continue

                    fcy_idx = self.csr_rate[i][j][0]
                    fcy_num_before = len(load)
                    if load.setdefault(fcy_idx, 0) + self.demand[i] <= self.capacity[fcy_idx]:
                        individual[i] = fcy_idx
                        load[fcy_idx] += self.demand[i]
                        # new open
                        if fcy_num_before < len(load):
                            curr_cap += self.capacity[fcy_idx]
                        break

                # can't be assigned to all the open facility
                if individual[i] == -1:
                    idx = randint(0, self.facility_num - 1)
                    while idx in load.keys():
                        idx = randint(0, self.facility_num - 1)

                    individual[i] = idx
                    load[idx] = self.demand[i]
                    curr_cap += self.capacity[idx]

            population.append((individual, self.__fitness(individual)))

        return population

    def __fitness(self, individual):
        total_cost = 0
        load = {}
        for i in range(len(individual)):
            total_cost += self.cost[individual[i], i]
            if individual[i] in load:
                load[individual[i]] += self.demand[i]
            else:
                load[individual[i]] = self.demand[i]

        punish = 0
        overload = False
        for idx, val in load.items():
            if self.capacity[idx] < val:
                overload = True
                punish += (self.alpha * (val - self.capacity[idx]) *
                           self.fcy_rate[idx][self.customer_num - 1][1])

        total_open = 0
        for fcy_idx in load.keys():
            total_open += self.open_cost[fcy_idx]

        total_cost += total_open

        f = 100000 - (total_cost + punish)
        return (f, total_cost, overload) if f > 0 else (0, total_cost, overload)

    def __breed_new(self, population):
        offspring = []
        while len(offspring) < self.size:
            par_1 = max(sample(population, 2), key=lambda one: one[1][0])
            par_2 = max(sample(population, 2), key=lambda one: one[1][0])
            ca, cb = self.__crossover(par_1[0], par_2[0])

            self.__mutation(ca)
            self.__mutation(cb)

            offspring.append((ca, self.__fitness(ca)))
            offspring.append((cb, self.__fitness(cb)))
        return offspring

    def __crossover(self, par_1, par_2):
        begin = randint(0, len(par_1) - 1)
        end = randint(begin, len(par_1) - 1)

        child_1 = par_1[:]
        child_2 = par_2[:]

        if random() < self.pc:
            child_1[begin:end + 1] = par_2[begin:end + 1]
            child_2[begin:end + 1] = par_1[begin:end + 1]

        return child_1, child_2

    def __mutation(self, individual):
        if random() > self.pm:
            return

        length = randint(1, len(individual) // 2)
        open_fcy = list(set(individual))

        for i in range(length):
            idx = randint(0, len(individual) - 1)
            individual[idx] = choice(open_fcy) if random() < 0.5 else randint(0, self.facility_num - 1)

    def __update(self, pool):
        new_pop = []

        while len(new_pop) < self.size:
            winner = max(sample(pool, self.size // 3), key=lambda one: one[1][0])
            pool.remove(winner)
            new_pop.append(winner)

        return new_pop
        # pool.sort(key=lambda one: one[1][0], reverse=True)
        # return pool[:self.size]

    def exec(self):
        global sol

        population = self.__init_population()

        best = population[1]
        gen_num = 10000
        curr_gen = 0
        cnt = 0

        while curr_gen < gen_num:
            offsprings = self.__breed_new(population)
            population.extend(offsprings)
            population = self.__update(population)

            # population has been sorted
            i = 0
            while i < len(population) and population[i][1][2]:
                i += 1

            if i < len(population) and best[1][0] < population[i][1][0]:
                self.pm = 0.1
                best = population[i][:]
                cnt = 0
            else:
                cnt += 1

            if cnt == 20:
                self.pm += 0.1
                cnt = 0

            curr_gen += 1

        print('\n', best)
        sol.append((best + (self.facility_num,)))


def write_sol():
    global sol

    cnt = 1
    with open('sol.md', 'w') as sol_file:
        for s in sol:
            open_fcy = set(s[0])
            fcy_seq = [0] * s[2]
            for f in open_fcy:
                fcy_seq[f] = 1

            sol_file.write('**' + str(cnt) + '**\n')

            sol_file.write(str(s[1][1]) + '\n')

            sio = StringIO()
            for i in fcy_seq:
                sio.write(str(i) + ' ')
            sol_file.write(sio.getvalue() + '\n')

            sio = StringIO()
            for i in s[0]:
                sio.write(str(i) + ' ')
            sol_file.write(sio.getvalue() + '\n')

            sol_file.write('\n----\n')

            cnt += 1


def main():
    global sol
    for i in range(1, 2):
        path = '../Instances/p' + str(i)
        start = timeit.default_timer()
        ga = Genetic(path, 1)
        ga.exec()
        print(timeit.default_timer() - start, '\n====')
        # greedy = Greedy(path)
        # greedy.exec()

    # write_sol()


if __name__ == '__main__':
    main()
