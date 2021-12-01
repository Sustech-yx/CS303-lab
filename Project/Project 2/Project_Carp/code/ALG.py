from time import time
import numpy as npy
import math
import random
from copy import deepcopy


class Solver(object):
    def __init__(self, instance, g):
        # only the demand edges are in such list.
        self.edges = {}
        self.cap = instance.capacity
        for index, e in enumerate(instance.r_edge_list):
            self.edges[(e[0], e[1])] = {'cost': e[2], 'dem': e[3]}
            self.edges[(e[1], e[0])] = {'cost': e[2], 'dem': e[3]}
        self.n = len(self.edges.keys()) / 2
        self.terminal = int(instance.terminal)
        self.depot = int(instance.depot)
        self.stop_sec = self.terminal/20
        if self.stop_sec < 4:
            self.stop_sec = 4
        elif self.stop_sec > 6:
            self.stop_sec = 6
        self.g = g
        random.seed(instance.seed)

    def work(self, initial_solution):
        start_time = time()
        state, cost = initial_solution, self.calCost(initial_solution)
        temp_state, temp_cost = state, cost
        t = 0
        isValid = True
        while 1:
            T = 0.99**t
            new_state, new_cost = self.localSearch(state if random.random() < .75 else temp_state)
            # if temp_state is not None:
            #     print(cost, new_cost, self.checkValid(new_state), self.checkValid(temp_state))
            if self.checkValid(new_state):
                if not isValid:
                    if temp_cost - new_cost >= 0:
                        state = new_state
                        cost = new_cost
                        isValid = True
                        t += 1
                        continue
                    elif T != 0 and random.random() < npy.exp((temp_cost - new_cost) / T):
                        state = new_state
                        cost = new_cost
                        isValid = True
                        t += 1
                        continue
                    else:
                        isValid = False
            else:
                if isValid:
                    temp_state = deepcopy(state)
                    temp_cost = deepcopy(cost)

                isValid = False

            if cost - new_cost >= 0:
                state = new_state
                cost = new_cost
            elif T != 0 and random.random() < npy.exp((cost - new_cost) / T):
                state = new_state
                cost = new_cost

            if self.terminal - (time() - start_time) < self.stop_sec:
                break
            t += 1
        if isValid:
            return state, cost
        else:
            return temp_state, temp_cost

    def localSearch(self, state):
        new_state = deepcopy(state)
        cost = self.calCost(new_state)
        k = random.random()
        if k < 0.25:  # swap
            # if the new_state has only one route, then skip the swap trying
            if len(new_state) < 2:
                return state, cost
            i = random.randint(0, len(new_state) - 1)
            route1 = new_state[i]
            x = random.randint(0, len(route1) - 1)

            j = random.randint(0, len(new_state) - 1)
            route2 = new_state[j]
            y = random.randint(0, len(route2) - 1)

            new_state[i][x], new_state[j][y] = new_state[j][y], new_state[i][x]
            # print('swap')
            # self.checkCompleteTask(new_state)
        elif 0.25 < k < 0.3:  # flip
            x = random.randint(0, len(new_state)-1)
            y = random.randint(0, len(new_state[x])-1)

            edge = new_state[x][y]
            edge_ = (edge[1], edge[0])

            new_state[x][y] = edge_
            # print('flip')
            # self.checkCompleteTask(new_state)
        elif 0.3 < k < 0.65:  # insert
            # if the new_state has only one route, then skip the insert trying
            if len(new_state) < 2:
                return state, cost
            i = random.randint(0, len(new_state) - 1)
            route1 = new_state[i]
            x = random.randint(0, len(route1) - 1)

            j = random.randint(0, len(new_state) - 1)
            route2 = new_state[j]
            # insert the edge x in route1 into route2
            y = random.randint(0, len(route2))
            if i == j and x == y:
                return state, cost
            elif i == j:
                if x > y:
                    route2.insert(y, route1[x])
                    del route1[x+1]
                else:
                    route2.insert(y, route1[x])
                    del route1[x]
            else:
                route2.insert(y, route1[x])
                del route1[x]
            new_state[i] = route1
            new_state[j] = route2
            if len(new_state[i]) == 0:
                del new_state[i]
            # self.checkCompleteTask(new_state)
        else:  # 2-opt
            i = random.randint(0, len(new_state) - 1)
            route1 = new_state[i]
            # this operation will do a single 2-opt in route1
            _l = len(route1)
            left = random.randint(0, _l - 1)
            right = random.randint(0, _l - 1)
            # choose the route between left and right
            if left > right:
                left, right = right, left
            temp = route1[left:right]
            for j in range(0, len(temp)):
                tt = temp[j]
                a, b = tt
                tt = (b, a)
                temp[j] = tt
            route1[left:right] = temp[::-1]
            new_state[i] = route1
            # print('2-opt')
            # self.checkCompleteTask(new_state)
        # check whether the new_state is a valid state.
        # if self.checkValid(new_state):
        #     return new_state, self.calCost(new_state)
        # else:
        #     return state, cost
        # Now the invalid state will be consider
        return new_state, self.calCost(new_state)

    def checkValid(self, sol):
        for route in sol:
            cap = self.cap
            for edge in route:
                cap -= self.edges[edge]['dem']
                if cap < 0:
                    return False
        return True

    def checkCompleteTask(self, sol):
        # for debug usage
        temp_edges = deepcopy(self.edges)
        for route in sol:
            for edge in route:
                del temp_edges[edge]
                del temp_edges[(edge[1], edge[0])]
        print(temp_edges)
        del temp_edges

    def calCost(self, sol):
        res = 0
        for route in sol:
            pointer = self.depot
            for index, edge in enumerate(route):
                res += self.g[pointer-1][edge[0]-1]
                res += self.edges[edge]['cost']
                pointer = edge[1]
                if index == len(route) - 1:
                    res += self.g[pointer - 1][self.depot-1]
                    pointer = self.depot
        return res