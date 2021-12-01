import numpy as npy
import sys
from Graph import *
from util import *
from ALG import *

debug = 0


class FileReader(object):
    class Instance(object):
        def __init__(self, name, vertices, depot, r_edges, n_edges, vehicles,
                     capacity, total_cost, r_edge_list, n_edge_list, edges, terminal, seed):
            self.name = name
            self.vertices = int(vertices)
            self.depot = int(depot)
            self.r_edges = int(r_edges)
            self.n_edges = int(n_edges)
            self.vehicles = int(vehicles)
            self.capacity = int(capacity)
            self.total_cost = int(total_cost)
            self.r_edge_list = r_edge_list
            self.n_edge_list = n_edge_list
            self.edges = edges
            self.terminal = terminal
            self.seed = seed

    def __init__(self, argv):
        path = argv[1]
        if argv[2] == '-t':
            self.termination = argv[3]
            self.seed = argv[5]
        elif argv[2] == '-s':
            self.seed = argv[5]
            self.termination = argv[3]
        with open(path, 'r') as f:
            name = f.readline().split(' : ')[1].replace('\n', '')
            vertices = f.readline().split(' : ')[1].replace('\n', '')
            depot = f.readline().split(' : ')[1].replace('\n', '')
            r_edges = f.readline().split(' : ')[1].replace('\n', '')
            n_edges = f.readline().split(' : ')[1].replace('\n', '')
            vehicles = f.readline().split(' : ')[1].replace('\n', '')
            capacity = f.readline().split(' : ')[1].replace('\n', '')
            total_cost = f.readline().split(' : ')[1].replace('\n', '')
            if debug:
                print(name, vertices, depot, r_edges, n_edges, vehicles, capacity, total_cost)
            f.readline()  # ignore the empty line
            line = f.readline()
            r_edge_list = []
            n_edge_list = []
            edges = []
            while 'END' not in line:
                tmp = line.replace('\n', '').split(' ')
                while '' in tmp:
                    tmp.remove('')
                if debug:
                    print(tmp)
                if int(tmp[3]) == 0:
                    n_edge_list.append((int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])))
                else:
                    r_edge_list.append((int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])))
                edges.append((int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])))
                line = f.readline()
            self.instance = self.Instance(name, vertices, depot, r_edges, n_edges, vehicles, capacity, total_cost,
                                          r_edge_list, n_edge_list, edges, self.termination, self.seed)

    def getSample(self):
        return self.instance

class HeuristicSolver(object):
    def __init__(self, ins):
        self.instance = ins
        self.g = Graph(self.instance.vertices, self.instance.edges).graph

    def solve(self, initial_solution, initial_cost):
        solver = Solver(self.instance, self.g)
        state, cost = solver.work(initial_solution)
        show_result(state, cost)

class ExactCARPSolver(object):
    def __init__(self, ins):
        self.instance = ins
        self.g = Graph(self.instance.vertices, self.instance.edges).graph
        edges = {}
        _edges = {}
        for edge in ins.r_edge_list:
            edges[(edge[0], edge[1])] = (edge[2], edge[3])
            edges[(edge[1], edge[0])] = (edge[2], edge[3])
        self.r_edges = edges

        for edge in ins.n_edge_list:
            _edges[(edge[0], edge[1])] = (edge[2], edge[3])
            _edges[(edge[1], edge[0])] = (edge[2], edge[3])
        self.all_edges = _edges

    def createSolution(self):
        free = self.r_edges
        routes = []
        pointer = self.instance.depot
        cost = 0
        while free:
            route = []
            cap = 0
            while 1:
                if not free:
                    break
                tmp = []
                flag = False
                for key, value in free.items():
                    tmp.append((self.g[pointer - 1][key[0]-1], value[0], value[1], key[0], key[1]))
                tmp.sort(key=lambda x: x[0])
                for choose in tmp:
                    if cap + choose[2] <= self.instance.capacity:
                        # can go to such edge
                        cost += choose[0]
                        cost += choose[1]
                        del free[(choose[3], choose[4])]
                        del free[(choose[4], choose[3])]
                        flag = True
                        route.append((choose[3], choose[4]))
                        pointer = choose[4]
                        cap += choose[2]
                        break
                if not flag:
                    break
            routes.append(route)
            cost += self.g[pointer - 1][self.instance.depot - 1]
            pointer = self.instance.depot
        return routes, cost

    def solve(self):
        ini_solution, ini_cost = self.createSolution()
        if debug:
            print(ini_solution)
            print(ini_cost)
        # s 0,(2,3),(3,4),0,0,(8,7),(7,6),(6,5),0
        # q 25
        return ini_solution, ini_cost


if __name__ == '__main__':
    start_time_0 = time()
    if debug:
        print('Reading file...')
    instance = FileReader(sys.argv).getSample()
    if debug:
        print('%s step, using time %.8f s.' % ('Reading file', time() - start_time_0))
    #####################
    # Reading file part #
    #####################
    solver1 = ExactCARPSolver(instance)
    solver2 = HeuristicSolver(instance)
    initial_solution, ini_cost = solver1.solve()
    solver2.solve(initial_solution, ini_cost)
    if debug:
        print('Ending... Completing all the calculations, the total time is {}'.format(time() - start_time_0))
