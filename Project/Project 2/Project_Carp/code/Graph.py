import numpy as npy


debug = 0

class Graph(object):
    def __init__(self, n, edges):
        self.n = n
        self.edges = edges
        self.graph = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            self.graph[i][i] = 0
        for edge in edges:
            self.graph[edge[0] - 1][edge[1] - 1] = edge[2]
        self.calDistance()

    def calDistance(self):
        # floyd
        length: int
        length = len(self.graph)

        for i in range(1, length):
            for j in range(0, i):
                m = min(self.graph[i][j], self.graph[j][i])
                self.graph[i][j], self.graph[j][i] = m, m

        for k in range(length):
            for i in range(length):
                for j in range(length):
                    if self.graph[i][j] > self.graph[i][k] + self.graph[k][j]:
                        self.graph[i][j] = self.graph[i][k] + self.graph[k][j]
        #####
        if debug:
            matrix = npy.array(self.graph)
            print(matrix)
            npy.savetxt('graph.txt', npy.c_[matrix], fmt='%d', delimiter='\t')
        #####
