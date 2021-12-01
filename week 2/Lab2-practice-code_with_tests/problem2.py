"""
    Args:
        graph: a given graph, as an adjacency matrix, the non-connected edge are set to infinity
        start: the index of the start vertex
        end: the index of the end vertex

    Returns:
        distance: the distance from the start point to the end point.

    sample:
    input (file):
    1          =>neighbors of node 0
    6          =>distances of (0,1)
    2 4        =>neighbors of node 1
    3 10       =>distances of (1,2) (1,4)
    3          =>neighbors of node 2
    7          =>distances of node (2,3)
    4          =>neighbors of node 3
    9          =>distances of node (3,4)
    0 3        =>neighbors of node 4
    3 9        =>distances of node (4,0) (4,3)

    graph: a distance matrix from above file
    [[0, 6.0, inf, inf, inf],
    [inf, 0, 3.0, inf, 10.0],
    [inf, inf, 0, 7.0, inf],
    [inf, inf, inf, 0, 9.0],
    [3.0, inf, inf, 9.0, 0]]

    output:
    16
"""


def floyd(graph, start, end):
    ##############
    #  Please write your own code in the given space.
    #############
    length: int
    length = len(graph)
    for i in range(length):
        for j in range(length):
            for k in range(length):
                if graph[i][j] > graph[i][k] + graph[k][j]:
                    graph[i][j] = graph[i][k] + graph[k][j]
    distance = graph[start][end]
    # acknowledge: https://blog.csdn.net/Lwenjiyou_/article/details/79548577
    return distance


if __name__ == '__main__':
    for i in range(3):
        test_case = i+1
        print(f'case {test_case}')
        with open(f'test_cases/problem2/{test_case}.txt', 'r') as f:
            content = f.read().strip()
            lines = content.split('\n')
        n = len(lines) // 2
        graph = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            graph[i][i] = 0
        for i in range(n):
            neighbors = [*map(int, lines[2 * i].strip().split(' '))]
            distances = [*map(float, lines[2 * i + 1].strip().split(' '))]
            for j in range(len(neighbors)):
                k = neighbors[j]
                graph[i][k] = distances[j]
        # print(graph)
        print(floyd(graph, 0, n - 1))
