import sys


def show_result(solution, cost):
    print('s ', end='')
    for ind, route in enumerate(solution):
        print('0,', end='')
        for edge in route:
            print('({},{}),'.format(edge[0], edge[1]), end='')
        if ind == len(solution) - 1:
            print('0')
        else:
            print('0,', end='')
    print('q {}'.format(cost))


if __name__ == '__main__':
    pass