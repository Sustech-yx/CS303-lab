"""
Example:
input:
map:
---------
------x--
-x-------
---@-----
---##----
------x--
--x----x-
-x-------
---------
action:
0 0 3 3 0 3 3 1 1 1 1 1 3 1 1 2 2 2 2 2

output:
7 3

"""


def solve(map_tot, action_list):
    running_time: int
    position: tuple
    fail: bool
    fail = False
    running_time = 0
    position = ()
    snack = []
    n: int
    n = len(map_tot)
    m: int
    m = len(map_tot[0])
    for row in range(n):
        for col in range(m):
            if map_tot[row][col] == '@':
                snack.append((row, col))

    direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    def find(po: tuple):
        for p in direction:
            # print(p[0], p[1])
            if 0 <= p[0] + po[0] < n and 0 <= p[1] + po[1] < m:
                if map_tot[p[0] + po[0]][p[1] + po[1]] == '#':
                    map_tot[p[0] + po[0]][p[1] + po[1]] = '$'
                    return True, (p[0] + po[0], p[1] + po[1])
        return False, ()

    index = 0
    while True:
        flag, pos_ = find(snack[index])
        if flag:
            snack.append(pos_)
            index += 1
        else:
            break
    # print(snack)

    def check(temp: tuple):
        return 0 <= temp[0] < n and 0 <= temp[1] < m

    for move in action_list:
        adj = direction[move]
        new_pos = (snack[0][0] + adj[0], snack[0][1] + adj[1])
        if check(new_pos) and new_pos not in snack and map_tot[new_pos[0]][new_pos[1]] == '-':
            map_tot[new_pos[0]][new_pos[1]] = '$'
            snack.insert(0, new_pos)
            tail = snack.pop()
            map_tot[tail[0]][tail[1]] = '-'
        else:
            fail = True
            break
        running_time += 1
    position = snack[0]
    return fail, running_time, position


if __name__ == '__main__':
    for i in range(4):
        test_case = i + 1
        print(f'case {test_case}', end='\t')
        with open(f'test_cases/problem3/{test_case}-map.txt', 'r') as f:
            game_map = [list(line.strip()) for line in f.readlines()]
        # print(game_map)
        with open(f'./test_cases/problem3/{test_case}-actions.txt', 'r') as f:
            actions = [*map(int, f.read().split(' '))]
        # print(actions)
        notOk, time, pos = solve(game_map, actions)
        if not notOk:
            print('%d %d' % pos)
        else:
            print(time)