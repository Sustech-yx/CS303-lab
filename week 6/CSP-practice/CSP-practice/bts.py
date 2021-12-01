"""
@DATE: 2021/10/17
@Author: Ziqi Wang
@File: bts.py
"""

import numpy as np
import time


def my_range(start, end):
    if start <= end:
        return range(start, end + 1)
    else:
        return range(start, end - 1, -1)


class Problem:
    char_mapping = ('·', 'Q')

    def __init__(self, _n=4):
        self.n = _n

    def is_valid(self, state):
        """
        check the state satisfy condition or not.
        :param state: list of points (in (row id, col id) tuple form)
        :return: bool value of valid or not
        """
        board = self.get_board(state)
        res = True
        for point in state:
            _i, _j = point
            condition1 = board[:, _j].sum() <= 1
            condition2 = board[_i, :].sum() <= 1
            condition3 = self.pos_slant_condition(board, point)
            condition4 = self.neg_slant_condition(board, point)
            res = res and condition1 and condition2 and condition3 and condition4
            if not res:
                break
        return res

    def is_satisfy(self, state):
        return self.is_valid(state) and len(state) == self.n

    def next_action(self, point):
        _i, _j = point
        if 0 <= _i < self.n and 0 <= _j < self.n and _i * self.n + _j < self.n ** 2 - 1:
            _j += 1
            if _j == self.n:
                _j = 0
                _i += 1
            return _i, _j
        else:
            return None

    def pos_slant_condition(self, board, point):
        _i, _j = point
        tmp = min(self.n - _i - 1, _j)
        start = (_i + tmp, _j - tmp)
        tmp = min(_i, self.n - _j - 1)
        end = (_i - tmp, _j + tmp)
        rows = my_range(start[0], end[0])
        cols = my_range(start[1], end[1])
        return board[rows, cols].sum() <= 1

    def neg_slant_condition(self, board, point):
        _i, _j = point
        tmp = min(_i, _j)
        start = (_i - tmp, _j - tmp)
        tmp = min(self.n - _i - 1, self.n - _j - 1)
        end = (_i + tmp, _j + tmp)
        rows = my_range(start[0], end[0])
        cols = my_range(start[1], end[1])
        return board[rows, cols].sum() <= 1

    def get_board(self, state):
        board = np.zeros([self.n, self.n], dtype=int)
        for point in state:
            board[point] = 1
        return board

    def print_state(self, state):
        board = self.get_board(state)
        print('_' * (2 * self.n + 1))
        for row in board:
            for item in row:
                print(f'|{Problem.char_mapping[item]}', end='')
            print('|')
        print('-' * (2 * self.n + 1))


# back track search
columns = set()
diagonal1 = set()
diagonal2 = set()
flag: bool


def backtrack(q, row=0):
    global flag
    if row == n:
        flag = True
    if flag:
        return
    else:
        for _i in range(n):
            if _i in columns or row - _i in diagonal1 or row + _i in diagonal2:
                continue
            q[row] = _i
            columns.add(_i)
            diagonal1.add(row - _i)
            diagonal2.add(row + _i)
            backtrack(q, row + 1)
            columns.remove(_i)
            diagonal1.remove(row - _i)
            diagonal2.remove(row + _i)
            if flag:
                break
    pass


def bts(problem):
    global flag
    flag = False
    q = [-1] * problem.n
    backtrack(q, 0)
    action_state = []
    row = 0
    for col in q:
        action_state.append((col, row))
        row = row + 1

    yield action_state


if __name__ == '__main__':
    n = 20
    render = input() == 1
    p = Problem(n)
    if render:
        import pygame

        w, h = 90 * n + 10, 90 * n + 10
        screen = pygame.display.set_mode((w, h))
        screen.fill('black')
        action_generator = bts(p)
        clk = pygame.time.Clock()
        queen_img = pygame.image.load('./queen.png')
        while True:
            for event in pygame.event.get():
                if event == pygame.QUIT:
                    exit()
            try:
                actions = next(action_generator)
                screen.fill(0)
                for i in range(n + 1):
                    pygame.draw.rect(screen, 'white', (i * 90, 0, 10, h))
                    pygame.draw.rect(screen, 'white', (0, i * 90, w, 10))
                for action in actions:
                    i, j = action
                    screen.blit(queen_img, (10 + 90 * j, 10 + 90 * i))
                pygame.display.flip()
            except StopIteration:
                pass
            clk.tick(5)
        pass
    else:
        start_time = time.time()
        for actions in bts(p):
            pass
        print(time.time() - start_time)
        p.print_state(actions)
        # print(actions)
