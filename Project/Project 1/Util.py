import numpy as npy
import threading
import random
from ai import *
import os
import time

debug = 1
ini_board = npy.zeros([8, 8], dtype=npy.int8)
ini_board[3][3], ini_board[4][4], ini_board[3][4], ini_board[4][3] = 1, 1, -1, -1
random.seed(114514)
ini_map = npy.array([
    [-100, 7, -10, -7, -7, -10, 7, -100],
    [7, 10, -3, -3, -3, -3, 10, 7],
    [-10, -3, -5, -4, -4, -5, -3, -10],
    [-7, -3, -4, 0, 0, -4, -3, -7],
    [-7, -3, -4, 0, 0, -4, -3, -7],
    [-10, -3, -5, -4, -4, -5, -3, -10],
    [7, 10, -3, -3, -3, -3, 10, 7],
    [-100, 7, -10, -5, -5, -10, 7, -100]
])


class BrooderHouse:
    class BattleSystem:
        def __init__(self, map_list):
            self.map_list = map_list
            self.map_num = len(map_list)
            self.result = [0] * self.map_num
            self.battle_cnt = 0

        def threa(self):
            t1 = time.time()
            threads = []
            for offset in range(1, self.map_num):
                for a1 in range(0, self.map_num):
                    a2 = (a1 + offset) % self.map_num
                    map_1 = self.map_list[a1]
                    map_2 = self.map_list[a2]
                    print('Battle between {} and {}'.format(a1, a2))
                    t = threading.Thread(target=self.battle, args=(map_1, map_2, (a1, a2)))
                    threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # 需要返回所有ai的矩阵和获胜次数
            print('This state is end, use time {}.'.format(time.time() - t1))
            res = []
            for i in range(0, self.map_num):
                res.append((self.map_list[i], self.result[i]))
            sorted(res, key=lambda x: x[1])
            return res

        def battle(self, map1, map2, index):
            ai1 = AI(8,  1, 5, map1)
            ai2 = AI(8, -1, 5, map2)
            board = ini_board.copy()
            cnt = 0
            while (cnt % 2 == 0 and not ai1.isTerminal(board)[0]) or (cnt % 2 == 1 and not ai2.isTerminal(board)[0]):
                if cnt % 2 == 0:  # ai 1 go
                    ai1.go(board)
                    if len(ai1.candidate_list) == 0:
                        cnt += 1
                        continue
                    move = ai1.candidate_list[-1]
                    if move not in ai1.genValidPos(chessboard=board, color=ai1.color):
                        if debug:
                            print(ai1.genValidPos(chessboard=board, color=ai1.color))
                            print(ai1.color)
                            print(board)
                            print(move)
                        return
                    board = ai1.action(board, move, ai1.color)
                else:
                    ai2.go(board)
                    if len(ai2.candidate_list) == 0:
                        cnt += 1
                        continue
                    move = ai2.candidate_list[-1]
                    if move not in ai2.genValidPos(chessboard=board, color=ai2.color):
                        if debug:
                            print(ai2.genValidPos(chessboard=board, color=ai2.color))
                            print(ai2.color)
                            print(board)
                            print(move)
                        return
                    board = ai2.action(board, move, ai2.color)
                # print('move')
                cnt += 1
            a1 = len(npy.where(board == ai1.color)[0])
            a2 = len(npy.where(board == ai2.color)[0])
            if a1 > a2:
                self.result[index[1]] += 1
            elif a1 < a2:
                self.result[index[0]] += 1
            self.battle_cnt += 1
            print('The {}th battle has over, the winner is ai{}'
                  .format(self.battle_cnt, index[0] if a1 < a2 else ([index[1]] if a1 > a2 else 'draw')))

    def __init__(self, count):
        self.map_list = [ini_map] * 4
        for i in range(2, 4):
            self.map_list[i] = self.mutant(self.map_list[i])
        i = 0
        while count > 0:
            self.go()
            i += 1
            print('The {}th time iter end.'.format(i))
            count -= 1

    def genAI(self):
        pass

    def crossChange(self, map1, map2):
        pass

    def mutant(self, mp):
        for i in range(len(mp)):
            for j in range(len(mp[i])):
                se = random.randint(1, 100)
                if se % 7 == 0:  # 发生突变
                    t = random.random() * 2.5 - 5
                    mp[i][j] += t
        return mp

    def go(self):
        Battle = BrooderHouse.BattleSystem(self.map_list)
        rank = Battle.threa()
        print(rank)

    def result(self):
        pass


if __name__ == '__main__':
    # house = BrooderHouse(2)
    # house.go()
    l = [1, 3, 2, 5, 7, 6]
    l = sorted(l, key=lambda x:-x)
    print(l)
