import numpy as npy
import random
import time
import threading
import inspect
import ctypes

TIMEOUT = 4.8
COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
ini_map = npy.array([
    [-200, 17, -20, -15, -15, -20, 17, -200],
    [17, 20, 10, 10, 10, 10, 20, 17],
    [-20, 10, -5, -4, -4, -5, 10, -20],
    [-15, 10, -4, 1, 1, -4, 10, -15],
    [-15, 10, -4, 1, 1, -4, 10, -15],
    [-20, 10, -5, -4, -4, -5, 10, -20],
    [17, 20, 10, 10, 10, 10, 20, 17],
    [-200, 17, -20, -15, -15, -20, 17, -200]
])


class STATE(enumerate):
    START = 1,
    MID = 2,
    ENDGAME = 3


# don't change the class name
class AI(object):
    direction = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [0, -1], [1, -1], [1, 0], [1, 1]]

    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out, a: list = ini_map):
        self.state = STATE.START
        self.a = a

        self.cnt = 0
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need add your decision into your candidate_list. System will get the end of your candidate_list as your
        # decision.
        self.candidate_list = []
        # The max depth for the agent to search.
        self.depth = 6
        # The max width for the agent to search.
        self.width = 7

    # The input is current chessboard.
    def go(self, chessboard):
        start = time.time()
        self.candidate_list.clear()
        temp = self.genValidPos(chessboard, self.color)

        if len(temp) == 0:
            return

        self.candidate_list = temp
        # t1_stop = threading.Event()
        # t1 = threading.Thread(target=self.getAction, args=(chessboard, t1_stop))
        # t1.setDaemon(True)
        # t1.start()
        # time.sleep(self.time_out - time.time() + start - 0.7)
        # t1_stop.set()

        self.candidate_list.append(self.getAction(chessboard))

        self.cnt += 1
        # if self.candidate_list[-1] not in temp:
        #     self.candidate_list = temp
        #     self.candidate_list.append(random.choice(temp))
        run_time = (time.time() - start)
        # print(run_time)

    def set_color(self, c):
        self.color = c

    def genValidPos(self, chessboard, color):
        idx = npy.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        res = []
        for pos in idx:
            for d in AI.direction:
                if self.checkOkPosition(pos, d, chessboard, color):
                    res.append(pos)
                    break
        return res

    def checkInBoard(self, pos) -> bool:
        return 0 <= pos[0] < self.chessboard_size and 0 <= pos[1] < self.chessboard_size

    def checkOkPosition(self, pos, d, chessboard, color) -> bool:
        p = pos[0]
        q = pos[1]
        if self.checkInBoard((p + d[0], q + d[1])):
            if chessboard[p + d[0]][q + d[1]] != color * -1:
                return False
        else:
            return False

        while self.checkInBoard((p + d[0], q + d[1])) and chessboard[p + d[0]][q + d[1]] == color * -1:
            p += d[0]
            q += d[1]
        return self.checkInBoard((p + d[0], q + d[1])) and chessboard[p + d[0]][q + d[1]] == color

    # Get the best action for the agent.
    def getAction(self, chessboard):
        nowPlayer = self.color
        if 0 <= self.cnt <= 5:
            self.state = STATE.START
        elif 5 < self.cnt <= 40:
            self.state = STATE.MID
        else:
            self.state = STATE.ENDGAME
        _, bestAction = self._getMax(chessboard)
        return bestAction

    def _getMax(self, chessboard, depth=0, alpha=-float('inf'), beta=float('inf')):
        if depth == self.depth:
            return self.score(chessboard), None
        valid_move = self.genValidPos(chessboard, self.color)
        if len(valid_move) == 0:
            return self.score(chessboard), None
        v, move = -float('inf'), None
        if len(valid_move) > self.width:
            tmp = []
            for m in valid_move:
                tmp.append((m, self.score(self.action(chessboard, m, self.color))))
            tmp = sorted(tmp, key=lambda x: -x[1])
            valid_move.clear()
            for i in range(0, self.width):
                valid_move.append(tmp[i][0])

        for a in valid_move:
            v2, _ = self._getMin(self.action(chessboard, a, self.color), depth + 1, alpha, beta)
            if v2 > v:
                v = v2
                move = a
                alpha = max(alpha, v)
            if v >= beta:
                return v, a
        return v, move

    def _getMin(self, chessboard, depth=0, alpha=-float('inf'), beta=float('inf')):
        if depth == self.depth:
            return self.score(chessboard), None
        valid_move = self.genValidPos(chessboard, self.color * -1)
        if len(valid_move) == 0:
            return self.score(chessboard), None
        v, move = float('inf'), None
        if len(valid_move) > self.width:
            tmp = []
            for m in valid_move:
                tmp.append((m, self.score(self.action(chessboard, m, self.color))))
            tmp = sorted(tmp, key=lambda x: -x[1])
            valid_move.clear()
            for i in range(0, self.width):
                valid_move.append(tmp[i][0])
        for a in valid_move:
            v2, _ = self._getMax(self.action(chessboard, a, self.color * -1), depth + 1, alpha, beta)
            if v2 < v:
                v = v2
                move = a
                beta = min(beta, v)
            if v <= alpha:
                return v2, a
        return v, move

    def action(self, chessboard, move: tuple, color):
        p = move[0]
        q = move[1]
        temp = chessboard.copy()
        temp[p][q] = color
        t = 0
        for d in AI.direction:
            while self.checkInBoard((p + d[0], q + d[1])) and temp[p + d[0]][q + d[1]] == color * -1:
                p += d[0]
                q += d[1]
                t += 1
            if self.checkInBoard((p + d[0], q + d[1])) and temp[p + d[0]][q + d[1]] == color:
                while t > 0 and self.checkInBoard((p, q)):
                    temp[p][q] = color
                    p -= d[0]
                    q -= d[1]
                    t -= 1
        return temp

    def score(self, chessboard):
        if self.state == STATE.ENDGAME:
            foo, bar = self.isTerminal(chessboard)
            if foo:
                return 10000 if bar else -10000

        return sum(sum(npy.multiply(chessboard, self.a))) * self.color

    # check the game is end.
    def isTerminal(self, chessboard):
        if len(npy.where(chessboard == 0)[0]) == 0 or \
                (len(self.genValidPos(chessboard, COLOR_BLACK)) == 0 and
                 len(self.genValidPos(chessboard, COLOR_WHITE == 0))):
            my = len(npy.where(chessboard == self.color)[0])
            opp = len(npy.where(chessboard == self.color * -1)[0])
            return True, my > opp

        else:
            return False, None

# ==============Find new pos========================================
# Make sure that the position of your decision in chess board is empty.
# If not, the system will return error.
# Add your decision into candidate_list, Records the chess board
# You need add all the positions which is valid
# candidate_list example: [(3,3),(4,4)]
# You need append your decision at the end of the candidate_list,
# we will choice the last element of the candidate_list as the position you choose
# If there is no valid position, you must return an empty list.
