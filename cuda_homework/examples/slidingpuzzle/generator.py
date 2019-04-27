import argparse
import random

MOVES = (
    (1, 0), (-1, 0), (0, 1), (0, -1)
)

REVERTS = {
    (1, 0): (-1, 0),
    (-1, 0): (1, 0),
    (0, 1): (0, -1),
    (0, -1): (0, 1),
}

class State():
    board = [
        1, 2, 0, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
    ]
    zeroPos = 2

    def __str__(self):
        s = ""
        for i in self.board:
            if i == 0:
                s += "_,"
            else:
                s += f"{i},"
        return s[:-1]

    def inBounds(self, ox, oy):
        x = self.zeroPos % 5
        y = self.zeroPos // 5
        return 0 <= x + ox <= 4 and 0 <= y + oy <= 4

    def swap(self, ox, oy):
        newZeroPos = self.zeroPos + ox + 5 * oy;
        self.board[self.zeroPos], self.board[newZeroPos] = self.board[newZeroPos], self.board[self.zeroPos]
        self.zeroPos = newZeroPos

    def generateState(self, m):
        lastMove = None

        for _ in range(m):
            move = random.choice(MOVES)
            # this still doesn't guarantee anything
            while not st.inBounds(*move) or (lastMove is not None and move == REVERTS[lastMove]):
                move = random.choice(MOVES)

            lastMove = move
            self.swap(*move)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('m', help="Number of moves to the end state")

    args = parser.parse_args()

    st = State()
    print(st)

    st.generateState(int(args.m))
    print(st)