import argparse
import random

def print_list(l):
    row = ""

    for i in l:
        row += f"{i} "

    print(row[:-1])


def generate_matrix(n, d):
    values = []
    col_indx = []
    row_se = [0]

    columns = list(range(n))

    for _ in range(n):
        cols = random.sample(columns, d)
        cols.sort()

        for col in cols:
            col_indx.append(col)
            values.append(random.random() * 2 - 1)

        row_se.append(len(values))

    print(f"{n} {n} {len(values)} {d}")
    print_list(values)
    print_list(row_se)
    print_list(col_indx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('n', help="Matrix dimension")
    parser.add_argument('d', help="Number of nonzero elements in a row")

    args = parser.parse_args()

    generate_matrix(int(args.n), int(args.d))
