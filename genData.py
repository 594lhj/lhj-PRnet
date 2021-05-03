import numpy as np


def main():
    N = 256 # 子载波数
    mu = 4

    train = np.random.randint(0, 2, 5000 * N * mu)

    val = np.random.randint(0, 2, 400 * N * mu)
    test = np.random.randint(0, 2, 1000* N * mu)

    np.savez('./data/train.npz', data=train)
    np.savez('./data/val.npz', data=val)
    np.savez('./data/test.npz', data=test)


if __name__ == "__main__":
    main()
