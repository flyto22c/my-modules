import numpy as np

def main():
    # :shape=[5,]\dtype=''\valRange=\
    w = np.array([2.1, 5.1, 7.4, -4.7, -8.5], dtype=np.float32)
    # :shape=[1,]\dtype=''\valRange=\
    b = np.array([3.2], dtype=np.float32)
    M = 10000
    N = 6
    data = np.zeros(shape=(M, N), dtype=np.float32)
    for i in range(M):
        x = np.random.normal(size=(5,))
        # 随机放大1000倍。
        x = x * 10 * np.random.uniform(size=(5,))
        x = x * 10 * np.random.uniform(size=(5,))
        y = np.dot(w, x) + b
        data[i, :-1], data[i, -1:] = x, y
    np.savetxt('zpf_train_data.csv', data[:-100], fmt='%.4f', delimiter=',')
    np.savetxt('zpf_val_data.csv', data[-100:], fmt='%.4f', delimiter=',')


if __name__ == '__main__':
    main()
