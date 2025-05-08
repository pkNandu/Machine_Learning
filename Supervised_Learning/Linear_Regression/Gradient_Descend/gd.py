import numpy as np
def gradient_descend(x, y, lr = 0.01, epoch = 100000):
    b, m = 0.0, 0.0
    for i in range(epoch):
        y_pred = m * x + b
        error  = y - y_pred
        cost = np.mean(error ** 2)
        dm = -2 * np.mean(error * x)
        db = -2 * np.mean(error)
        b -= db * lr
        m -= dm * lr
        print(f"Epoch {i} : Cost = {cost}, b = {b} , m = {m}")


if __name__ == '__main__':
    x = np.array([1,2,3,4,5])
    y = np.array([5,7,9,11,13])
    gradient_descend(x,y)