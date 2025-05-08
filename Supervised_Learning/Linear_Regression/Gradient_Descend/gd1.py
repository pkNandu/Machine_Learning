import numpy as np
import pandas as pd
def gradient_descend(x, y, lr = 0.01, epoch = 3000):
    b, m = 0.0, 0.0
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_scaled = (x-x_min)/(x_max - x_min)
    y_scaled = (y-y_min)/(y_max - y_min)
    for i in range(epoch):
        y_pred = m * x_scaled + b
        error  = y_scaled - y_pred
        cost = np.mean(error ** 2)
        dm = -2 * np.mean(error * x_scaled)
        db = -2 * np.mean(error)
        b -= db * lr
        m -= dm * lr
        if i%100 == 0:
            print(f"Epoch {i} : Cost = {cost}, b = {b} , m = {m}")
        
    m_original = m * (y_max - y_min) / (x_max - x_min)
    b_original = b * (y_max - y_min) + y_min - m_original * x_min
    return b_original, m_original


if __name__ == '__main__':
    
    df = pd.read_csv("home_prices.csv")
    x = df["area_sqr_ft"].to_numpy()
    y = df["price_lakhs"].to_numpy()
    
    b, m = gradient_descend(x,y)
    print(f"Final Results: m={m}, b={b}")