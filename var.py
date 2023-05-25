import numpy as np

def train_var_model(data, lag_order):
    n, k = data.shape
    p = lag_order
    
    X = np.zeros((n-p, p*k))
    y = np.zeros((n-p, k))
    
    for i in range(p, n):
        X[i-p] = data[i-p:i].flatten()
        y[i-p] = data[i]
    
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    return beta

def predict_var_model(data, beta, lag_order, future_steps):
    n, k = data.shape
    p = lag_order
    
    predictions = np.zeros((future_steps, k))
    current_data = data[-p:]
    
    for i in range(future_steps):
        prediction = np.dot(current_data.flatten(), beta)
        predictions[i] = prediction
        current_data = np.vstack((current_data[1:], prediction))
    
    return predictions

# 示例用法
# 假设我们有一个2维时间序列数据
data = np.array([[1, 2],
                 [2, 3],
                 [3, 4],
                 [4, 5],
                 [5, 6],
                 [6, 7]])

# 训练VAR模型
lag_order = 2
beta = train_var_model(data, lag_order)

# 预测未来3个时间步
future_steps = 3
predictions = predict_var_model(data, beta, lag_order, future_steps)

print("预测结果：")
print(predictions)
