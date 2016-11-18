import numpy as np
x = [0]
x_new = np.arange(0, 5, 1/9)
x_new = x_new[:45]
x = x + x_new
minus_ones = np.full(45,-1,int)
x = np.multiply(x,minus_ones)
y = np.arange(1, 46, 1)
y = np.flipud(y)
exp_v = np.exp(x)
sum_w = np.sum(exp_v)
weighted_v = np.dot(y,exp_v)
weighted_y = weighted_v/sum_w
print(weighted_y)

