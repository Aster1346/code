import numpy as np

P = np.array([0.2, 0.5, 0.3])
Q = np.array([0.3, 0.4, 0.3])

kl_pq = np.sum(P * np.log(P / Q))
kl_qp = np.sum(Q * np.log(Q / P))
print(kl_pq, kl_qp)