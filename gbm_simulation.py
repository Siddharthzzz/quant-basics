import numpy as np
import matplotlib.pyplot as plt

s0 = 100
mu = 0.20
sigma = 0.35
T = 1
N = 252

dt = T / N 

#this line is the timeline 
tp = np.linspace(0,T,N)

#this is for the market randomness
Z=np.random.standard_normal(N)

S = np.zeros(N)

S[0] = s0

for i in range(1, N):
    S[i] = S[i-1] * np.exp(
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z[i]
    )

plt.plot(tp, S)
plt.title("Geometric Brownian Motion Simulation")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.show()

