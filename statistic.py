import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i+1)].sum() / (i - start + 1))
    return y

df=pd.read_excel("./Excel_test1/Adventure.xlsx")
x = np.array(df["NumOfState"])
y = smooth(x)
plt.title("Number of States Comparison of RiverRaid")
plt.plot(x,label='Orig')
plt.plot(y,label='Smoothed')
plt.ylabel('Number of States')
plt.xlabel('Episode')
plt.legend()
plt.show()
"""x=np.array(df["Global_Step"])
y=smooth(rewards)
df2=pd.read_excel("./Excel_CNN_LSTM_batch_20/Hybrid_RiverRaid_lstm.xlsx")
rewards2=np.array(df2["Rewards"])
x2=np.array(df2["Global_Step"])
y2=smooth(rewards2)

plt.plot(x,y, label='CNN')
plt.plot(x2,y2,label="CNN and LSTM")
plt.xlabel('Global Steps')
# naming the y axis
plt.ylabel('score')
plt.title("River Raid")
plt.legend()
plt.show()"""
"""words = ["This ", "is ", "a ", "test"]
with open("Model/example.txt", "w") as f:
    #for word in words:
        f.write(str(words))"""