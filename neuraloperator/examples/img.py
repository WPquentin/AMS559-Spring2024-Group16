import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 生成均值为0，标准差为1的100维高斯噪声
mean = np.zeros(100)
stddev = np.ones(100)
noise = np.random.normal(mean, stddev)

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)

ax.xaxis.set_major_locator(ticker.AutoLocator())
ax.yaxis.set_major_locator(ticker.AutoLocator())

plt.show()

test_samples = [{'x':np.sin(x), 'y':np.sin(x), 'out':np.sin(x)}, {'x':np.sin(x), 'y':np.sin(x), 'out':np.sin(x)}, {'x':np.sin(x), 'y':np.sin(x), 'out':np.sin(x)}]
fig = plt.figure(figsize=(6, 6))
for index in range(3):
    data = test_samples[index]
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = data['out']

    coordinate = np.linspace(0, 10, 100)

    ax = fig.add_subplot(3, 4, index*4 + 1)
    ax.plot(coordinate, x)
    if index == 0: 
        ax.set_title('Input x')
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())

    ax = fig.add_subplot(3, 4, index*4 + 2)
    ax.plot(coordinate, y, label = 'Ground-truth y')
    ax.plot(coordinate, out, label = 'Model prediction')
    if index == 0: 
        ax.set_title('GT and Prediction')
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())

    ax = fig.add_subplot(3, 4, index*4 + 3)
    ax.plot(coordinate, out)
    if index == 0: 
        ax.set_title('Prediction')
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())

    ax = fig.add_subplot(3, 4, index*4 + 4)
    ax.plot(coordinate, )
    if index == 0: 
        ax.set_title('abs(Difference)')
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    
fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
plt.show()


