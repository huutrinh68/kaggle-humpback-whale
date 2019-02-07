# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt  # 导入模块
data = 'car'
fig = 'distance'
# fig = 'loss'
r = np.load("%s.npz" % data) #加载一次即可
print(r)
print(r.files)

plt.figure(figsize=(16, 8))

loss = r['loss']
epoch = r['epoch']

pos = r['pos']
neg = r['neg']

# plt.plot(epoch, loss)
# plt.savefig('product_loss_epoch.jpg')

# plt.show()  # 输出图像
if fig == 'distance':
    plot1 = plt.plot(epoch, neg, label='Negative Dist',  linewidth=3.0)
    plot2 = plt.plot(epoch, pos, label='Positive  Dist', linewidth=3.0)
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    plt.legend(loc='best', numpoints=1)
    plt.savefig('%s_distance.eps' % data, format='eps', dpi=1200)
else:
    plt.plot(epoch, loss,  linewidth=3.0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.legend(loc='best', numpoints=1)
    plt.savefig('%s_%s.eps' % (data, fig), format='eps', dpi=1200)


plt.show()  # 输出图像



