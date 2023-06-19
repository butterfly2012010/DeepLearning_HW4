import numpy as np
import matplotlib.pyplot as plt

# train loss for object detection
data = np.loadtxt('train_loss_obj.txt', delimiter=' ')
epoch = data[:, 0]
loss = data[:, 1]
plt.plot(epoch, loss, label='train loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
# plt.grid(True)
plt.legend()
plt.savefig('./img/train_obj_loss_vs_epoch.png')
plt.show()
plt.clf()

# # train acc. for object detection
# data = np.loadtxt('train_acc_obj.txt', delimiter=' ')
# epoch = data[:, 0]
# acc = data[:, 1]
# plt.plot(epoch, acc. label='train acc.')
# plt.xlabel('Epoch')
# plt.ylabel('Acc')
# plt.title('Acc vs. Epoch')
# # plt.grid(True)
# plt.legend()
# plt.savefig('./img/train_obj_acc_vs_epoch.png')
# plt.show()
# plt.clf()

# # test acc. for object detection
# data = np.loadtxt('test_acc_obj.txt', delimiter=' ')
# epoch = data[:, 0]
# acc = data[:, 1]
# plt.plot(epoch, acc, label='test acc.')
# plt.xlabel('Epoch')
# plt.ylabel('Acc')
# plt.title('Acc vs. Epoch')
# # plt.grid(True)
# plt.legend()
# plt.savefig('./img/test_obj_acc_vs_epoch.png')
# plt.show()
# plt.clf()

# train & test acc. for object detection
data = np.loadtxt('train_acc_obj.txt', delimiter=' ')
epoch = data[:, 0]
train_acc = data[:, 1]
plt.plot(epoch, train_acc, label='train acc.')
data = np.loadtxt('test_acc_obj.txt', delimiter=' ')
test_acc = data[:, 1]
plt.plot(epoch, test_acc, label='test acc.')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.title('Acc vs. Epoch')
plt.legend()
plt.savefig('./img/train_test_obj_acc_vs_epoch.png')
plt.show()
plt.clf()


# train loss for semantic segmentation
data = np.loadtxt('train_loss_seg.txt', delimiter=' ')
epoch = data[:, 0]
loss = data[:, 1]
plt.plot(epoch, loss, label='train loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
# plt.grid(True)
plt.legend()
plt.savefig('./img/train_seg_loss_vs_epoch.png')
plt.show()
plt.clf()

# train acc. for semantic segmentation
data = np.loadtxt('train_acc_seg.txt', delimiter=' ')
epoch = data[:, 0]
acc = data[:, 1]
plt.plot(epoch, acc, label='train acc.')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.title('Acc vs. Epoch')
# plt.grid(True)
plt.legend()
plt.savefig('./img/train_seg_acc_vs_epoch.png')
plt.show()
plt.clf()

