#!/usr/bin/python
# -*-coding:UTF-8-*-
"""
Author: PengfeiZhang
DateTime: 2021/4/13/013
Description: LR.py
"""
import numpy as np
def train_data_generator(file_name, batch_size=128, feature_size=6):
    batch_xy = np.zeros(shape=(batch_size, feature_size), dtype=np.float32)
    count = 0
    with open(file_name) as f:
        for line in f:
            list_line = line.strip().split(',')
            if len(list_line) != feature_size:
                continue

            array_line = np.array([float(v.strip()) for v in list_line], dtype=np.float32)
            batch_xy[count] = array_line
            count = count + 1

            if count == batch_size:
                yield batch_xy[:, :-1], batch_xy[:, -1:]
                count = 0
    # File f closed.

def val_data_loader(file_name):
    val_xy = np.loadtxt(file_name, dtype='float32', delimiter=",")
    return val_xy[:, :-1], val_xy[:, -1:]

def compute_loss_mse(batch_x, batch_y, w, b):
    """
    loss = 1/M * {i=1->M | (w^T@x_i + b - y_i)**2}
    """
    loss = np.zeros(shape=(1, 1), dtype=np.float32)
    M = len(batch_x)
    for i in range(M):
        x_i = batch_x[i:i + 1, :].T
        y_i = batch_y[i:i + 1, :]
        loss_i = (w.T @ x_i + b - y_i) ** 2
        loss += loss_i
    # avg  loss ：
    # loss乘以常数(1/2或2)不会影响求导结果，但样本数量(batch_size)不能算常数，因为最后一个批次样本可能不足batch_size。
    return loss / float(M)

def compute_gradient(batch_x, batch_y, w, b):
    """         
                      i=1->M;j=1->N;
    dl_db          =  2/M * {i=1->M | (w^T@x_i + b - y_i) }
    dl_dw^j        =  2/M * {i=1->M | (w^T@x_i + b - y_i) * x_i^j }
    [dl_dw]_N*1    =  2/M * [X^T]_N*M @ [(w^T@x_i + b - y_i)]_M*1
    """
    M = len(batch_x)

    dl_db = np.zeros_like(b, dtype=np.float32)
    dl_dw = np.zeros_like(w, dtype=np.float32)
    dl_dw_temp = np.zeros(shape=(M, 1), dtype=np.float32)
    for i in range(M):
        x_i = batch_x[i:i + 1, :].T
        y_i = batch_y[i:i + 1, :]
        temp = w.T @ x_i + b - y_i
        dl_db += temp
        dl_dw_temp[i, :] = temp[0, 0]
    dl_db = 2. / float(M) * dl_db
    dl_dw = 2. / float(M) * (batch_x.T @ dl_dw_temp)
    return dl_dw, dl_db


def main():
    EPOCH = 5
    M = 128  # batch_size
    N = 5    # feature_size of X
    LR = 5e-4
    w = np.random.normal(size=(5, 1)).astype(np.float32)
    b = np.zeros(shape=(1, 1), dtype=np.float32)

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_data_generator('zpf_train_data.csv', M, N + 1)):
            # step1: 计算损失
            #       GradientTape()：  batch_x     --net--> batcy_y_
            #       GradientTape()：  batch_y & batch_y_  -->   loss
            #       loss = compute_loss_mse(batch_x, batch_y, w, b)
            # step2: 计算梯度
            dl_dw, dl_db = compute_gradient(batch_x, batch_y, w, b)
            # step3: 更新梯度
            w = w - LR * dl_dw
            b = b - LR * dl_db
            # step3 : 观察loss\metrics
            if step % 10 == 0:
                train_loss = compute_loss_mse(batch_x, batch_y, w, b)
                val_x, val_y = val_data_loader('zpf_val_data.csv')
                val_loss = compute_loss_mse(val_x, val_y, w, b)

                print('{0} {1} train_loss:{2:.4f} val_loss:{3:.4f}'
                      .format(epoch, step, train_loss[0, 0], val_loss[0, 0]), w[:, 0], b)
        # ----------------


if __name__ == '__main__':
    main()
