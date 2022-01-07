import numpy as np

''' 
Description: Batch data generator.
Args:
    file_name: 
    batch_size: 
    feature_size: 
Returns:
    batch_x:
    batch_y:
'''
def data_generator(file_name, batch_size=128, feature_size=6):
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
    # File closed.


def data_loader(file_name):

    val_xy = np.loadtxt(file_name, dtype='float32', delimiter=",")
    return val_xy[:, :-1], val_xy[:, -1:]
