import numpy as np

def bp():
    import pdb; pdb.set_trace()

def read_overpotential_time(path, select = 'max'):
    overpotential_data = np.genfromtxt('.\overpotential_file\overpotential_pos_charge_1A_Mohtat2020.csv', delimiter=',')
    if select == 'max':
        selected_row = int(np.where(overpotential_data == np.amax(overpotential_data))[0])
    else:
        selected_row = int(np.where(overpotential_data == np.amin(overpotential_data))[0])

    time_data = np.genfromtxt('.\overpotential_file\\time_step_charge_1A_Mohtat2020.csv', delimiter=',')

    return overpotential_data, time_data, selected_row

if __name__ == '__main__':
    # 生成一个二维数组
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    # 找到最大值和最小值的位置
    max_pos = np.where(arr == np.amax(arr))
    min_pos = np.where(arr == np.amin(arr))
    print(max_pos, min_pos)