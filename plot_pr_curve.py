import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
import xlrd
import numpy as np

def plot_AP():
    df = xlrd.open_workbook('./results1.0/loss and ap.xls')
    df_sheet = df.sheet_by_name('Sheet1')
    nRows = df_sheet.nrows
    plt.figure()
    plt.title('AP')
    plt.xticks(np.arange(0, 21, step=1))
    plt.yticks(np.arange(0, 0.6, step=0.1))
    plt.xlabel('epoch')
    plt.ylabel('AP')
    colors = ['red', 'green', 'brown', 'blue', 'cyan', 'orange']
    labels = []
    for i in range(6):
        x_data = range(nRows - 1)
        y_data = []
        for j in range(nRows):
            cap = df_sheet.col_values(i)
            if j == 0:
                labels.append(cap[j])
            else:
               y_data.append(cap[j])
        if i == 5:
            plt.plot(x_data, y_data, colors[i], label=labels[i], marker='*', markersize=5)
        else:
            plt.plot(x_data, y_data, colors[i], label=labels[i], marker='o', markersize=4)

    plt.legend(labels)
    plt.savefig('./results1.0/APs.png')
    plt.show()

def plot_loss():
    df = xlrd.open_workbook('./results1.0/loss and ap.xls')
    df_sheet = df.sheet_by_name('Sheet2')
    nRows = df_sheet.nrows
    plt.figure()
    plt.title('AP')
    plt.xticks(np.arange(0, 21, step=1))
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 2, 4, 6, 8])
    plt.xlabel('epoch')
    plt.ylabel('AP@50')
    colors = ['red', 'green', 'brown', 'blue', 'cyan', 'orange']
    labels = []
    for i in range(6):
        x_data = range(nRows - 1)
        y_data = []
        for j in range(nRows):
            cap = df_sheet.col_values(i)
            if j == 0:
                labels.append(cap[j])
            else:
               y_data.append(cap[j])
        plt.plot(x_data, y_data, colors[i], label=labels[i], marker='o', markersize=4)

    plt.legend(labels)
    plt.savefig('./results1.0/APs.png')
    plt.show()

def plot_LOSS():
    df = xlrd.open_workbook('./results1.0/loss and ap.xls')
    df_sheet = df.sheet_by_name('Sheet2')
    nRows = df_sheet.nrows
    colors = ['red', 'green', 'brown', 'blue', 'cyan', 'orange']
    labels = []

    host = host_subplot(111, axes_class=axisartist.Axes)
    # plt.subplots_adjust(right=0.75)
    par1 = host.twinx()
    # par2 = host.twinx()

    # par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(30, 0))
    par1.axis["right"].toggle(all=True)
    # par2.axis["right"].toggle(all=True)


    for i in range(6):
        x_data = range(nRows - 1)
        y_data = []
        for j in range(nRows):
            cap = df_sheet.col_values(i)
            if j == 0:
                labels.append(cap[j])
            else:
                # if i <= 1:
                #     y_data.append(cap[j] / 10)
                # else:
                y_data.append(cap[j])
        if i <= 1:
            p, = par1.plot(x_data, y_data, colors[i], label=labels[i])
        elif i == 5:
            p, = host.plot(x_data, y_data, colors[i], label=labels[i], marker='*', markersize=5)
        # elif i == 1:
        #     p, = par2.plot(x_data, y_data, colors[i], label=labels[i])
        else:
            p, = host.plot(x_data, y_data, colors[i], label=labels[i])

    host.set_xticks(np.arange(0, 21, step=1))
    host.set_yticks(np.arange(0, 1, step=0.1))
    par1.set_yticks(np.arange(0, 8, step=1))
    # par2.set_yticks(np.arange(0, 6, step=1))
    host.set_xlabel('epoch')
    host.set_ylabel('Loss')
    par1.set_ylabel('Loss of YOLO and SSD')
    # par2.set_ylabel('Loss of SSD')

    # host.axis["left"].label.set_color(p1.get_color())
    # par1.axis["right"].label.set_color(colors[1])
    # par2.axis["right"].label.set_color(colors[1])

    # plt.legend(labels)
    host.legend()
    plt.title('Loss')
    plt.plot()
    plt.savefig('./results1.0/LOSSes.png')
    plt.show()

if __name__ == '__main__':
   plot_AP()
   plot_LOSS()
