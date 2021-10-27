import datetime
import matplotlib.pyplot as plt

def CE_loss_PR():

    # CE LOSS
    y1 = [0.1759, 0.2676, 0.3021, 0.3342, 0.3521, 0.3633, 0.3713, 0.3769, 0.3781, 0.3833, 0.3835, 0.3843, 0.389, 0.3868,
          0.3857]  # map
    y2 = []
    x = list(range(len(y1)))
    plt.figure()
    # plt.plot(x1, y1, color="blue")
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(x, y1, 'r', label='mAP(CE LOSS)')
    ax1.set_xlabel("step")
    ax1.set_ylabel("mAP(CE LOSS)")
    ax1.set_title("mAP")
    plt.legend(loc='best')

    ax2 = ax1.twinx()
    ax2.plot(x, y2, label='mAP(Focal LOSS)')
    ax2.set_ylabel("map(Focal Loss)")
    ax2.set_xlim(0, len(y1))  # 设置横坐标整数间隔
    plt.legend(loc='best')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
    # plt.show()
    plt.savefig('./results/CEPR.png')
    plt.close()
    print("successful save PR curve!")

if __name__ == '__main__':
    CE_loss_PR()