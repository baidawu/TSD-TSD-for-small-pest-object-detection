import datetime
import matplotlib.pyplot as plt


def plot_loss_and_lr(train_loss, learning_rate):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        fig.savefig('./results/loss_and_lr{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(mAP):
    try:
        x = list(range(len(mAP)))
        plt.plot(x, mAP, label='mAp')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.title('Eval mAP')
        plt.xlim(0, len(mAP))
        plt.legend(loc='best')
        plt.savefig('./results/mAP{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save mAP curve!")
    except Exception as e:
        print(e)

def plot_PR(mAR,mAP):
    try:
        plt.figure()
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('PR cruve')
        plt.plot(mAR, mAP, color="red")
        # plt.show()
        plt.savefig('./results/PR{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save PR curve!")
    except Exception as e:
        print(e)

def plot_loss(train_loss, cls_loss, box_reg_loss, objectness_loss, rpn_box_reg_loss):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Loss")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, cls_loss, label='cls_loss')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        ax3 = ax1.twinx()
        ax3.plot(x, box_reg_loss, label='box_reg_loss')
        ax3.set_ylabel("box_reg_loss")
        ax3.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        ax4 = ax1.twinx()
        ax4.plot(x, objectness_loss, label='objectness_loss')
        ax4.set_ylabel("objectness_loss")
        ax4.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        ax5 = ax1.twinx()
        ax5.plot(x, rpn_box_reg_loss, label='rpn_box_reg_loss')
        ax5.set_ylabel("rpn_box_reg_loss")
        ax5.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles3, labels3 = ax3.get_legend_handles_labels()
        handles4, labels4 = ax4.get_legend_handles_labels()
        handles5, labels5 = ax5.get_legend_handles_labels()
        plt.legend(handles1 + handles2 + handles3 + handles4 + handles5,
                   labels1 + labels2 + labels3 + labels4 + labels5, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        fig.savefig('./results/loss{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save losses curve! ")
    except Exception as e:
        print(e)
