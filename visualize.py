import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

def read_csv(csv_file):
    df = pd.read_csv(csv_file, header=None)
    df.columns = ['Epoch', 'DSC', 'Jdet', 'Loss', 'lr']
    return df
    
def plot_dsc(data):
    # plt.figure(figsize=(15, 2.5))
    plt.title('Val DSC')
    plt.xlabel('Epoch')
    plt.plot(data, '-')
    plt.xticks(np.arange(1, 500, 100))
    plt.legend()
    plt.savefig('./imgs/val_dsc.png')
    
def plot_loss(data):
    # plt.figure(figsize=(15, 2.5))
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.plot(data, '-')
    # plt.xticks(np.arange(1, 500, 100))
    plt.savefig('./imgs/train_loss.png')
    
def plot_Jdet(data):
    # plt.figure(figsize=(15, 2.5))
    plt.title('Val Jdet')
    plt.xlabel('Epoch')
    plt.plot(data, '-')
    # plt.xticks(np.arange(1, 500, 100))
    plt.savefig('./imgs/val_Jdet.png')
    
def find_nearest_index(lst, target):
    for i in range(len(lst)):
        if lst[i] >= target:
            break
    return min(range(i+1), key=lambda i: abs(lst[i] - target))
    
def compare(normal, loop):
    # epochs1 = list(range(1, len(normal)+1))
    # epochs2 = list(range(1, len(loop)+1))
    plt.title('TransMoprh VS TransMorph in the loop')
    plt.xlabel('Epoch')
    plt.ylabel('Val DSC')
    plt.plot(normal, label='TransMorph')
    plt.plot(loop, label='TransMorph in the loop')
    
    # given_y = 0.756
    # plt.axhline(y=given_y, color='gray', linestyle='--')
    # plt.annotate(f'y={given_y}', xy=(0, given_y), xytext=(0.1, given_y + 0.02), arrowprops=dict(arrowstyle='->', color='gray'))
    
    # model1_x_estimate = find_nearest_index(normal, given_y)
    # model2_x_estimate = find_nearest_index(loop, given_y)
    # plt.axvline(x=model1_x_estimate, color='blue', linestyle='--' )
    # plt.annotate(f'x={model1_x_estimate}', xy=(model1_x_estimate, 0.45), xytext=(model1_x_estimate - 40, 0.47),
    #             arrowprops=dict(arrowstyle='->', color='blue'))  
    # plt.axvline(x=model2_x_estimate, color='orange', linestyle='--')
    # plt.annotate(f'x={model2_x_estimate}', xy=(model2_x_estimate, 0.45), xytext=(model2_x_estimate - 40, 0.47),
    #             arrowprops=dict(arrowstyle='->', color='orange'))  
    # x_ticks = list(range(1, 31, 2))
    # x_ticks.sort()
    # plt.xticks(x_ticks)
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig('./imgs/comparison.png')
    
if __name__ == '__main__':
    loop = read_csv(csv_file='?')
    normal = read_csv(csv_file='?')
    
    plt.figure(1)
    compare(normal['DSC'], loop['DSC'])
    
    # plt.figure(2)
    # plot_loss(loop['Loss'])
    # plot_loss(normal['Loss'])
    
    # plt.figure(3)
    # plot_Jdet(loop['Jdet'])
    # plot_Jdet(normal['Jdet'])
