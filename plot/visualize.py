import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import os

def read_csv(csv_file):
    df = pd.read_csv(csv_file, header=None)
    df.columns = ['Epoch', 'DSC', 'Jdet', 'Loss', 'lr']
    return df

def plot(data, metric, save_dir):
    if metric == 'Loss':
        plt.title('Train Loss')
    elif metric == 'DSC':
        plt.title('Val DSC')
    elif metric == 'Jdet':
        plt.title('Val Jdet')
    plt.xlabel('Epoch')
    plt.plot(data, '-')
    plt.savefig(save_dir + metric + '.png')

def compare(normal, ofg):
    plt.title('TransMoprh VS TransMorph_ofg')
    plt.xlabel('Epoch')
    plt.ylabel('Val DSC')
    plt.plot(normal, label='TransMorph')
    plt.plot(ofg, label='TransMorph_ofg')

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig('./imgs/comparison.png')

if __name__ == '__main__':
    ofg = read_csv(csv_file='./logs/vxm_opt_lpba.csv')
    normal = read_csv(csv_file='./logs/vxm_lpba.csv')
    ofg_save_dir = './imgs/vxm_opt_lpba/'
    normal_save_dir = './imgs/vxm_lpba/'
    if not os.path.exists(ofg_save_dir):
      os.makedirs(ofg_save_dir)
    if not os.path.exists(normal_save_dir):
      os.makedirs(normal_save_dir)

    compare(normal['DSC'], ofg['DSC'])
    plt.clf()

    # visualize train loss
    plot(data=ofg['Loss'], metric='Loss', save_dir=ofg_save_dir)
    plt.clf()
    plot(data=normal['Loss'], metric='Loss', save_dir=normal_save_dir)
    plt.clf()

    # visualize val Jdet
    plot(data=ofg['Jdet'], metric='Jdet', save_dir=ofg_save_dir)
    plt.clf()
    plot(data=normal['Jdet'], metric='Jdet', save_dir=normal_save_dir)
    plt.clf()
