import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

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
            
def compare(normal, optron):
    plt.title('TransMoprh VS TransMorph_Optron')
    plt.xlabel('Epoch')
    plt.ylabel('Val DSC')
    plt.plot(normal, label='TransMorph')
    plt.plot(optron, label='TransMorph_Optron')
        
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig('./imgs/comparison.png')
    
if __name__ == '__main__':
    optron = read_csv(csv_file='?')
    normal = read_csv(csv_file='?')
    
    plt.figure(1)
    compare(normal['DSC'], optron['DSC'])
    
    # visualize train loss
    plt.figure(2)
    plot(data=optron['Loss'], metric='Loss', save_dir='./imgs/xxx_optron_xxx/')
    plot(data=normal['Loss'], metric='Loss', save_dir='./imgs/xxx_xxx/')
    
    # visualize val Jdet
    plt.figure(2)
    plot(data=optron['Jdet'], metric='Jdet', save_dir='./imgs/xxx_optron_xxx/')
    plot(data=normal['Jdet'], metric='Jdet', save_dir='./imgs/xxx_xxx/')
