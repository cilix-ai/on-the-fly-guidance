import pandas as pd
import matplotlib.pyplot as plt
import json

def read_json(json_file, mode, select):
    # x = []  # epoch number
    y = []  # evaluation indicator e.g. accuracy, loss

    with open(json_file, 'r') as f:
        for jsonstr in f.readlines():
            row_data = json.loads(jsonstr)
            if row_data['mode'] == mode:
                y_select = float(row_data[select])
                y.append(y_select)
    x = list(range(1, len(y)+1))

    return y, x

# load the csv file into a pandas DataFrame
df = pd.read_csv('path/vvn+oasis.csv', header=None)
df1 = pd.read_csv('path/vvn+oasis+opt.csv', header=None)
# df = read_json('/Users/yuelin_xin/Downloads/optron_data/trm+ixi.json', 'val', 'val_dsc')

# give columns appropriate names
df.columns = ['Epoch', 'DSC', 'JDet', 'Loss', 'LR']
df1.columns = ['Epoch', 'DSC', 'JDet', 'Loss', 'LR']
# df1.columns = ['Epoch', 'DSC', 'Loss']

# plot Loss, DSC as functions of Epoch
plt.figure(figsize=(5, 4))
# print(df['DSC'].max())
print(df1['DSC'].max())
# plt.plot(df['Epoch'], df['Loss'], label='Loss', color='orange')
plt.plot(df['Epoch'][:450], df['DSC'][:450], label='ViT-V-Net Base', linewidth=1)
# plt.plot(df[1][:450], df[0][:450], label='TransMorph Base', linewidth=1)
print(len(df['Epoch']))
plt.plot(df1['Epoch'][:450], df1['DSC'][:450], label='ViT-V-Net with Optron', linewidth=1, color=(0.89, 0.47, 0.18))
# plt.plot(df['Epoch'], df['Loss'], label='Val. DSC', linewidth=1, color='gray')
# plt.plot(df1['Epoch'], df1['Loss'], label='Val. DSC optimized', linewidth=1, color=(0.89, 0.47, 0.18))

plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Validation DSC', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
plt.title('ViT-V-Net on OASIS', fontsize=10)
plt.show()
