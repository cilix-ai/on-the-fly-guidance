import matplotlib.pyplot as plt
import pandas as pd

labels = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
colors = ['orange', 'yellow', 'orchid', 'lightblue', 'salmon', 'green'] # orange yellow salmon
categories = ['VoxelMorph', 'VoxelMorph_OFG', 'ViTVNet', 'ViTVNet_OFG', 'TransMorph', 'TransMorph_OFG']

def get_data(path):
    df = pd.read_csv(path)
    ret = []
    for label in labels:
        v = []
        for index, row in df.iterrows():
            if row['label'] == label:
                v.append(row['trans'])
        ret.append(v)
    # print(ret)
    return ret

data_tsm = get_data('TransMorph_label.csv')
data_tsm_ofg = get_data('TransMorph_opt_label.csv')
data_vvn = get_data('ViTVNet_label.csv')
data_vvn_ofg = get_data('ViTVNet_opt_label.csv')
data_vxm = get_data('VoxelMorph_label.csv')
data_vxm_ofg = get_data('VoxelMorph_opt_label.csv')

data = {}
position = {}

for i, label in enumerate(labels):
    data[label] = []
    data[label].append(data_vxm[i])
    data[label].append(data_vxm_ofg[i])
    data[label].append(data_vvn[i])
    data[label].append(data_vvn_ofg[i])
    data[label].append(data_tsm[i])
    data[label].append(data_tsm_ofg[i])
    position[label] = [i + 1, i + 1.16, i + 1.32, i + 1.48, i + 1.64, i + 1.8]

plt.figure(figsize=(15, 3))

for i, label in enumerate(labels):
    for (d, p, c) in zip(data[label], position[label], colors):
        plt.boxplot(d, positions=[p], widths=0.08, patch_artist=True, 
                    boxprops=dict(facecolor=c, color=c),
                    medianprops=dict(color='black'),
                    whiskerprops=dict(color=c),
                    capprops=dict(color=c),
                    flierprops=dict(marker='o', color=c, alpha=0.5, markersize=3))
        plt.axvline(x=p, color='black', linestyle='--', linewidth=0.3)

# plt.boxplot(data[1], positions=position[1], widths=0.15)

# plt.title('Box')

plt.ylabel('DSC')

plt.xticks(range(1, len(labels) + 1), labels, fontsize='large')
plt.yticks(fontsize='large')

ylines = [0.2, 0.4, 0.6, 0.8]
for yline in ylines:
    plt.axhline(y=yline, color='black', linestyle='--', linewidth=0.3)

legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
plt.legend(legend_handles, categories, loc='lower left', fontsize='small')

plt.savefig('result_vsm.png')

# plt.show()
