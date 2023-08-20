import pandas as pd
import matplotlib.pyplot as plt

# load the csv file into a pandas DataFrame
df = pd.read_csv('file1.csv', header=None)
# df1 = pd.read_csv('file2.csv', header=None)

# give columns appropriate names
df.columns = ['Epoch', 'DSC', 'Loss']
# df1.columns = ['Epoch', 'DSC', 'Loss']

# plot Loss, DSC as functions of Epoch
plt.figure(figsize=(6, 4))
# plt.plot(df['Epoch'], df['Loss'], label='Loss', color='orange')
plt.plot(df['Epoch'], df['DSC'], label='Val. DSC')
# plt.plot(df1['Epoch'], df1['DSC'], label='Val. DSC')

plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Value', fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
plt.title('Training Progress', fontsize=10)
plt.show()