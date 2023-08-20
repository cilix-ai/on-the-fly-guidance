import os

def log_csv(save_dir, epoch, dsc, loss):
    # check if directory exists
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    with open('logs/'+save_dir+'log.csv', 'a') as f:
        f.write('{},{},{}\n'.format(epoch, dsc, loss))
