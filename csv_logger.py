import os

def log_csv(epoch, dsc, loss):
    # check if directory exists
    if not os.path.exists('log'):
        os.makedirs('log')
    with open('log/log.csv', 'a') as f:
        f.write('{},{},{}\n'.format(epoch, dsc, loss))
