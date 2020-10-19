import os, sys
import glob
import matplotlib.pyplot as plt
import numpy as np

def extract_success_rate(lines):
    success_rate = []
    lines = [l for l in lines if 'Success rate' in l]
    for l in lines:
        l = l.split(' ')
        idx = l.index('Success')
        success_rate.append(float(l[idx-1]))
    return success_rate

def main():
    num_args = len(sys.argv) - 1
    if num_args != 1:
        print('Specify path to output folder.')
        exit(1)

    folder_path = sys.argv[1] # '../results/201009_moving_layer_norm/'
    files = glob.glob(os.path.join(os.path.abspath(folder_path),'*.txt'))
    for file_path in files:
        f = open(file_path, 'r')
        lines = f.readlines()
        success_rate = extract_success_rate(lines)
        label = file_path.split('/')[-1][:-4]
        steps = np.linspace(1e3,1e3*len(success_rate),len(success_rate))
        plt.plot(steps,success_rate,label=label)
        plt.ylim((0.,1.1))
        plt.xlim((0,8e4))
        plt.hlines(y = 0.99, xmin=0,xmax=8e4,linestyle='--',color='r')
        plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
        plt.xlabel('Steps')
        plt.ylabel('Success Rate')
        plt.title(f"Max. Success Rate: {max(success_rate)}")
        plt.legend(loc='upper left')
        out_path = file_path[:-4] +'.png'
        print(out_path)
        # plt.savefig(out_path,dpi=100)
        plt.show()

if __name__ == '__main__':
    main()
