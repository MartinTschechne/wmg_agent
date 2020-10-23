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

def plot_all(files):
    for file_path in files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        success_rate = extract_success_rate(lines)
        label = file_path.split('/')[-1][:-4]
        steps = np.linspace(1e3,1e3*len(success_rate),len(success_rate))
        plt.figure()
        plt.plot(steps,success_rate,label=label)
        plt.ylim((0.,1.1))
        plt.xlim((0,8e4))
        plt.hlines(y = 0.99, xmin=0,xmax=2e5,linestyle='--',color='r')
        plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
        plt.xlabel('Steps')
        plt.ylabel('Success Rate')
        plt.title(f"Max. Success Rate: {max(success_rate)}")
        plt.legend(loc='upper left')
        out_path = file_path[:-4] +'.png'
        # plt.savefig(out_path,dpi=100)
        plt.show()

def plot_one(folder_path, model=None):
    if model == None:
        print("Please specify model {Original, NAP}.")
        exit(1)
    files = glob.glob(os.path.join(os.path.abspath(folder_path),f"{model}_*.txt"))
    max_success_rates = []
    max_len = 0
    success_rates = []
    for file_path in files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        success_rate = extract_success_rate(lines)
        max_len = max(max_len, len(success_rate))
        success_rates.append(success_rate)
        max_success_rates.append(max(success_rate))
    label = files[0].split('/')[-1].split('_')[0]

    mean_success_rate = [[],[]]
    for i in range(max_len):
        tmp = [sr[i] if len(sr) > i else sr[-1] for sr in success_rates]
        mean_success_rate[0].append(np.mean(tmp))
        mean_success_rate[1].append(np.std(tmp))

    mean_success_rate = np.array(mean_success_rate)
    steps = np.linspace(1e3,1e3*max_len,max_len)
    plt.plot(steps,mean_success_rate[0],label=label)
    plt.fill_between(steps, mean_success_rate[0]-mean_success_rate[1],
                            mean_success_rate[0]+mean_success_rate[1],
                            color='r',
                            alpha=0.1)
    plt.ylim((0.,1.1))
    plt.xlim((0,2e5))
    plt.hlines(y = 0.99, xmin=0,xmax=2e5,linestyle='--',color='r')
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
    plt.xlabel('Steps')
    plt.ylabel('Success Rate')
    plt.title(f"Mean-Max-Succes-Rate: {np.mean(max_success_rates):.4f} +/- {np.std(max_success_rates):.4f}\n"
                f"Min-Max-SR: {np.min(max_success_rates):.4f}, Max.-Max.-SR: {np.max(max_success_rates):.4f}")
    plt.legend()
    plt.savefig(f"./{model}_array.png",dpi=100)
    plt.show()
    print(f"Mean-Max-Succes-Rate: {np.mean(max_success_rates):.3f} +/- {np.std(max_success_rates):.3f}")

def who_won(folder_path):
    files_org = glob.glob(os.path.join(os.path.abspath(folder_path),'Original_*.txt'))
    files_nap = glob.glob(os.path.join(os.path.abspath(folder_path),'NAP_*.txt'))

    stats = {'Original':0,
            'NAP':0}
    max_org_sr, max_nap_sr = [], []
    for org, nap in zip(files_org, files_nap):
        with open(org, 'r') as f:
            lines = f.readlines()
        max_org_sr.append(max(extract_success_rate(lines)))

        with open(nap, 'r') as f:
            lines = f.readlines()
        max_nap_sr.append(max(extract_success_rate(lines)))

        if max_org_sr[-1] > max_nap_sr[-1]:
            stats['Original'] += 1
        else:
            stats['NAP'] += 1
    for name, score in stats.items():
        print(name, score)
    plt.figure()
    plt.hist(max_org_sr,bins=50,range=(0.,1.),alpha=0.5,color='b',label='Original')
    plt.hist(max_nap_sr,bins=50,range=(0.,1.),alpha=0.5,color='r',label='NAP')
    plt.legend(loc='upper left')
    plt.xlabel('Success Rate')
    plt.savefig('Success-rate-distribution.png',dpi=100)
    plt.show()

def main():
    num_args = len(sys.argv) - 1
    if num_args != 1:
        print('Specify path to output folder.')
        exit(1)

    folder_path = sys.argv[1] # e.g. '../results/201009_moving_layer_norm/'

    ### Plot all learning curves ###
    # plot_all(files)

    ### Plot Average over all Learning Curves ###
    # plot_one(folder_path, "NAP")

    ### Check which model won ###
    who_won(folder_path)


if __name__ == '__main__':
    main()
