import os, sys
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import json

def extract_success_rate(lines):
    '''Read success rate from output file.'''
    success_rate = []
    lines = [l for l in lines if 'Success rate' in l]
    for l in lines:
        l = l.split(' ')
        idx = l.index('Success')
        success_rate.append(float(l[idx-1]))
    return success_rate

def plot_single(folder_path):
    files = glob.glob(os.path.join(os.path.abspath(folder_path),'_*.txt'))
    for file_path in files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        success_rate = extract_success_rate(lines)
        label = file_path.split('/')[-1][:-4]
        steps = np.linspace(1e3,1e3*len(success_rate),len(success_rate))
        plt.plot(steps,success_rate,label=label)
        plt.ylim((0.,1.1))
        plt.xlim((0,2e5))
        plt.hlines(y = 0.99, xmin=0,xmax=2e5,linestyle='--',color='r')
        plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
        plt.xlabel('Steps')
        plt.ylabel('Success Rate')
        # plt.title(f"Max. Success Rate: {max(success_rate)}")
        plt.legend(loc='upper left')
        # out_path = file_path[:-4] +'.png'
        # plt.savefig('./results/name.png',dpi=100)
        # plt.show()

def plot_all_in_one(folder_path, model=None):
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
    # plt.savefig(f"./{model}_array.png",dpi=100)
    plt.show()
    print(f"Mean-Max-Succes-Rate: {np.mean(max_success_rates):.3f} +/- {np.std(max_success_rates):.3f}")
    print(f"Median-Max-Success-Rate: {np.median(max_success_rates):.3f}")

def who_won(folder_path):
    files_org = glob.glob(os.path.join(os.path.abspath(folder_path),'201022_array_job_results/Original_*.txt'))
    files_nap = glob.glob(os.path.join(os.path.abspath(folder_path),'NormalizedOriginal_*.txt'))

    stats = {'Original':0,
            'NormalizedOriginal':0}
    max_org_sr, max_nap_sr = [], []
    for org, nap in zip(files_org, files_nap):
        with open(org, 'r') as f:
            lines = f.readlines()
        max_org_sr.append(max(extract_success_rate(lines)))

        with open(nap, 'r') as f:
            lines = f.readlines()
        max_nap_sr.append(max(extract_success_rate(lines)))
        if max_org_sr[-1] < 0.99 or max_nap_sr[-1] < 0.99: # else considered a tie
            if max_org_sr[-1] > max_nap_sr[-1]:
                stats['Original'] += 1
            else:
                stats['NormalizedOriginal'] += 1

    for name, score in stats.items():
        print(name, score)
    plt.figure()
    plt.hist(max_org_sr,bins=50,range=(0.,1.),alpha=0.5,color='b',label='Original')
    plt.hist(max_nap_sr,bins=50,range=(0.,1.),alpha=0.5,color='r',label='NormalizedOriginal')
    plt.legend(loc='upper left')
    plt.xlabel('Success Rate')
    plt.savefig('Success-rate-distribution-no-vs-orig.png',dpi=100)
    plt.show()

def correlation(path, model):
    with open('./specs/hyperparameter-combinations.json','r') as f:
        params = json.load(f)
    files = glob.glob(os.path.join(os.path.abspath(path),f"{model}_*.txt"))

    max_sr, lr, model_dim = list(), list(), list()
    for file_path in files:
        idx = file_path.split('/')[-1].split('_')[1]
        lr.append(params[idx]['learning_rate'])
        model_dim.append(params[idx]['attention_head_size']*params[idx]['attention_heads'])
        with open(file_path, 'r') as f:
            lines = f.readlines()
        max_sr.append(max(extract_success_rate(lines)))

    plt.set_cmap('coolwarm')
    fig, ax = plt.subplots(2)
    ax[0].set_title(f"{model}")
    ax0 = ax[0].scatter(np.log(lr), max_sr, c=np.log(model_dim))
    x = np.linspace(np.min(np.log(lr)), np.max(np.log(lr)))
    m, b, r_val, p_val, _ = stats.linregress(np.log(lr), max_sr)
    ax[0].plot(x,b+m*x,c='black', label=f"Corr.-Coef.: {r_val:.3f}")
    ax[0].set_xlabel('log Learning Rate')
    ax[0].set_ylabel('Success Rate')
    ax[0].legend(loc='lower left')
    plt.colorbar(ax0, ax=ax[0])

    ax1 = ax[1].scatter(np.log(model_dim), max_sr, c=np.log(lr))
    x = np.linspace(np.min(np.log(model_dim)), np.max(np.log(model_dim)))
    m, b, r_val, p_val, _ = stats.linregress(np.log(model_dim), max_sr)
    ax[1].plot(x,b+m*x,c='black', label=f"Corr.-Coef.: {r_val:.3f}")
    ax[1].set_xlabel('log Model Dimension')
    ax[1].legend(loc='lower left')
    plt.colorbar(ax1, ax=ax[1])

    plt.tight_layout()
    plt.savefig(f"{model}-correlation.png",dpi=100)
    plt.show()

def correlation2(path, model):
    with open('./specs/hyperparameter-combinations.json','r') as f:
        params = json.load(f)
    files = glob.glob(os.path.join(os.path.abspath(path),f"{model}_*.txt"))

    max_sr, lr, model_dim = list(), list(), list()
    for file_path in files:
        idx = file_path.split('/')[-1].split('_')[1]
        with open(file_path, 'r') as f:
            lines = f.readlines()
        if "overall reward per step" in lines[-1]:
            lr.append(params[idx]['learning_rate'])
            max_sr.append(max(extract_success_rate(lines)))
            model_dim.append(params[idx]['attention_head_size']*params[idx]['attention_heads'])
        else:
            print(idx)

    plt.set_cmap('coolwarm')
    fig, ax = plt.subplots()
    ax.set_title(f"{model}")
    ax0 = ax.scatter(np.log(model_dim), np.log(lr), c=max_sr,vmin=0.5,vmax=1.)
    ax.set_ylabel('log Learning Rate')
    ax.set_xlabel('log Model Dimension')
    plt.colorbar(ax0, ax=ax)

    plt.tight_layout()
    # plt.savefig(f"./plots/{model}-correlation2.png",dpi=100)
    plt.show()

def correlation_size(path, model):
    with open('./specs/hyperparameter-combinations.json','r') as f:
        params = json.load(f)
    files = glob.glob(os.path.join(os.path.abspath(path),f"{model}_*.txt"))

    max_sr, lr, head_size, num_heads = list(), list(), list(), list()
    for file_path in files:
        idx = file_path.split('/')[-1].split('_')[1]
        with open(file_path, 'r') as f:
            lines = f.readlines()
        if "overall reward per step" in lines[-1]:
            lr.append(params[idx]['learning_rate'])
            max_sr.append(max(extract_success_rate(lines)))
            head_size.append(params[idx]['attention_head_size'])
            num_heads.append(params[idx]['attention_heads']*10)
        else:
            print(idx)

    plt.set_cmap('coolwarm')
    fig, ax = plt.subplots()
    ax.set_title(f"{model}")
    ax0 = ax.scatter(np.log(lr), np.log(head_size), c=max_sr,vmin=.5,vmax=1.,s=num_heads)
    ax.set_ylabel('log Head Size')
    ax.set_xlabel('log LR')
    plt.colorbar(ax0, ax=ax)

    plt.tight_layout()
    # plt.savefig(f"./plots/{model}-correlation2.png",dpi=100)
    plt.show()


def main():
    num_args = len(sys.argv) - 1
    if num_args != 1:
        print('Specify path to output folder.')
        exit(1)

    folder_path = sys.argv[1] # e.g. '../results/201009_moving_layer_norm/'

    ### Plot single learning curve ###
    # plot_single(folder_path)

    ### Plot Average over all Learning Curves ###
    # plot_all_in_one(folder_path, "Original")

    ### Check which model won ###
    # who_won(folder_path)

    ### correlation
    correlation_size(folder_path, "NormalizedOriginal")


if __name__ == '__main__':
    main()
