import random
import json

# from pprint import pprint

random.seed(13)

HP = {
    "learning_rate": [4e-6, 6.3e-6, 1e-5, 1.6e-5, 2.5e-5, 4e-5, 6.3e-5, 1e-4, 1.6e-4, 2.5e-4, 4e-4],
    "attention_head_size":[8, 12, 16, 24, 32, 48, 64, 90, 128, 180, 256, 360, 512],
    "attention_heads":[1,2,3,4,6,8,10,12,16,20]
}

def generate_new_config(lr, head_size, num_heads):
    config = {
        "learning_rate":lr,
        "attention_head_size":head_size,
        "attention_heads":num_heads
    }
    return config

def main():
    original_config = generate_new_config(lr=6.3e-5, head_size=128, num_heads=2)
    configs = {
        0:original_config
    }

    for i in range(1,50):
        lr = random.choice(HP["learning_rate"])
        head_size = random.choice(HP["attention_head_size"])
        num_heads = random.choice(HP["attention_heads"])
        configs[i] = generate_new_config(lr, head_size, num_heads)

    # to check for duplicate configurations, not the case for random.seed(13)
    # for i in range(50):
    #     for j in range(i):
    #         if configs[i] == configs[j]:
    #             print(i, j)

    with open("../specs/hyperparameter-combinations.json","w") as file:
        json.dump(configs,file)

if __name__ == '__main__':
    main()
