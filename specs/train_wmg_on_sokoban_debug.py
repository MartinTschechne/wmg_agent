# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, test_episodes, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_wmg_sokoban.pth

# worker.py
ENV = Sokoban_Env
ENV_RANDOM_SEED = randint  # Use an integer for deterministic training.
AGENT_RANDOM_SEED = 2
REPORTING_INTERVAL = 10
TOTAL_STEPS = 50
ANNEAL_LR = True
ANNEALING_START = 10000000
LR_GAMMA = 0.98

# A3cAgent
AGENT_NET = WMG_Network

# WMG
V2 = True

# Sokoban_Env
SOKOBAN_MAX_STEPS = 120
SOKOBAN_DIFFICULTY = unfiltered
SOKOBAN_SPLIT = train
SOKOBAN_ROOM_OVERRIDE = None
SOKOBAN_BOXES_REQUIRED = 4
SOKOBAN_OBSERVATION_FORMAT = factored

###  HYPERPARAMETERS  (tunable)  ###

# Sokoban_Env
SOKOBAN_REWARD_PER_STEP = 0.
SOKOBAN_REWARD_SUCCESS = 2.

# A3cAgent
A3C_T_MAX = 4
LEARNING_RATE = 1.6e-05
DISCOUNT_FACTOR = 0.995
GRADIENT_CLIP = 512.0
ENTROPY_TERM_STRENGTH = 0.02
ADAM_EPS = 1e-10
REWARD_SCALE = 4.
WEIGHT_DECAY = 0.

# WMG
WMG_MAX_OBS = 0
WMG_MAX_MEMOS = 1
WMG_MEMO_SIZE = 2048
WMG_NUM_LAYERS = 10
WMG_NUM_ATTENTION_HEADS = 8
WMG_ATTENTION_HEAD_SIZE = 32
WMG_HIDDEN_SIZE = 8
AC_HIDDEN_LAYER_SIZE = 2880
WMG_TRANSFORMER_TYPE = NAP
