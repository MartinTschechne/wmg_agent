#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, sys, random, time
import json

# Add the wmg_agent dir to the system path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# CODE_DIR = os.path.dirname(os.path.abspath(__file__))
spec_rand = random.Random(time.time())  # For 'truly' random hyperparameter selection.

# No need for argparse. All settings are contained in the spec file.
num_args = len(sys.argv) - 1
if num_args != 1:
    print('run.py accepts a one argument specifying the path to the runspec.')
    # print("Config-Id 0 represents original hyperparameters.")
    exit(1)

# Read the runspec.
# from utils.spec_reader import SpecReader
# SpecReader(sys.argv[1])
full_path = os.path.join(sys.argv[1])
with open(full_path, 'r') as f:
    spec = json.load(f)
if spec["ENV_RANDOM_SEED"] == "randint":
    spec["ENV_RANDOM_SEED"] = spec_rand.randint(0,999999999)

# Execute the runspec.
from utils.worker import Worker
worker = Worker(spec)
worker.execute()
