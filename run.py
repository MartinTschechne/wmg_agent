# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, sys

# Add the wmg_agent dir to the system path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# No need for argparse. All settings are contained in the spec file.
num_args = len(sys.argv) - 1
if num_args != 2:
    print('run.py accepts a two arguments specifying the runspec and the configuration id.')
    print("Config-Id 0 represents original hyperparameters.")
    exit(1)

# Read the runspec.
from utils.spec_reader import SpecReader
SpecReader(sys.argv[1])

# Execute the runspec.
from utils.worker import Worker
worker = Worker(sys.argv[2])
worker.execute()
