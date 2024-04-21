import json
import numpy as np
import argparse
import os
import pickle

if __name__ == '__main__':
    print("-->>-->>-- Splitting dataset! --<<--<<--")
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--', type=int, default=0, help='')
    parser.add_argument('--use-gpt-dataset', type=bool, default=True, help='')

    args = parser.parse_args()