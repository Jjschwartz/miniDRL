"""Base configuration for PORL."""
import os
import os.path as osp
from pathlib import Path

PKG_DIR = osp.dirname(osp.abspath(__file__))
REPO_DIR = osp.abspath(osp.join(PKG_DIR, os.pardir))
BASE_RESULTS_DIR = osp.join(str(Path.home()), "d2rl_results")


if not osp.exists(BASE_RESULTS_DIR):
    os.makedirs(BASE_RESULTS_DIR)
