# import libraries
import shutil
import subprocess
import time
import json
import cv2
import argparse
import random
import subprocess
import pandas as pd
from PIL import Image
import numpy as np
import copy, os, collections
from tqdm import tqdm
from datetime import datetime
## torch
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
## helper functions
from schp_utils import networks
from schp_utils.utils.transforms import transform_logits
from schp_utils.datasets.simple_extractor_dataset import SimpleFolderDataset

# ..... Image parser ..... #
cmd_parse="python3 schp_utils/simple_extractor.py --dataset 'lip' --model-restore schp_utils/checkpoints/final.pth --input-dir ./dataroot/d_test/ --output-dir dataroot/testM_lip"
subprocess.call(cmd_parse, shell=True)

