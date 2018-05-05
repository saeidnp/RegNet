import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

use_cuda = torch.cuda.is_available()