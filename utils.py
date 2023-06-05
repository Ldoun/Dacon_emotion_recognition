import os
import torch
import random
import numpy as np

def seed_everything(seed: int): #for deterministic result; currently wav2vec2 model and torch.use_deterministic_algorithms is incompatible
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True