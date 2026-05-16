import numpy as np
import random

def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)