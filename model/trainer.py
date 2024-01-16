from collections import defaultdict
from model.utils import CfgNode as CN

class Trainer:
    def get_default_config():
        C=CN()
        C.learning_rate=1e-4
        C.weight_decay=0.1
        
    def __init__(self) -> None:
        pass