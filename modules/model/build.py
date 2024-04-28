# from .model_add import RPF
from .model import RPF
from .memorybank import MemoryBank

def build_model(cfg):
	return RPF(cfg)

def build_memorybank(cfg):
	return MemoryBank(cfg)
