import numpy as np

class Param:
    def __init__(self, name, init_val, min_val, max_val):
        self.name = name
        self.init_val = init_val
        self.min_val = min_val
        self.max_val = max_val

class Point:
	def __init__(self, vector, no_of_samples, mean):
		self.vector = vector
		self.no_of_samples = no_of_samples
		self.mean = mean