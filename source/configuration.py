'''
Adapted from http://blender.cs.illinois.edu/software/oneie.
'''

import copy
import json
import os

from typing import Dict, Any

class Config(object):
	def __init__(self, **kwargs):
		# dirs
		self.root_dir = kwargs.pop('root_dir', None)

		# devices
		self.use_gpu = kwargs.pop('use_gpu', True)
		self.gpu_device = kwargs.pop('gpu_device', -1)

	@classmethod
	def from_dict(cls, dict_obj):
		"""Creates a Config object from a dictionary.
		Args:
			dict_obj (Dict[str, Any]): a dict where keys are
		"""
		config = cls()
		for k, v in dict_obj.items():
			setattr(config, k, v)
		return config

	@classmethod
	def from_json_file(cls, path):
		with open(path, 'r', encoding='utf-8') as r:
			return cls.from_dict(json.load(r))

	def to_dict(self):
		output = copy.deepcopy(self.__dict__)
		return output

	def save_config(self, path):
		"""Save a configuration object to a file.
		:param path (str): path to the output file or its parent directory.
		"""
		if os.path.isdir(path):
			path = os.path.join(path, 'config.json')
		print('Save config to {}'.format(path))
		with open(path, 'w', encoding='utf-8') as w:
			w.write(json.dumps(self.to_dict(), indent=2,
			                   sort_keys=True))