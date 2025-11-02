import os
from typing import Dict, List, Any, Callable, Union

from evoagentx.benchmark import Benchmark
from evoagentx.utils.utils import download_file
from evoagentx.code.module_utils import load_json
from evoagentx.core.logging import logger
from evoagentx.measures import exact_match_score, f1_score, acc_score



def FeynmanHardBenchmark(Benchmark):
	"""
	Benchmark class for evaluating the self-evolving agent
	on an symbolic regression tasks.
	"""

	def __init__(self, path: str = None, mode: str = "all", **kwargs):
		path = os.path.expanduser(path or "/.evoagentx/data/feynmanhard")
		super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

	def _load_data_from_file(self, file_name: str):
		if file_name is None:
			return None
		file_path = os.path.join(self.path, file_name)
		if not os.path.exists(file_path):
			download_raw_feynman_hard(name=file_name, save_folder=self.path)
		logger.info(f"loading FeynmanHard data from {file_path} ...")
		return load_json(path=file_path, type="json")