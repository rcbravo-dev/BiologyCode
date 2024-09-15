CWD = f'/Users/Shared/SharedProjects/Projects/BiologyCode/Replicators/'

import argparse


# Define a function to validate log level
def valid_log_level(level):
    levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level.upper() not in levels:
        raise argparse.ArgumentTypeError(f"Invalid log level: {level}. Choose from {levels}.")
    return level.upper()


def setup_logging(path: str = 'configs/logging_config.yaml') -> None:
	import logging as log
	import logging.config
	import yaml

	default_path = CWD + path

	with open(default_path, 'rt') as file:
		cfg = yaml.safe_load(file.read())
	log.config.dictConfig(cfg)
	
	# print(f'Logging configured')

	return log


def import_util(cwd: str = CWD, paths: list = ['src/lib']) -> None:
	'''Function to ensure import paths are properly updated. '''
	import sys

	import_paths = ['..'] + [cwd + p for p in paths]

	for i, path in enumerate(import_paths):
		if path not in sys.path: 
			sys.path.insert(i, path)


def convert_notebook(notebook_path: str, script_path: str, code_type: str = 'code', tag: str = 'convert_to_py') -> str:
	'''Cell metadata needs to be tagged with "convert_to_py" or "convert_to_md" prior to running script.'''
	import json
	
	with open(notebook_path, 'r') as f:
		notebook = json.load(f)
	
	# Extract code cells
	notebook_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == code_type]
	
	notebook_cells = [cell for cell in notebook_cells if 'tags' in cell['metadata']]
	
	# Extract source
	code = [cell['source'] for cell in notebook_cells if tag in cell['metadata']['tags']]
	lines = len(code[0])
	
	# Flatten list and join code lines
	if code_type == 'code':
		code = '\n'.join([''.join(cell) for cell in code])
	elif code_type == 'markdown':
		code = '\n\n'.join([''.join(cell) for cell in code])
	
	# Write to file
	with open(script_path, 'w+') as f:
		f.write(code)
	
	# Output
	if len(code) == 0:
		print('No Data written to "script_path"')
		print(f'Code cells parsed = {len(notebook_cells)}')
		print(f'Tagged cells parsed = {len(notebook_cells)}\nLength of source code = {len(code)}\n')
		print(f'Notebook converter wrote {len(code)} to {script_path}')
		return notebook
	else:
		split_path = script_path.split('/')
		split_path = '/'.join(split_path[-3:])
		print(f'Converter successfully wrote {lines} lines to {split_path}')
		

def notebook2python(filename: str, notebook: str = 'src/', script: str = 'src/lib/', markdown: str = 'src/markdown/') -> None:
	import pathlib
	
	notebook_path = CWD + f"{notebook}{filename}.ipynb"
	script_path = CWD + f"{script}{filename}.py"
	# markdown_path = CWD + f"{markdown}{filename}.md"

	try:
		if pathlib.Path(notebook_path).is_file():
			test_passed = True
		else:
			raise FileNotFoundError(f'File not found. Check "{filename}" variable.')

	except FileNotFoundError as err:
		print(err)
		test_passed = False
		
	finally:
		if test_passed:
			convert_notebook(notebook_path, script_path, code_type='code', tag='convert_to_py')
			# convert_notebook(notebook_path, markdown_path, code_type='markdown', tag='convert_to_md')

