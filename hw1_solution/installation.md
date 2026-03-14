## Install dependencies

There are two options:

A. (Recommended) Install with conda:

1. Install conda, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

2. Create a conda environment:
	```
	conda create -n cs224r python=3.11
	```
3. Activate the environment (do this every time you open a new terminal):
	```
	conda activate cs224r
	```
4. Install the requirements:
	```
	pip install -r requirements.txt
	```

B. Install on system Python:
	```
	pip install -r requirements.txt
	```

## Troubleshooting

If you encounter display-related errors when rendering videos, try:
```
export SDL_VIDEODRIVER=dummy
```
