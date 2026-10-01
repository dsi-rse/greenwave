# greenwave

## Project Background

GreenWave is a non-profit who train and support regenerative kelp farmers. They’ve developed an app called My Kelp that equips farmers with the tools to track kelp growth, monitor quality, and quantify their farms’ climate benefits. The app calculates a biomass estimation per species based on logged samples. Farmers can use this estimate to forecast harvest volumes and communicate with buyers. However, the biomass estimates are biased in ways that vary by species. For their most widely grown species, sugar kelp, the estimates are consistently higher than what the farmers actually harvested. This undermines the farmers’ confidence in the app and their ability to give buyers accurate forecasts. The goal is to understand what’s driving the bias, whether better data collection could close the gap, and whether the estimation math itself be improved.

## Project Goals
The goal has been to beat GreenWave's current method, which carries the last sample forward as the harvest estimate. The first model found that harvests come in at about two-thirds of what samples suggest, which explains much of the current method's over-prediction. The Bayesian model builds on that by fitting a growth curve to each farm-season, and it has two versions. The baseline uses a broad, fixed prior (global to all farms) and treats each farm-season on its own. On the 25/26 season it does fairly well. Its errors are smaller on average, mainly because it avoids big misses, though the current method is closer on more of the individual farm-seasons. The baseline also gives a forecast range that usually contains the actual harvest, though those ranges stay wide until a season's first harvest. The hierarchy, still a work in progress, aims to close that gap by learning from other farms and past seasons, so forecasts are sharper before any harvest data comes in.


## Usage

### Docker

### Docker & Make

We use `docker` and `make` to run our code. There are three built-in `make` commands:

* `make build-only`: This will build the image only. It is useful for testing and making changes to the Dockerfile.
* `make run-notebooks`: This will run a jupyter server which also mounts the current directory into `\program`.
* `make run-interactive`: This will create a container (with the current directory mounted as `\program`) and loads an interactive session. 

The file `Makefile` contains information about about the specific commands that are run using when calling each `make` statement.

### Developing inside a container with VS Code

If you prefer to develop inside a container with VS Code then do the following steps. Note that this works with both regular scripts as well as jupyter notebooks.

1. Open the repository in VS Code
2. At the bottom right a window may appear that says `Folder contains a Dev Container configuration file...`. If it does, select, `Reopen in Container` and you are done. Otherwise proceed to next step. 
3. Click the blue or green rectangle in the bottom left of VS code (should say something like `><` or `>< WSL`). Options should appear in the top center of your screen. Select `Reopen in Container`.





## Style
We use [`ruff`](https://docs.astral.sh/ruff/) to enforce style standards and grade code quality. This is an automated code checker that looks for specific issues in the code that need to be fixed to make it readable and consistent with common standards. `ruff` is run before each commit via [`pre-commit`](https://pre-commit.com/). If it fails, the commit will be blocked and the user will be shown what needs to be changed.

To check for errors locally, first ensure that `pre-commit` is installed by running `pip install pre-commit` followed by `pre-commit install`. Once installed, check for errors by running:
```
pre-commit run --all-files
```





