# Arduino Apps Brick Library

Library is composed by configurable and reusable 'bricks', based on optional infrastructure (executed via docker compose) and wrapping python code (to simplify code usage). 

## What is a Brick?

A **brick** is a modular, reusable building block that provides specific functionality for Arduino applications. Each brick is self-contained with standardized configuration, consistent APIs, and optional Docker service definitions.

## Directory Structure

Every brick must follow this standardized directory structure:

```
src/arduino/app_bricks/brick_name/
├── __init__.py                 # Required: Public API exports
├── brick_config.yaml          # Required: Brick metadata
├── brick_compose.yaml         # Optional: Docker services
├── README.md                  # Required: Documentation
├── examples/                  # Required: Usage examples
│   ├── 1_basic_usage.py
│   ├── 2_advanced_usage.py
│   └── ...
├── [implementation_files.py]  # Brick logic
└── [assets]                   # Static resources
```

## Configuration variables

| Variable  | Description |
| ------------- | ------------- |
| APP_HOME  | Base application directory context  |
| LOCAL_DEV | To switch logic for local library development |
| APPSLAB_VERSION | To override the image versions referenced in brick_compose.yaml files |

## Library compile and build 

To build wheel file suitable for release, use following commands:
```sh
pip install build
python -m build .
```
To build package as snapshot for latest development build, use following build command:
```sh
pip install build
python -m build --config-setting "build_type=dev" .
```

## Library development steps
To start the development, clone the repository and create a virtual environment.

Install the Taskfile CLI tool: https://taskfile.dev/installation/.

Then, run the following command to set up the development environment:

```sh
task init
```

This task will check the python version and install the required dependencies.

To force a specific Apps Lab container version, use 'APPSLAB_VERSION' environment variable.

## Linting and formatting

To improve the development experience in VS Code, we recommend adding a `.vscode` folder to the repository root containing the following JSON files:

- `extensions.json`

```json
{
  "recommendations": [
    "charliermarsh.ruff",
    "github.vscode-pull-request-github",
    "ms-python.python",
    "tamasfe.even-better-toml"
  ],
  "unwantedRecommendations": [
    "ms-python.pylint"
  ]
}
```

- `settings.json`

```json
{
    // Set the Python interpreter to the virtual environment
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv",
    "flake8.enabled": false,  // Disable flake8 since we use ruff
    "ruff.enable": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,

    // Linting and fromatting settings on save
    "[python]": {
        // 1) use ruff as the default formatter
        "editor.defaultFormatter": "charliermarsh.ruff",
        
        // 2) automatically format the code on save
        // comment this setting if you don't want to automatically format your code on save
        "editor.formatOnSave": true,

        // 3) apply secure linter fixes on save
        // comment this setting if you don't want to automatically fix with the linter your code on save
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
        }
    }
}
```

After adding those files, VS Code will suggest installing the Python and Ruff extensions, which are properly configured for this project.

Alternatively, you can use the Ruff CLI to safely auto-fix linting issues and format your code by running:

```sh
task lint
```

```sh
task fmt
```

## Testing

All tests must be added in tests/ folder. To execute tests, run command:
```sh
task test
```

or, to execute specific tests, use:
```sh
task test:arduino/app_bricks
```

Modules can use LOCAL_DEV=true env variable to set development specific configurations.

For development purposes, it is possible to change docker registry path using variable:
```sh
DOCKER_REGISTRY_BASE=ghcr.io/arduino/
```
For containers built as part of this library, 'dev-latest' tag is used to point to latest development container.
If it is needed to use a different version, override it via 'APPSLAB_VERSION' env variable.

## Release

Release is based on tag. After tagging main branch with desired version number (e.g. release/1.0.0), build will be triggered.
Build will start:
* docker containers build
* containers upload to public registry with specified tag version
* release of python library as 'wheel' file

Release is split in 2 main actions.
* Tag "ai/{version}" to release AI containers
* Tag "release/{version}" to release bricks library and containers

This is done because release cycle for AI containers and bricks is different. So release is independent.
After releasing a new version of AI containers, compose files that use AI containers must be updated.

To perform development, it is possible to use Dev build pipeline, suited to rebuild a full dev stack with containers updated to 'dev-latest' tag.

### Container layers

Library containers are based on a set of pre-defined python base images that are updated with a different frequency wrt library release.
Base images are built by "BASE - build base images" flow. This flow should be triggered only in case of base images update needs.

Base images are required to:
* reduce the amount of updated layers during a single library update
* promote reuse of existing layers in multiple build
* cache pre-compiled python library as much as possible

Non-base images should start from common base images for performance and disk usage needs.

## License
See [LICENSE](./LICENSE.txt) file for details.

## SBOM (Software Bill of Materials)
Each container includes an SBOM file listing all installed packages, their versions, and licenses:

- `containers/ei-models-runner/sbom.spdx.json`
- `containers/python-apps-base/sbom.spdx.json`

Each SBOM file is generated in SPDX format, which is a standard format for SBOMs.

To generate SBOM files, run:
```sh
task sbom EI_TAG= BRICKS_TAG=
```
where `EI_TAG` and `BRICKS_TAG` represent the versions of the `ei-models-runner` and `python-apps-base` containers, 
respectively.

Example:
```sh
task sbom EI_TAG=1.0.0 BRICKS_TAG=1.0.0
```

**Note**: To run this task, you need to have Docker installed and running on your machine 
and the Docker sbom plugin installed.
