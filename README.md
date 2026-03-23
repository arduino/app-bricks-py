# Arduino Apps Brick Library

Library is composed by configurable and reusable 'Bricks', based on optional infrastructure (executed via Docker Compose) and wrapping Python® code (to simplify code usage). 

## What is a Brick?

A **Brick** is a modular, reusable building block that provides specific functionality for Arduino applications. Each Brick is self-contained with standardized configuration, consistent APIs, and optional Docker service definitions.

## Directory Structure

Every Brick must follow this standardized directory structure:

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

To force a specific Arduino App Lab container version, use 'APPSLAB_VERSION' environment variable.

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

Release is based on tags pushed to `main`. A single workflow (`docker-github-publish.yml`) handles all container releases and detects which container to build from the tag prefix defined in each container's `ci.json`.

| Tag | What it releases |
|---|---|
| `base/X.Y.Z` | `python-base` base image |
| `release/X.Y.Z` | `python-apps-base` container + Python `.whl` uploaded to GitHub Release |
| `ai/X.Y.Z` | `ei-models-runner` AI container |

Release cycles for AI containers and Bricks are independent — they use separate tag prefixes and can be released at any time without affecting each other.

After releasing a new version of AI containers, compose files that use AI containers are updated automatically via a generated PR.

**Downstream cascade**: when `python-base` is released, the workflow automatically triggers a rebuild of `python-apps-base` (and any other container declared as a downstream dependency). No manual step required.

For development, the dev build pipeline (`docker-github-build.yml`) rebuilds only the containers whose source files changed on the branch. Dependent containers are built in the correct order — downstream containers wait for their upstream to finish and use the freshly built image.

See [`.github/README.md`](.github/README.md) for full CI documentation.

### Container layers

Library containers are based on a set of pre-defined Python base images that are updated with a different frequency wrt library release.
Base images are built by tagging `base/X.Y.Z`. This should be done only when base image dependencies or infrastructure change.

Base images are required to:
* reduce the amount of updated layers during a single library update
* promote reuse of existing layers in multiple builds
* cache pre-compiled python libraries as much as possible

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
