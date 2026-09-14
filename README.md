# Arduino Apps Brick Library

The library is composed of configurable and reusable 'Bricks', based on optional infrastructure (executed via Docker Compose) and wrapping Python® code (to simplify code usage).

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
├── [implementation_files.py]  # Brick logic
└── [assets]                   # Static resources
```

Brick usage examples live in the [app-bricks-examples](https://github.com/arduino/app-bricks-examples) repository, under the `bricks/` folder.

## Configuration variables

| Variable  | Description |
| ------------- | ------------- |
| APP_HOME  | Base application directory context  |
| LOCAL_DEV | To switch logic for local library development |
| BRICKS_RELEASE_VERSION | Version stamped in place of the `__BRICKS_RELEASE_VERSION__` placeholder of compose and models files, defaults to the installed library version |

## Building the wheel

```sh
task build
```

The wheel version is read from `src/arduino/version.py`, which stays at `0.0.0` in the repository: the release workflow injects the tag version into it before building. The same version is stamped in place of the `__BRICKS_RELEASE_VERSION__` placeholder in the compose and models files bundled in the wheel, so they reference the containers published by the same release. To point them at other images, dev images for example, override it:

```sh
BRICKS_RELEASE_VERSION=dev-my-branch task build
```

## Library development steps
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the [Taskfile](https://taskfile.dev/installation/) CLI tool, clone the repository and run:

```sh
task init
```

uv provides Python 3.13, creates `.venv` and installs the library with its development dependencies, exactly the versions pinned in `uv.lock`. Every task runs inside that environment through `uv run`, there is nothing to activate.

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

    // Linting and formatting settings on save
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

For development purposes, it is possible to point to development containers (instead of the released ones) using two variables:
```sh
export DOCKER_REGISTRY_BASE=ghcr.io/<githubuser>/
export DOCKER_PYTHON_BASE_IMAGE=app-bricks/python-apps-base:dev-pose-classification
```
Development containers are published by the dev CI (`docker-build.yml`) tagged as `dev-<branch-name>` (e.g. branch `pose-classification` → tag `dev-pose-classification`).

## Pyright checks

Type checking is driven by `pyright-rules.json` at the repository root, shipped in the wheel as `arduino/app_bricks/static/pyright-rules.json` so that the same rules reach the CI of this repository, the CI of [app-bricks-examples](https://github.com/arduino/app-bricks-examples) and the App Lab editor. The library owns the rules, through two profiles: `app-bricks-py` for its own sources (strict, so the public API carries complete and truthful annotations) and `api-user` for code written against its API (standard, for the published examples and the apps edited in App Lab). The tools own the environment: paths, interpreter, execution root.

Two local checks, both needing the project venv with the current dependencies installed (`pip install -e ".[dev]"`; the checks refuse to run against an outdated environment) and, for the first, a clone of app-bricks-examples next to this repository:

```sh
task check:api      # the examples analyzed against this checkout (profile api-user), then the bricks without examples
task check:typing   # the library sources analyzed against themselves (profile app-bricks-py)
```

Extra arguments go to the underlying `run`/`typing` mode of `scripts/check_pyright.py` (custom paths, JSON output); see `python3 scripts/check_pyright.py --help` for the other modes, including the PR base/head `diff` the workflows use.

On pull requests the `check-pyright.yml` workflow runs both checks against the PR base and head and is informative: it never blocks the merge, since a library change may legitimately require a matching change in the examples and blocking the two repositories on each other would deadlock. The report (new errors introduced by the PR, errors fixed, pre-existing ones collapsed) goes to the job summary and to a sticky comment on the PR, with a label while new errors exist. On PRs from forks the analysis job runs with a read-only token, so the comment is posted by `comment-pyright.yml`, which runs afterwards with a write token and never executes code from the PR. New API errors mean the change breaks the contract the published examples rely on: either adapt the change, or open the matching PR on app-bricks-examples and merge the library first.

## Release

Release is based on tags pushed to `main`. A single workflow (`docker-publish.yml`) publishes **every**
container when a `release/X.Y.Z` tag is pushed, and uploads the Python wheel and the SBOMs to the GitHub
Release. The library and the containers it runs ship together with the same version: the compose files
bundled in the wheel reference the containers published by the same release.

**Prerelease**: if the version contains `rc`, `alpha` or `beta`, images are tagged with the version only
and no `:latest` tag is pushed.

**Dependencies**: base images in `containers/base/` are not released on their own. They are rebuilt first,
in dependency order, as the base of the images that derive from them, and tagged with the same version.

For development, the dev build pipeline (`docker-build.yml`) is triggered manually (`workflow_dispatch`) on a branch and builds the selected containers (or all of them), tagging the images as `dev-<branch-name>`. The selection is widened with the containers deriving from it and with its bases, and `docker buildx bake` builds them in dependency order.

See [`.github/README.md`](.github/README.md) for full CI documentation.

### Container layers

Library containers are based on a set of pre-defined Python base images, in `containers/base/`.
Base images are never released on their own: they are rebuilt as a dependency of the images that derive
from them, and tagged with the release version.

Base images are required to:
* reduce the amount of updated layers during a single library update
* promote reuse of existing layers in multiple builds
* cache pre-compiled python libraries as much as possible

Non-base images should start from common base images for performance and disk usage needs.

## License
See [LICENSE](./LICENSE.txt) file for details.

## Dependencies
Every Python package is declared in a `pyproject.toml` and pinned with hashes in the `uv.lock` next to it. Locks must resolve for the boards (`required-environments`) but install on Windows, macOS and Linux developer machines too; packages missing on some platforms carry an environment marker, like `pyalsaaudio` outside Linux.

The library is described by the root files. `task init` installs it with its development tools into `.venv`, where every task runs through `uv run`. The `python-apps-base` image installs it from the same lock.

Each container that installs Python packages has its own files (see [containers/README.md](containers/README.md#anatomy-of-a-container-directory)) and its Dockerfile installs from the lock alone. `task deps:sync`, run by `task init`, also creates a `.venv` in every container directory to point the IDE at. A container with Python tests declares pytest in a `test` dependency group, kept out of the image, and `task test` runs its suite in that venv. `pyaudio` needs the PortAudio headers on macOS and Linux (`brew install portaudio` or `apt install portaudio19-dev`).

After editing any `pyproject.toml` run `task deps:lock`, with `-- --upgrade` to move to newer versions; `task deps:check` verifies the locks are current and CI runs it on every pull request. Dependabot opens weekly upgrade pull requests, checked by the license scan and the container builds.

## Dependency licenses
`task license:deps` checks the licenses of the Python packages shipped by the library and by every container, using Docker. Records live under `.licenses/`, the allowed licenses and reviewed packages in `.licensed.yml`. See [scripts/licensed/README.md](scripts/licensed/README.md) for how it works and what to do when it fails.

## SBOM (Software Bill of Materials)
SBOMs are not kept in the tree. Each release attaches `sboms.zip` to the GitHub Release, with one folder per published image holding three SPDX documents:

- `base.spdx.json` — packages of the base image the container derives `FROM`, read from the final stage of its Dockerfile
- `full.spdx.json` — complete package list of the container image
- `delta.spdx.json` — packages added by the container on top of its base image

See [containers/README.md](containers/README.md#sboms) for how they are generated. To generate delta SBOMs locally, run:
```sh
task sbom:delta
```
optionally passing container names and the image tag to scan, e.g.:
```sh
task sbom:delta -- python-apps-base --version 1.0.0
```

**Note**: To run this task, you need `syft` installed and access to the container registry.
