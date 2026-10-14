# Source patches

Patches to the Python packages python-apps-base installs, applied by its Dockerfile right after the install, in
name order. This README keeps the directory in the build context when it holds no patch.

- One `*.patch` per change, in `git format-patch` format, with paths relative to `site-packages` (`a/fastapi/...`):
  the Dockerfile applies it with `patch -p1 --forward --fuzz=0` to `/usr/local/lib/python3.13/site-packages`.
- Made against the version in `uv.lock`. A patch that stops applying exactly, after a bump of its package or once
  upstream has merged it, fails the build and `tests/containers/test_python_apps_base_patches.py`: regenerate it
  against the new version, or delete it.
- Keep the change upstreamable and say where it comes from in the commit message of the patch.
