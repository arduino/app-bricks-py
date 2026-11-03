# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

# The Edge Impulse inference server and its locked packages, as a layer: an image FROM scratch
# holding only those files, never run and never published. edge-impulse-runner and
# edge-impulse-npu-runner add /app with COPY --link and mount /deps while installing the
# packages, see containers/README.md.

FROM scratch

COPY src/ /app/
COPY pyproject.toml uv.lock /deps/
