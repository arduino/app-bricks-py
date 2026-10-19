# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

FROM public.ecr.aws/g7a8t7v6/inference-container:v1.92.3@sha256:50c83ccca1b926650e8df28aad59a4ed9f5f89e8996b381b410a7000760daa53

ARG DEBIAN_FRONTEND=noninteractive

# Create the user and group needed to run the container as non-root
RUN set -ex; \
    groupadd -g 1000 arduino || true; \
    useradd -u 1000 -g arduino -ms /bin/bash arduino

# Add all EI OOTB models to the container with specific ownership
COPY --chown=arduino:arduino ./models/ei-ootb-models/arm64/* /models/ootb/ei/
COPY --chown=arduino:arduino ./src/start-runner.sh /home/arduino/start-runner.sh

RUN chmod +x /home/arduino/start-runner.sh

WORKDIR /home/arduino
ENV HOME=/home/arduino

USER arduino
ENV USER=arduino

# V8 compile cache of the runner's JavaScript, filled at build time: loading the runner's
# modules dominates its start on the boards' Cortex cores and the cache cuts that by a fifth.
# Node keeps the cache in a per-uid subdirectory, so it is filled as the user running the runner.
ENV NODE_COMPILE_CACHE=/home/arduino/.cache/node-compile-cache
RUN node /app/linux/node/build/cli/linux/runner.js --help > /dev/null

ENTRYPOINT ["/home/arduino/start-runner.sh"]
