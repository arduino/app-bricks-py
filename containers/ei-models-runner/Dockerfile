# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

FROM public.ecr.aws/z9b3d4t5/inference-container-qc-adreno-702:4d7979284677b6bdb557abe8948fa1395dc89a63

# Create the user and group needed to run the container as non-root
RUN set -ex; \
    groupadd -g 44 video || true; \
    groupadd -g 29 audio || true; \
    groupadd -g 991 render || true; \
    groupadd -g 1000 arduino || true; \
    useradd -u 1000 -g arduino -G audio,video,render -ms /bin/bash arduino;

# Add all EI OOTB models to the container with specific ownership
COPY --chown=arduino:arduino ./models/ei-ootb-models/arm64/* /models/ootb/ei/

USER arduino

WORKDIR /app

ENV HOME=/home/arduino
ENV USER=arduino

ENTRYPOINT ["node", "/app/linux/node/build/cli/linux/runner.js"]
