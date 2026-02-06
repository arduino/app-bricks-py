# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

FROM public.ecr.aws/g7a8t7v6/inference-container-qc-adreno-702:v1.84.1

# Create the user and group needed to run the container as non-root
RUN set -ex; \
    groupadd -g 44 video || true; \
    groupadd -g 29 audio || true; \
    groupadd -g 991 render || true; \
    groupadd -g 1000 arduino || true; \
    useradd -u 1000 -g arduino -G audio,video,render -ms /bin/bash arduino

# Add all EI OOTB models to the container with specific ownership
COPY --chown=arduino:arduino ./models/ei-ootb-models/arm64/* /models/ootb/ei/
COPY --chown=arduino:arduino ./src/start-runner.sh /home/arduino/start-runner.sh

RUN chmod +x /home/arduino/start-runner.sh

WORKDIR /home/arduino
ENV HOME=/home/arduino

USER arduino
ENV USER=arduino

ENTRYPOINT ["/home/arduino/start-runner.sh"]
