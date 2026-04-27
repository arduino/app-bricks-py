# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

FROM public.ecr.aws/z9b3d4t5/inference-container-qnn:v1.92.19-test-f19ce2e7

# Create the user and group needed to run the container as non-root
RUN set -ex; \
    groupadd -g 1000 arduino || true; \
    useradd -u 1000 -g arduino -ms /bin/bash arduino

# Add all EI OOTB models to the container with specific ownership
COPY --chown=arduino:arduino ./models/ei-ootb-models/arm64/* /models/ootb/ei/
COPY --chown=arduino:arduino ./src/start-runner.sh /home/arduino/start-runner.sh

RUN set -ex; \
    apt-get update && apt-get install -y --no-install-recommends python3-pip && apt-get dist-clean && rm -rf /var/lib/apt/lists/* ; \
    pip3 install --no-cache-dir --break-system-packages requests;\
    chmod +x /home/arduino/start-runner.sh

WORKDIR /home/arduino
ENV HOME=/home/arduino

USER arduino
ENV USER=arduino

ENTRYPOINT ["/home/arduino/start-runner.sh"]
