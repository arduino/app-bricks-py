#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# Generate ALSA config for multiple cards upfront

ASOUND_CONF="$HOME/.asoundrc"

MAX_CARDS=10  # Number of cards
MAX_DEVICES=5 # Number of devices per card

cat > "$ASOUND_CONF" << 'EOF'
# Auto-generated ALSA configurations for sound cards 0-9 each with devices 0-4
EOF


for CARD_NUM in $(seq 0 $((MAX_CARDS - 1))); do
  for DEV_NUM in $(seq 0 $((MAX_DEVICES - 1))); do
    cat >> "$ASOUND_CONF" << EOF

# Configuration for card $CARD_NUM, device $DEV_NUM
pcm.card_${CARD_NUM}_${DEV_NUM}_dmix {
    type dmix
    ipc_key $((1024 + CARD_NUM * 10 + DEV_NUM))
    ipc_key_add_uid true
    slave {
        pcm "hw:${CARD_NUM},${DEV_NUM}"
    }
}

pcm.card_${CARD_NUM}_${DEV_NUM}_plug_dmix {
    type plug
    slave.pcm "card_${CARD_NUM}_${DEV_NUM}_dmix"
}

pcm.card_${CARD_NUM}_${DEV_NUM}_spk_wr {
    type softvol
    slave {
        pcm "card_${CARD_NUM}_${DEV_NUM}_plug_dmix"
    }
    control {
        name "card_${CARD_NUM}_${DEV_NUM}_spk_wr"
        card ${CARD_NUM}
    }
}
EOF
  done
done


for CARD_NUM in $(seq 0 $((MAX_CARDS - 1))); do
  for DEV_NUM in $(seq 0 $((MAX_DEVICES - 1))); do
    cat >> "$ASOUND_CONF" << EOF

# Configuration for card $CARD_NUM, device $DEV_NUM
pcm.card_${CARD_NUM}_${DEV_NUM}_dsnoop {
    type dsnoop
    ipc_key $((1124 + CARD_NUM * 10 + DEV_NUM))
    ipc_key_add_uid true
    slave {
        pcm "hw:${CARD_NUM},${DEV_NUM}"
    }
}

pcm.card_${CARD_NUM}_${DEV_NUM}_softvol_dsnoop {
    type softvol
    slave.pcm "card_${CARD_NUM}_${DEV_NUM}_dsnoop"
    control {
        name "card_${CARD_NUM}_${DEV_NUM}_mic_wr"
        card ${CARD_NUM}
    }
}

pcm.card_${CARD_NUM}_${DEV_NUM}_mic_wr {
    type plug
    slave {
        pcm "card_${CARD_NUM}_${DEV_NUM}_softvol_dsnoop"
    }
}
EOF
  done
done