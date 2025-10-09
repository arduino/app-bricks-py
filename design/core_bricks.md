# Core bricks

The following modules wrap system peripherals and re-expose them to the user without any need for driver and library configuration. These bricks will be available OOTB, without any explicit import.

High priority (mostly MPU-native):
- RPC
- USBCamera (Webcams)
- XOutput (Xorg server)
- AudioInput*
- AudioOutput*
- CSICamera (CSI interface)*
- ScreenOutput (DSI interface)*
- LED (the 2x MPU LEDs)*

Low priority (mostly off-loaded to MCU):
- LED Matrix*
- GPIO*
- Analog I/O -> analogRead, analogReadResolution, analogWrite, analogWriteResolution, analogReference*
- Digital I/O -> pinMode, digitalRead, digitalWrite*

* not yet available or known

## Assumptions

/dev will be mounted inside the container with user capabilities. We'll also need the libraries for interacting with the peripherals (e.g. Alsa or V4L2) inside the container.