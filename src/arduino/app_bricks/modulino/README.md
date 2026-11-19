# Modulino brick

## Modulino Hub

A Modulino Hub is an I2C multiplexer with 8 ports (0-7). It lets you use several Modulini that have the same I2C address, for example two Buttons with the default address, by plugging them into different ports.

Create a `ModulinoHub` with its address, take a port with `port(n)` and pass it to the Modulino constructor as `hub_port`:

```python
from arduino.app_bricks.modulino import ModulinoHub, Modulino_Buttons, Modulino_Thermo

hub = ModulinoHub(0x70)

buttons_a = Modulino_Buttons(hub_port=hub.port(0))
buttons_b = Modulino_Buttons(hub_port=hub.port(3))
thermo = Modulino_Thermo()   # main bus, no hub
```

`ModulinoHub` doesn't use any resources and doesn't talk to the board by itself. The hub is set up on the board together with the first Modulino that uses one of its ports, and released when the last one is stopped.

### Rules and limitations

- **Hub address**: 0x70 by default. You can use up to 8 hubs on the same bus by setting the A0-A2 jumpers on the back of each hub (0x70-0x77). Each hub needs a different address.
- **Reserved addresses**: the addresses of the hubs you use are reserved, both on the main bus and behind the hub ports. Don't use them for other devices: a Modulino moved to the 0x70-0x77 range with the AddressChanger, or a third-party I2C device at one of those addresses, collides with the hub. The brick doesn't check for this.
- **Same port**: Modulini chained on the same port share that bus, so they need different addresses. Creating a second Modulino with the same address on the same port (or on the main bus) raises `ModulinoError`, and the first one keeps working.
- **No cascading**: a hub connected to the port of another hub is not supported. Connect every hub to the main bus.
- **LED Matrix**: the Modulino LED Matrix is not supported behind a hub. Connect it to the main bus.

## Modulino Buzzer

`Modulino_Buzzer` plays a tone with `tone(frequency, duration_ms)` and stops it with `noTone()`:

```python
from arduino.app_bricks.modulino import Modulino_Buzzer

buzzer = Modulino_Buzzer()   # autoscan for the address
buzzer.start()

buzzer.tone(440, 500)   # 440 Hz for 500 ms
buzzer.noTone()         # stop the current tone
```

`frequency` (Hz) and `duration_ms` must be non-negative integers, otherwise `ModulinoError` is raised.

## Modulino Distance

`Modulino_Distance` is a time-of-flight sensor at the fixed I2C address 0x29. `getDistance()` returns the last measure in millimeters, or NaN when it is not valid (for example nothing in range). `start()` waits up to 50 ms for the first measure, so the first `getDistance()` is already valid:

```python
from arduino.app_bricks.modulino import Modulino_Distance

distance = Modulino_Distance()
distance.start()

print(distance.getDistance())
```

`setThresholdCallback(threshold_mm, callback, hysteresis_mm=20)` calls `callback(state, distance_mm)` only when the proximity state changes:

- `Modulino_Distance.NEAR`: the distance went below `threshold_mm`.
- `Modulino_Distance.FAR`: the distance went above `threshold_mm + hysteresis_mm`. An invalid measure (nothing in range) also counts as `FAR`.

`distance_mm` is -1 when the measure is not valid.

```python
def on_proximity(state, distance_mm):
    if state == Modulino_Distance.NEAR:
        robot.stop()

distance.setThresholdCallback(150, on_proximity)   # stop below 15 cm, restart above 17 cm
```

The sensor measures every 20 ms. By default a new state is notified after the next measure confirms it: a single measure below `threshold_mm` followed by one at or above it does not raise `NEAR`. Calling `setThresholdCallback` again restarts from `FAR`, so an object already near is notified again. Use `setDebounce(ms)` to require a longer stable time. `removeThresholdCallback()` stops the notifications.

The notification goes from the board through the Linux side to your app, so it can arrive late or not at all if the app is busy or stuck. Don't use it as a safety-rated emergency stop.

## Modulino Pixels

`Modulino_Pixels` drives 8 RGB LEDs (index 0-7). `set(idx, r, g, b, brightness=25)` sets the color (0-255 per channel) and the brightness (0-100 %) of one LED, `clear(idx)` turns one off and `clearAll()` turns all of them off. Each call updates the LEDs right away.

```python
from arduino.app_bricks.modulino import Modulino_Pixels

pixels = Modulino_Pixels()   # autoscan for the address
pixels.start()

pixels.set(0, 255, 0, 0, 20)   # LED 0 red at 20 %
pixels.set(7, 0, 0, 255)       # LED 7 blue at the default 25 %

pixels.clearAll()              # all LEDs off
```

Out-of-range values raise `ModulinoError`.

To use a color, pass a `ModulinoColor` (or any `(r, g, b)` tuple) to `setColor(idx, color, brightness=25)`:

```python
from arduino.app_bricks.modulino import ModulinoColor

orange = ModulinoColor(255, 136, 0)
pixels.setColor(1, orange, 20)   # LED 1 orange at 20 %
```

## Modulino Light

`Modulino_Light` measures the color and the ambient light. `getLux()` returns the illuminance in lux, `getAmbientLight()` and `getIR()` the raw ambient and infrared measures, and `getColor()` the color as a `ModulinoColor`. Every call takes a new measure, which can take up to 200 ms.

```python
from arduino.app_bricks.modulino import Modulino_Light, Modulino_Pixels

light = Modulino_Light()
light.start()
pixels = Modulino_Pixels()
pixels.start()

print(light.getLux(), "lux")
color = light.getColor()
print(color.r, color.g, color.b)
pixels.setColor(0, color)        # show the color seen by the sensor
```

`ModulinoColor` is a tuple `(r, g, b)`, so it can be unpacked (`r, g, b = color`) and passed to libraries that take an RGB tuple, such as Pillow (`Image.new("RGB", size, color)`). OpenCV takes colors in BGR order: pass `color[::-1]`.

A failed measure raises `ModulinoError`.

## Modulino Latch Relay

`Modulino_LatchRelay` drives a latching relay, which keeps its position (on or off) when the power is removed. `set()` turns it on, `reset()` turns it off and `getStatus()` returns `Modulino_LatchRelay.ON`, `Modulino_LatchRelay.OFF` or `Modulino_LatchRelay.UNKNOWN`:

```python
from arduino.app_bricks.modulino import Modulino_LatchRelay

relay = Modulino_LatchRelay()   # autoscan for the address
relay.start()

relay.set()                     # relay on
print(relay.getStatus() == Modulino_LatchRelay.ON)
relay.reset()                   # relay off
```

After a power on the status is `UNKNOWN` until the first `set()` or `reset()`: the relay keeps its previous position, but the module doesn't know it. If reading the status fails, `getStatus()` returns the last status it read.

## Modulino Vibro

`Modulino_Vibro` drives a vibration motor. `on(duration_ms, power=Modulino_Vibro.MAXIMUM, block=False)` vibrates for `duration_ms` milliseconds and `off()` stops it earlier:

```python
from arduino.app_bricks.modulino import Modulino_Vibro

vibro = Modulino_Vibro()   # autoscan for the address
vibro.start()

vibro.on(500)                                   # 500 ms at full power, returns right away
vibro.on(1000, Modulino_Vibro.GENTLE, block=True)   # waits 1 s, then stops
vibro.off()                                     # stop the current vibration
```

The module stops by itself after `duration_ms`, so by default `on()` returns right away and your app keeps running. With `block=True` it waits `duration_ms` and then calls `off()`; only your app waits, the other Modulini keep working.

`power` goes from `Modulino_Vibro.STOP` (0) to `Modulino_Vibro.MAXIMUM` (50). The named levels are `GENTLE` (25), `MODERATE` (30), `MEDIUM` (35), `INTENSE` (40) and `POWERFUL` (45), and any integer in between is allowed. Out-of-range values raise `ModulinoError`.

## Modulino Motors

`Modulino_Motors` drives two DC motors (channels A and B) or one stepper motor. It starts in DC mode. `setSpeedA(percent)` and `setSpeedB(percent)` set one channel at a time, from -100 to 100 percent, and the sign gives the direction:

```python
from arduino.app_bricks.modulino import Modulino_Motors

motors = Modulino_Motors()   # autoscan for the address
motors.start()

motors.setSpeedA(60)     # channel A forward at 60%
motors.setSpeedB(-30)    # channel B backwards at 30%, A keeps running
motors.stopMotors()      # stop both
```

For a stepper motor, enable the stepper mode and pass the full steps per revolution of your motor. `move(steps, rpm, release_ms=0, block=False)` moves by `steps` (negative to go backwards) at `rpm`:

```python
motors = Modulino_Motors(steps_per_revolution=200)
motors.start()
motors.setStepperMode(True)

motors.setOnMoveCompleteCallback(lambda: print("done"))
motors.move(400, 60)                  # two revolutions at 60 rpm, returns right away
motors.move(-200, 30, block=True)     # one revolution backwards, waits for the end
motors.release()                      # free the shaft
```

With `release_ms=0` the motor keeps holding its position at the end of the move; 1 to 255 releases the coils that many milliseconds after the end. `hold()` and `release()` energize or release the coils right away. `setHalfStep(True)` doubles the steps per revolution for a smoother movement. The stepper methods raise `ModulinoError` until `setStepperMode(True)` is called, and `setSpeedA`, `setSpeedB` and `stopMotors` raise it in stepper mode.

Both modes share the driver settings: `setDecay(mode)` with `DECAY_SLOW`, `DECAY_MIXED_SLOW`, `DECAY_MIXED_FAST` or `DECAY_FAST`, `setHalfFullScale(enabled)` for the 1.9 A range (better accuracy at low currents) instead of 3.8 A, and `setFrequency(hz)` for the DC PWM frequency (200 to 60000 Hz, 20000 by default). `isStepperMode()`, `isHalfStep()`, `isBusy()`, `getDecay()` and `isHalfFullScale()` read the state from the module; `getFrequency()` returns the last frequency set. `getCurrentA()` and `getCurrentB()` return the current of each channel in milliamps.

`stop()` stops the motors (or releases the stepper coils) before closing the modulino.

## Modulino LED Matrix

`Modulino_LEDMatrix` shows 12x8 frames, the `Frame` objects of `arduino.app_utils` (the same ones used for the UNO Q LED matrix). `designer()` returns a `FrameDesigner` bound to the matrix size (`MODULINO_MATRIX`) and to the brightness levels of the current mode. `draw(frame)` shows a frame and `clear()` turns off every LED:

```python
from arduino.app_bricks.modulino import Modulino_LEDMatrix

matrix = Modulino_LEDMatrix()   # default address 0x39
matrix.start()

designer = matrix.designer()
frame = designer.empty()
frame.set_value(3, 5, 1)        # row 3, column 5 on
matrix.draw(frame)
```

The matrix starts in monochrome mode: a pixel is on when its value is greater than 0, whatever the brightness levels of the frame. `setGrayscale(True)` switches to 16 brightness levels: the frame values are rescaled from the levels of the frame, so a frame designed with 8 levels (as in the LED Matrix Painter) is shown without conversions.

`playAnimation(frames, loop=False)` plays an animation timed by the board, so it keeps a steady pace whatever your app is doing. `frames` is a list of `(frame, duration_ms)` items, the same taken by `sequence_to_c()`, up to `Modulino_LEDMatrix.MAX_FRAMES` (20):

```python
frames = [(frame_a, 200), (frame_b, 200), (frame_c, 400)]
matrix.playAnimation(frames, loop=True)
...
matrix.stopAnimation()          # the current frame stays on the matrix
```

The animation frames are shown on/off, also in grayscale mode. A new animation replaces the running one, and `draw()` or `clear()` stop it. Frames that are not 12x8 raise `ModulinoError`.

`setOnAnimationEndCallback(callback)` calls `callback()` when an animation without loop ends, after the duration of its last frame. A stopped or replaced animation does not call it. The callback runs on the bridge thread, where the Modulino methods can't be called: to chain animations, signal your app and play the next one from there:

```python
import threading

ended = threading.Event()
matrix.setOnAnimationEndCallback(ended.set)

matrix.playAnimation(frames)
ended.wait()
ended.clear()
matrix.playAnimation(next_frames)
```
