# Modulino brick

## Modulino Hub

A Modulino Hub is an I2C multiplexer with 8 ports (0-7). It lets you use several Modulini that have the same I2C address, for example two Buttons with the default address, by plugging them into different ports.

Create a `ModulinoHub` with its address, take a port with `port(n)` and pass it to the Modulino constructor as `hub_port`:

```python
from arduino.app_bricks.modulino import ModulinoHub, ModulinoButtons, ModulinoThermo

hub = ModulinoHub(0x70)

buttons_a = ModulinoButtons(hub_port=hub.port(0))
buttons_b = ModulinoButtons(hub_port=hub.port(3))
thermo = ModulinoThermo()   # main bus, no hub
```

`ModulinoHub` doesn't use any resources and doesn't talk to the board by itself. The hub is set up on the board together with the first Modulino that uses one of its ports, and released when the last one is stopped.

### Rules and limitations

- **Hub address**: 0x70 by default. You can use up to 8 hubs on the same bus by setting the A0-A2 jumpers on the back of each hub (0x70-0x77). Each hub needs a different address.
- **Reserved addresses**: the addresses of the hubs you use are reserved, both on the main bus and behind the hub ports. Don't use them for other devices: a Modulino moved to the 0x70-0x77 range with the AddressChanger, or a third-party I2C device at one of those addresses, collides with the hub. The brick doesn't check for this.
- **Same port**: Modulini chained on the same port share that bus, so they need different addresses. Creating a second Modulino with the same address on the same port (or on the main bus) raises `ModulinoError`, and the first one keeps working.
- **No cascading**: a hub connected to the port of another hub is not supported. Connect every hub to the main bus.
- **LED Matrix**: the Modulino LED Matrix is not supported behind a hub. Connect it to the main bus.

## Modulino Buzzer

`ModulinoBuzzer` plays a tone with `tone(frequency, duration_ms)` and stops it with `no_tone()`:

```python
from arduino.app_bricks.modulino import ModulinoBuzzer

buzzer = ModulinoBuzzer()   # autoscan for the address
buzzer.start()

buzzer.tone(440, 500)   # 440 Hz for 500 ms
buzzer.no_tone()         # stop the current tone
```

`frequency` (Hz) and `duration_ms` must be non-negative integers, otherwise `ModulinoError` is raised.

## Modulino Distance

`ModulinoDistance` is a time-of-flight sensor at the fixed I2C address 0x29. `get_distance()` returns the last measure in millimeters, or NaN when it is not valid (for example nothing in range). `start()` waits up to 50 ms for the first measure, so the first `get_distance()` is already valid:

```python
from arduino.app_bricks.modulino import ModulinoDistance

distance = ModulinoDistance()
distance.start()

print(distance.get_distance())
```

`set_threshold_callback(threshold_mm, callback, hysteresis_mm=20)` calls `callback(state, distance_mm)` only when the proximity state changes:

- `ModulinoDistance.NEAR`: the distance went below `threshold_mm`.
- `ModulinoDistance.FAR`: the distance went above `threshold_mm + hysteresis_mm`. An invalid measure (nothing in range) also counts as `FAR`.

`distance_mm` is -1 when the measure is not valid.

```python
def on_proximity(state, distance_mm):
    if state == ModulinoDistance.NEAR:
        robot.stop()

distance.set_threshold_callback(150, on_proximity)   # stop below 15 cm, restart above 17 cm
```

The sensor measures every 20 ms. By default a new state is notified after the next measure confirms it: a single measure below `threshold_mm` followed by one at or above it does not raise `NEAR`. Calling `set_threshold_callback` again restarts from `FAR`, so an object already near is notified again. Use `set_debounce(ms)` to require a longer stable time. `remove_threshold_callback()` stops the notifications.

The notification goes from the board through the Linux side to your app, so it can arrive late or not at all if the app is busy or stuck. Don't use it as a safety-rated emergency stop.

## Modulino Pixels

`ModulinoPixels` drives 8 RGB LEDs (index 0-7). `set(idx, r, g, b, brightness=25)` sets the color (0-255 per channel) and the brightness (0-100 %) of one LED, `clear(idx)` turns one off and `clear_all()` turns all of them off. Each call updates the LEDs right away.

```python
from arduino.app_bricks.modulino import ModulinoPixels

pixels = ModulinoPixels()   # autoscan for the address
pixels.start()

pixels.set(0, 255, 0, 0, 20)   # LED 0 red at 20 %
pixels.set(7, 0, 0, 255)       # LED 7 blue at the default 25 %

pixels.clear_all()              # all LEDs off
```

Out-of-range values raise `ModulinoError`.

To use a color, pass a `ModulinoColor` (or any `(r, g, b)` tuple) to `set_color(idx, color, brightness=25)`:

```python
from arduino.app_bricks.modulino import ModulinoColor

orange = ModulinoColor(255, 136, 0)
pixels.set_color(1, orange, 20)   # LED 1 orange at 20 %
```

## Modulino Light

`ModulinoLight` measures the color and the ambient light. `get_lux()` returns the illuminance in lux, `get_ambient_light()` and `get_ir()` the raw ambient and infrared measures, and `get_color()` the color as a `ModulinoColor`. Every call takes a new measure, which can take up to 200 ms.

```python
from arduino.app_bricks.modulino import ModulinoLight, ModulinoPixels

light = ModulinoLight()
light.start()
pixels = ModulinoPixels()
pixels.start()

print(light.get_lux(), "lux")
color = light.get_color()
print(color.r, color.g, color.b)
pixels.set_color(0, color)        # show the color seen by the sensor
```

`ModulinoColor` is a tuple `(r, g, b)`, so it can be unpacked (`r, g, b = color`) and passed to libraries that take an RGB tuple, such as Pillow (`Image.new("RGB", size, color)`). OpenCV takes colors in BGR order: pass `color[::-1]`.

A failed measure raises `ModulinoError`.

## Modulino Latch Relay

`ModulinoLatchRelay` drives a latching relay, which keeps its position (on or off) when the power is removed. `set()` turns it on, `reset()` turns it off and `get_status()` returns `ModulinoLatchRelay.ON`, `ModulinoLatchRelay.OFF` or `ModulinoLatchRelay.UNKNOWN`:

```python
from arduino.app_bricks.modulino import ModulinoLatchRelay

relay = ModulinoLatchRelay()   # autoscan for the address
relay.start()

relay.set()                     # relay on
print(relay.get_status() == ModulinoLatchRelay.ON)
relay.reset()                   # relay off
```

After a power on the status is `UNKNOWN` until the first `set()` or `reset()`: the relay keeps its previous position, but the module doesn't know it. If reading the status fails, `get_status()` returns the last status it read.

## Modulino Vibro

`ModulinoVibro` drives a vibration motor. `on(duration_ms, power=ModulinoVibro.MAXIMUM, block=False)` vibrates for `duration_ms` milliseconds and `off()` stops it earlier:

```python
from arduino.app_bricks.modulino import ModulinoVibro

vibro = ModulinoVibro()   # autoscan for the address
vibro.start()

vibro.on(500)                                   # 500 ms at full power, returns right away
vibro.on(1000, ModulinoVibro.GENTLE, block=True)   # waits 1 s, then stops
vibro.off()                                     # stop the current vibration
```

The module stops by itself after `duration_ms`, so by default `on()` returns right away and your app keeps running. With `block=True` it waits `duration_ms` and then calls `off()`; only your app waits, the other Modulini keep working.

`power` goes from `ModulinoVibro.STOP` (0) to `ModulinoVibro.MAXIMUM` (50). The named levels are `GENTLE` (25), `MODERATE` (30), `MEDIUM` (35), `INTENSE` (40) and `POWERFUL` (45), and any integer in between is allowed. Out-of-range values raise `ModulinoError`.

## Modulino Motors

`ModulinoMotors` drives two DC motors (channels A and B) or one stepper motor. It starts in DC mode. `set_speed_a(percent)` and `set_speed_b(percent)` set one channel at a time, from -100 to 100 percent, and the sign gives the direction:

```python
from arduino.app_bricks.modulino import ModulinoMotors

motors = ModulinoMotors()   # autoscan for the address
motors.start()

motors.set_speed_a(60)     # channel A forward at 60%
motors.set_speed_b(-30)    # channel B backwards at 30%, A keeps running
motors.stop_motors()      # stop both
```

For a stepper motor, enable the stepper mode and pass the full steps per revolution of your motor. `move(steps, rpm, release_ms=0, block=False)` moves by `steps` (negative to go backwards) at `rpm`:

```python
motors = ModulinoMotors(steps_per_revolution=200)
motors.start()
motors.set_stepper_mode(True)

motors.set_on_move_complete_callback(lambda: print("done"))
motors.move(400, 60)                  # two revolutions at 60 rpm, returns right away
motors.move(-200, 30, block=True)     # one revolution backwards, waits for the end
motors.release()                      # free the shaft
```

With `release_ms=0` the motor keeps holding its position at the end of the move; 1 to 255 releases the coils that many milliseconds after the end. `hold()` and `release()` energize or release the coils right away. `set_half_step(True)` doubles the steps per revolution for a smoother movement. The stepper methods raise `ModulinoError` until `set_stepper_mode(True)` is called, and `set_speed_a`, `set_speed_b` and `stop_motors` raise it in stepper mode.

Both modes share the driver settings: `set_decay(mode)` with `DECAY_SLOW`, `DECAY_MIXED_SLOW`, `DECAY_MIXED_FAST` or `DECAY_FAST`, `set_half_full_scale(enabled)` for the 1.9 A range (better accuracy at low currents) instead of 3.8 A, and `set_frequency(hz)` for the DC PWM frequency (200 to 60000 Hz, 20000 by default). `is_stepper_mode()`, `is_half_step()`, `is_busy()`, `get_decay()` and `is_half_full_scale()` read the state from the module; `get_frequency()` returns the last frequency set. `get_current_a()` and `get_current_b()` return the current of each channel in milliamps.

`stop()` stops the motors (or releases the stepper coils) before closing the modulino.

## Modulino LED Matrix

`ModulinoLEDMatrix` shows 12x8 frames, the `Frame` objects of `arduino.app_utils` (the same ones used for the UNO Q LED matrix). `designer()` returns a `FrameDesigner` bound to the matrix size (`MODULINO_MATRIX`) and to the brightness levels of the current mode. `draw(frame)` shows a frame and `clear()` turns off every LED:

```python
from arduino.app_bricks.modulino import ModulinoLEDMatrix

matrix = ModulinoLEDMatrix()   # default address 0x39
matrix.start()

designer = matrix.designer()
frame = designer.empty()
frame.set_value(3, 5, 1)        # row 3, column 5 on
matrix.draw(frame)
```

The matrix starts in monochrome mode: a pixel is on when its value is greater than 0, whatever the brightness levels of the frame. `set_grayscale(True)` switches to 16 brightness levels: the frame values are rescaled from the levels of the frame, so a frame designed with 8 levels (as in the LED Matrix Painter) is shown without conversions.

`play_animation(frames, loop=False)` plays an animation timed by the board, so it keeps a steady pace whatever your app is doing. `frames` is a list of `(frame, duration_ms)` items, the same taken by `sequence_to_c()`, up to `ModulinoLEDMatrix.MAX_FRAMES` (20):

```python
frames = [(frame_a, 200), (frame_b, 200), (frame_c, 400)]
matrix.play_animation(frames, loop=True)
...
matrix.stop_animation()          # the current frame stays on the matrix
```

The animation frames are shown on/off, also in grayscale mode. A new animation replaces the running one, and `draw()` or `clear()` stop it. Frames that are not 12x8 raise `ModulinoError`.

`set_on_animation_end_callback(callback)` calls `callback()` when an animation without loop ends, after the duration of its last frame. A stopped or replaced animation does not call it. The callback runs on the bridge thread, where the Modulino methods can't be called: to chain animations, signal your app and play the next one from there:

```python
import threading

ended = threading.Event()
matrix.set_on_animation_end_callback(ended.set)

matrix.play_animation(frames)
ended.wait()
ended.clear()
matrix.play_animation(next_frames)
```
