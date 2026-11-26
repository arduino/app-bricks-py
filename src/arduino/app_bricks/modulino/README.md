# Modulino brick

The Modulino brick lets your app use the Modulino modules connected to the Qwiic connector of the board. The I2C communication, the polling of the inputs and the debounce run on the microcontroller (through the `Arduino_ModulinoBridge` sketch library, added by the brick); your Python app calls the methods of one class per Modulino:

| Class | Modulino | I2C address |
|---|---|---|
| `ModulinoThermo` | temperature and humidity | 0x44, fixed |
| `ModulinoMovement` | accelerometer and gyroscope | 0x6A by default |
| `ModulinoLight` | color and ambient light | 0x53, fixed |
| `ModulinoDistance` | time-of-flight distance | 0x29, fixed |
| `ModulinoButtons` | 3 buttons with LEDs | autoscan |
| `ModulinoJoystick` | joystick with button | autoscan |
| `ModulinoKnob` | rotary encoder with button | autoscan |
| `ModulinoBuzzer` | buzzer | autoscan |
| `ModulinoPixels` | 8 RGB LEDs | autoscan |
| `ModulinoVibro` | vibration motor | autoscan |
| `ModulinoLatchRelay` | latching relay | autoscan |
| `ModulinoMotors` | two DC motors or one stepper | autoscan |
| `ModulinoLEDMatrix` | 12x8 LED matrix | 0x39 by default |

## Using a Modulino

Create the Modulini at the top of your app. `App.run()` starts them and stops them when the app closes, so call their methods from your loop (or from code that runs after `App.run()` has started them):

```python
import time

from arduino.app_bricks.modulino import ModulinoThermo, ModulinoPixels
from arduino.app_utils import App

thermo = ModulinoThermo()
pixels = ModulinoPixels()

def loop():
    temperature = thermo.get_temperature()
    if temperature > 28:
        pixels.set_pixel(0, 255, 0, 0)   # LED 0 red
    else:
        pixels.clear_pixel(0)
    time.sleep(1)

App.run(user_loop=loop)
```

To use a Modulino before `App.run()`, start it yourself with `App.start_brick(modulino)`: `App.run()` doesn't start it again, and stops it when the app closes. Don't call `start()` directly before `App.run()`: the Modulino would be started twice, and the second start resets it (for example it turns the LEDs of the Pixels off).

`start()` initializes the Modulino on the board and puts it back in its default state. `stop()` stops its notifications and turns its outputs off: motors, LEDs, sound and vibration (the Latch Relay keeps its position). If a Modulino doesn't answer, its `start()` fails: the app logs the error and keeps running.

The snippets in the next sections show only the calls: put them in your loop, as above.

### Addresses

- **Autoscan**: the Modulini with a microcontroller of their own are found on the bus by default. Pass `i2c_address` only if you changed the address of the module.
- **Default address**: `ModulinoMovement` (0x6A) and `ModulinoLEDMatrix` (0x39) take an `i2c_address` too.
- **Fixed address**: `ModulinoThermo`, `ModulinoLight` and `ModulinoDistance` have no `i2c_address` argument: the address is the class constant `I2C_ADDRESS`.

`i2c_address` is the first argument of the constructors that take it: pass the hub port by name, `ModulinoButtons(hub_port=hub.port(0))` (see [Modulino Hub](#modulino-hub)).

### Errors

The methods raise:

- `TypeError` when an argument has the wrong type, for example `TypeError("frequency must be an integer, got float")`, and `ValueError` when it is out of range, for example `ValueError("power must be between 0 and 50, got 80")`;
- `RuntimeError` when a method can't be used in the current mode, for example `ModulinoMotors.hold()` in DC mode;
- `ModulinoError` when the board can't do the operation, as `modulino <name>: cannot <operation>`, for example `modulino thermo_44: cannot get humidity`. If the call to the board failed (for example a timeout), the original exception is its `__cause__`.

`ModulinoError` is an `AppError` (`arduino.app_utils`): when it reaches the top of your app uncaught, the app prints a readable report, with a hint on how to fix the problem when there is one (for example, to check the cable of a Modulino that doesn't answer).

`Modulino` is the base class of the Modulino classes and is not exported: creating it directly raises `TypeError`.

### Callbacks

Some Modulini call a function of your app when something happens: a button pressed, an object near, a move or an animation completed. Your callback runs on the bridge thread, where the Modulino methods can't be called. To act on a Modulino, signal your app (for example with a `threading.Event`) and call the method from your loop:

```python
import threading

from arduino.app_bricks.modulino import ModulinoButtons, ModulinoBuzzer
from arduino.app_utils import App

buttons = ModulinoButtons()
buzzer = ModulinoBuzzer()
pressed = threading.Event()

def on_buttons(button_a, button_b, button_c):
    if button_a:
        pressed.set()

buttons.set_on_change_callback(on_buttons)

def loop():
    if pressed.wait(timeout=1):
        pressed.clear()
        buzzer.play_tone(880, 100)   # a Modulino method: called from the loop, not from the callback

App.run(user_loop=loop)
```

Keep the callbacks short: while one runs, the other notifications wait. Passing something that is not callable raises `TypeError`.

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

## Modulino Thermo

`ModulinoThermo` measures the temperature and the relative humidity. `get_temperature()` returns degrees Celsius and `get_humidity()` the relative humidity in percent, both as `float`:

```python
from arduino.app_bricks.modulino import ModulinoThermo

thermo = ModulinoThermo()

print(f"{thermo.get_temperature():.1f} C, {thermo.get_humidity():.0f} %")
```

## Modulino Movement

`ModulinoMovement` measures the acceleration on 3 axes (in g) and the angular rate around them (in degrees per second). `get_sample()` takes a new measure and returns a `MovementSample`, a tuple `(x, y, z, roll, pitch, yaw)` that can also be read by field:

```python
from arduino.app_bricks.modulino import ModulinoMovement, MovementSample

movement = ModulinoMovement()

sample = movement.get_sample()
print(sample.x, sample.y, sample.z)   # about (0, 0, 1) g when the module lies flat
x, y, z, roll, pitch, yaw = sample
```

To receive the samples at a regular interval, `start_sample(ms, callback)` calls `callback(sample)` with a `MovementSample` every `ms` milliseconds (at least 100), until `stop_sample()`:

```python
def on_sample(sample: MovementSample):
    print(sample.roll, sample.pitch, sample.yaw)

movement.start_sample(200, on_sample)   # 5 samples per second
...
movement.stop_sample()
```

## Modulino Light

`ModulinoLight` measures the color and the ambient light. `get_lux()` returns the illuminance in lux, `get_ambient_light()` and `get_ir()` the raw ambient and infrared measures, and `get_color()` the color as a `RGBColor`. Every call takes a new measure, which can take up to 200 ms.

```python
from arduino.app_bricks.modulino import ModulinoLight, ModulinoPixels

light = ModulinoLight()
pixels = ModulinoPixels()

print(light.get_lux(), "lux")
color = light.get_color()
print(color.r, color.g, color.b)
pixels.set_color(0, color)   # show the color seen by the sensor
```

`RGBColor` is a tuple `(r, g, b)`, so it can be unpacked (`r, g, b = color`) and passed to libraries that take an RGB tuple, such as Pillow (`Image.new("RGB", size, color)`). OpenCV takes colors in BGR order: pass `color[::-1]`.

A failed measure raises `ModulinoError`.

## Modulino Distance

`ModulinoDistance` is a time-of-flight sensor at the fixed I2C address 0x29. `get_distance()` returns the last measure in millimeters, or NaN when it is not valid (for example nothing in range). `start()` waits up to 50 ms for the first measure, so the first `get_distance()` is already valid:

```python
from arduino.app_bricks.modulino import ModulinoDistance

distance = ModulinoDistance()

print(distance.get_distance())
```

`set_threshold_callback(threshold_mm, callback, hysteresis_mm=20)` calls `callback(state, distance_mm)` only when the proximity state changes:

- `ModulinoDistance.NEAR`: the distance went below `threshold_mm`.
- `ModulinoDistance.FAR`: the distance went above `threshold_mm + hysteresis_mm`. An invalid measure (nothing in range) also counts as `FAR`.

`distance_mm` is -1 when the measure is not valid.

```python
def on_proximity(state, distance_mm):
    if state == ModulinoDistance.NEAR:
        # Add your code here to stop a movement, e.g. set a threading.Event checked by your app
        pass

distance.set_threshold_callback(150, on_proximity)   # NEAR below 15 cm, FAR again above 17 cm
```

The callback can't call the Modulino methods (see [Callbacks](#callbacks)): to stop the motors, for example, signal your app and stop them from there.

The sensor measures every 20 ms. By default a new state is notified after the next measure confirms it: a single measure below `threshold_mm` followed by one at or above it does not raise `NEAR`. Calling `set_threshold_callback` again restarts from `FAR`, so an object already near is notified again. Use `set_debounce(ms)` to require a longer stable time. `remove_threshold_callback()` stops the notifications.

The notification goes from the board through the Linux side to your app, so it can arrive late or not at all if the app is busy or stuck. Don't use it as a safety-rated emergency stop.

## Modulino Buttons

`ModulinoButtons` has 3 buttons (A, B and C), each with an LED. `set_on_change_callback(callback)` calls `callback(button_a, button_b, button_c)` when a button is pressed or released, with the state of every button: 1 pressed, 0 released. `set_leds(led_a, led_b, led_c)` turns the LEDs on or off:

```python
from arduino.app_bricks.modulino import ModulinoButtons

buttons = ModulinoButtons()   # autoscan for the address

def on_buttons(button_a, button_b, button_c):
    print("A" if button_a else "-", "B" if button_b else "-", "C" if button_c else "-")

buttons.set_on_change_callback(on_buttons)
buttons.set_leds(True, False, True)   # LEDs A and C on
```

A change is notified when it has been stable for the debounce time, 30 ms by default: change it with `set_debounce(ms)`. `remove_on_change_callback()` stops the notifications. The LEDs are off after `start()` and `stop()`, and `start()` brings the debounce back to 30 ms.

## Modulino Joystick

`ModulinoJoystick` has two axes and a button. `set_on_change_callback(callback)` calls `callback(x, y, button)` when the position or the button changes: `x` and `y` go from about -128 to 127 and are 0 at the center, `button` is 1 pressed and 0 released.

```python
from arduino.app_bricks.modulino import ModulinoJoystick

joystick = ModulinoJoystick()   # autoscan for the address

def on_joystick(x, y, button):
    print(f"x {x:4} y {y:4} {'pressed' if button else ''}")

joystick.set_on_change_callback(on_joystick)
```

Near the center (within about 26 on both axes) the position reads 0, 0, so a joystick at rest doesn't send notifications. A change is notified when it has been stable for the debounce time, 30 ms by default: change it with `set_debounce(ms)`. `remove_on_change_callback()` stops the notifications.

## Modulino Knob

`ModulinoKnob` is a rotary encoder with a button. `set_on_change_callback(callback)` calls `callback(position, button, direction)` when the knob turns or the button changes:

- `position`: the encoder counter, which goes up turning one way and down turning the other way;
- `button`: 1 pressed, 0 released;
- `direction`: how much the position changed since the previous notification, positive or negative by the direction of the rotation (0 when only the button changed).

```python
from arduino.app_bricks.modulino import ModulinoKnob

knob = ModulinoKnob()   # autoscan for the address

def on_knob(position, button, direction):
    if direction:
        print("turned", "up" if direction > 0 else "down", "to", position)
    if button:
        print("pressed")

knob.set_on_change_callback(on_knob)
knob.set_start_position(0)   # the counter starts from 0
```

The knob keeps its position across apps. `set_start_position(position)` sets the counter: the new position is notified with `direction` 0, as it is not a rotation. In the same way, the first notification after `start()` doesn't count the position left by the previous app as a rotation. A change is notified when it has been stable for the debounce time, 30 ms by default: change it with `set_debounce(ms)`. `remove_on_change_callback()` stops the notifications.

## Modulino Buzzer

`ModulinoBuzzer` plays a tone with `play_tone(frequency, duration_ms)` and stops it with `stop_tone()`:

```python
from arduino.app_bricks.modulino import ModulinoBuzzer

buzzer = ModulinoBuzzer()   # autoscan for the address

buzzer.play_tone(440, 500)   # 440 Hz for 500 ms
buzzer.stop_tone()           # stop the current tone
```

`frequency` (Hz) and `duration_ms` must be non-negative integers: a wrong type raises `TypeError`, a negative value `ValueError`.

## Modulino Pixels

`ModulinoPixels` drives 8 RGB LEDs (index 0-7). `set_pixel(idx, r, g, b, brightness=25)` sets the color (0-255 per channel) and the brightness (0-100 %) of one LED, `clear_pixel(idx)` turns one off and `clear_all_pixels()` turns all of them off. Each call updates the LEDs right away.

```python
from arduino.app_bricks.modulino import ModulinoPixels

pixels = ModulinoPixels()   # autoscan for the address

pixels.set_pixel(0, 255, 0, 0, 20)   # LED 0 red at 20 %
pixels.set_pixel(7, 0, 0, 255)       # LED 7 blue at the default 25 %

pixels.clear_all_pixels()            # all LEDs off
```

Out-of-range values raise `ValueError`.

To use a color, pass a `RGBColor` (or any `(r, g, b)` tuple) to `set_color(idx, color, brightness=25)`:

```python
from arduino.app_bricks.modulino import RGBColor

orange = RGBColor(255, 136, 0)
pixels.set_color(1, orange, 20)   # LED 1 orange at 20 %
```

## Modulino Vibro

`ModulinoVibro` drives a vibration motor. `on(duration_ms, power=ModulinoVibro.MAXIMUM, block=False)` vibrates for `duration_ms` milliseconds and `off()` stops it earlier:

```python
from arduino.app_bricks.modulino import ModulinoVibro

vibro = ModulinoVibro()   # autoscan for the address

vibro.on(500)                                      # 500 ms at full power, returns right away
vibro.on(1000, ModulinoVibro.GENTLE, block=True)   # waits 1 s, then stops
vibro.off()                                        # stop the current vibration
```

The module stops by itself after `duration_ms`, so by default `on()` returns right away and your app keeps running. With `block=True` it waits `duration_ms` and then calls `off()`; only your app waits, the other Modulini keep working.

`power` goes from `ModulinoVibro.STOP` (0) to `ModulinoVibro.MAXIMUM` (50). The named levels are `GENTLE` (25), `MODERATE` (30), `MEDIUM` (35), `INTENSE` (40) and `POWERFUL` (45), and any integer in between is allowed. Out-of-range values raise `ValueError`.

## Modulino Latch Relay

`ModulinoLatchRelay` drives a latching relay, which keeps its position (on or off) when the power is removed. `set()` turns it on, `reset()` turns it off and `get_status()` returns `ModulinoLatchRelay.ON`, `ModulinoLatchRelay.OFF` or `ModulinoLatchRelay.UNKNOWN`:

```python
from arduino.app_bricks.modulino import ModulinoLatchRelay

relay = ModulinoLatchRelay()   # autoscan for the address

relay.set()                    # relay on
print(relay.get_status() == ModulinoLatchRelay.ON)
relay.reset()                  # relay off
```

After a power on the status is `UNKNOWN` until the first `set()` or `reset()`: the relay keeps its previous position, but the module doesn't know it. If reading the status fails, `get_status()` returns the last status it read. Stopping the Modulino, or closing the app, doesn't move the relay.

## Modulino Motors

`ModulinoMotors` drives two DC motors (channels A and B) or one stepper motor. It starts in DC mode. `set_speed_a(percent)` and `set_speed_b(percent)` set one channel at a time, from -100 to 100 percent, and the sign gives the direction:

```python
from arduino.app_bricks.modulino import ModulinoMotors

motors = ModulinoMotors()   # autoscan for the address

motors.set_speed_a(60)    # channel A forward at 60%
motors.set_speed_b(-30)   # channel B backwards at 30%, A keeps running
motors.stop_motors()      # stop both
```

For a stepper motor, enable the stepper mode and pass the full steps per revolution of your motor. `move(steps, rpm, release_ms=0, block=False)` moves by `steps` (negative to go backwards) at `rpm`:

```python
motors = ModulinoMotors(steps_per_revolution=200)

motors.set_stepper_mode(True)
motors.set_on_move_complete_callback(lambda: print("done"))
motors.move(400, 60)                # two revolutions at 60 rpm, returns right away
motors.move(-200, 30, block=True)   # one revolution backwards, waits for the end
motors.release()                    # free the shaft
```

With `release_ms=0` the motor keeps holding its position at the end of the move; 1 to 255 releases the coils that many milliseconds after the end. `hold()` and `release()` energize or release the coils right away. `set_half_step(True)` doubles the steps per revolution for a smoother movement. The stepper methods raise `RuntimeError` until `set_stepper_mode(True)` is called, and `set_speed_a`, `set_speed_b` and `stop_motors` raise it in stepper mode.

Both modes share the driver settings: `set_decay(mode)` with `DECAY_SLOW`, `DECAY_MIXED_SLOW`, `DECAY_MIXED_FAST` or `DECAY_FAST`, `set_half_full_scale(enabled)` for the 1.9 A range (better accuracy at low currents) instead of 3.8 A, and `set_frequency(hz)` for the DC PWM frequency (200 to 60000 Hz, 20000 by default). The configuration is cached: `is_stepper_mode()`, `is_half_step()`, `get_decay()`, `is_half_full_scale()` and `get_frequency()` return the values set (or the defaults `start()` reads from the board) without reading the module, which is read again only after a failure or a mode change. `is_busy()` always reads the module. If the module is powered off or reconnected, call `start()` again. `get_current_a()` and `get_current_b()` return the current of each channel in milliamps.

`start()` brings the module back to DC mode with the default settings. `stop()`, also when the app closes, stops the DC motors and, in stepper mode, releases the coils.

## Modulino LED Matrix

`ModulinoLEDMatrix` shows 12x8 frames, the `Frame` objects of `arduino.app_utils` (the same ones used for the UNO Q LED matrix). `designer()` returns a `FrameDesigner` bound to the matrix size (`MODULINO_MATRIX`) and to the brightness levels of the current mode. `draw(frame)` shows a frame and `clear()` turns off every LED:

```python
from arduino.app_bricks.modulino import ModulinoLEDMatrix

matrix = ModulinoLEDMatrix()   # default address 0x39

designer = matrix.designer()
frame = designer.empty()
frame.set_value(3, 5, 1)       # row 3, column 5 on
matrix.draw(frame)
```

The matrix starts in monochrome mode: a pixel is on when its value is greater than 0, whatever the brightness levels of the frame. `set_grayscale(True)` switches to 16 brightness levels: the frame values are rescaled from the levels of the frame, so a frame designed with 8 levels (as in the LED Matrix Painter) is shown without conversions.

`play_animation(frames, loop=False)` plays an animation timed by the board, so it keeps a steady pace whatever your app is doing. `frames` is a list of `(frame, duration_ms)` items, the same taken by `sequence_to_c()`, up to `ModulinoLEDMatrix.MAX_FRAMES` (20):

```python
frames = [(frame_a, 200), (frame_b, 200), (frame_c, 400)]
matrix.play_animation(frames, loop=True)
...
matrix.stop_animation()   # the current frame stays on the matrix
```

The animation frames are shown on/off, also in grayscale mode. A new animation replaces the running one, and `draw()` or `clear()` stop it. Frames that are not 12x8 raise `ValueError`.

`on_animation_end_callback(callback)` calls `callback()` when an animation without loop ends, after the duration of its last frame. A stopped or replaced animation does not call it. To chain animations, signal your app from the callback and play the next one from your loop:

```python
import threading

ended = threading.Event()
matrix.on_animation_end_callback(ended.set)

matrix.play_animation(frames)
ended.wait()
ended.clear()
matrix.play_animation(next_frames)
```
