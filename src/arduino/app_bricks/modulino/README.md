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

`Modulino_Distance` is a time-of-flight sensor at the fixed I2C address 0x29. `getDistance()` returns the last measure in millimeters, or NaN when it is not valid (for example nothing in range):

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

The sensor measures every 20 ms. By default a new state is notified after the next measure confirms it. Use `setDebounce(ms)` to require a longer stable time. `removeThresholdCallback()` stops the notifications.

The notification goes from the board through the Linux side to your app, so it can arrive late or not at all if the app is busy or stuck. Don't use it as a safety-rated emergency stop.
