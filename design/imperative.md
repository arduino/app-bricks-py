# Imperative style design

## Requirements

1. Imperative programming style
2. Simple and straightforward: no need to adapt to a framework, the user "just writes" code in the most basic way

## Example syntax
```python
userinput = UserTextInput()
city = userinput.get()

weather_forecast = WeatherForecast()
forecast = weather_forecast.get_forecast_by_city(city)

print(forecast.temperature_c)

db = DBStore()
db.save(forecast.temperature_c)
```