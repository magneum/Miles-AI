import requests


def get_weather(location, units="celsius"):
    api_key = "c9f2e2102e9849f3d601a41937d85a46"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    response = requests.get(url)
    data = response.json()

    if data["cod"] == 200:
        description = data["weather"][0]["description"]
        temperature_kelvin = data["main"]["temp"]
        if units.lower() == "kelvin":
            temperature = temperature_kelvin
            unit_symbol = "K"
        elif units.lower() == "fahrenheit":
            temperature = (temperature_kelvin - 273.15) * 9 / 5 + 32
            unit_symbol = "°F"
        else:
            temperature = temperature_kelvin - 273.15
            unit_symbol = "°C"
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        print(f"Weather in {location}:")
        print(f"Description: {description}")
        print(f"Temperature: {temperature:.2f} {unit_symbol}")
        print(f"Humidity: {humidity}%")
        print(f"Wind Speed: {wind_speed} m/s")
    else:
        print("Failed to fetch weather data.")


def chat_bot():
    while True:
        user_input = input(
            "Enter your location and temperature unit (e.g. 'Location, Unit'): "
        )
        input_parts = user_input.split(",")
        if len(input_parts) != 2:
            print("Invalid input. Please enter in the format: Location, Unit")
            continue
        location = input_parts[0].strip()
        units = input_parts[1].strip()
        get_weather(location.lower(), units.lower())


# Example usage:
chat_bot()
