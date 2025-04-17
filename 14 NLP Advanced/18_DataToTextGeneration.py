"""
Project 538: Data-to-Text Generation
Description:
Data-to-text generation involves converting structured data (like tables or databases) into coherent, human-readable text. This is useful in applications like automated report generation, weather forecasting, and financial summaries. In this project, we will generate text based on structured data using a transformer model.
"""

from transformers import pipeline
 
# 1. Load pre-trained model for text generation from structured data
data_to_text_generator = pipeline("text2text-generation", model="t5-small")
 
# 2. Define structured data (e.g., weather forecast data)
weather_data = {
    'city': 'New York',
    'temperature': '22°C',
    'humidity': '60%',
    'forecast': 'sunny',
    'wind_speed': '10 km/h'
}
 
# 3. Convert structured data into a text prompt
text_input = f"The weather forecast for {weather_data['city']} is as follows: The temperature is {weather_data['temperature']}, the humidity is {weather_data['humidity']}, with {weather_data['forecast']} skies and a wind speed of {weather_data['wind_speed']}."
 
# 4. Use the model to generate text based on the data
generated_text = data_to_text_generator(text_input, max_length=100, num_return_sequences=1)
 
# 5. Display the generated text
print(f"Generated Text: {generated_text[0]['generated_text']}")