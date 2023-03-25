# import re
# import random


# class RuleBot:
#     # Potential Nevgative Reaponses
#     Nevgative_Responses = {"no", "nah", "naw"}
#     exit_responses = {"Goodbye", "Bye", "quit"}
#     starter_responses = {"hello", "hi"}

#     def __init__(self):
#         self.


# RuleBot()


conversation = {"previous_input": None, "previous_response": None}


def chatbot(input):
    response = generate_response(input, conversation)
    conversation["previous_input"] = input
    conversation["previous_response"] = response
    return response


def generate_response(input, conversation):
    if conversation["previous_input"] is None:
        return "Hello, how can I assist you today?"

    # Use the previous input and response to generate a more personalized response
    if "weather" in input.lower():
        if "kelvin" in input.lower():
            return "Sorry, I currently do not support Kelvin temperature units. Please try Celsius or Fahrenheit."
        elif "celsius" in input.lower():
            return "The temperature in London is 15 degrees Celsius."
        elif "fahrenheit" in input.lower():
            return "The temperature in London is 59 degrees Fahrenheit."
        else:
            return "What temperature unit would you like me to use?"

    # If we do not have a specific response for the input, use a generic response
    return "I'm sorry, I don't understand what you're asking for."


# Example usage
print(chatbot("Hello"))
print(chatbot("What is the weather like in London?"))
print(chatbot("What is the weather in Kelvin?"))
print(chatbot("What is the weather in Celsius?"))
print(chatbot("What is the weather like Fahrenheit?"))
