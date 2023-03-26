import os
import openai
import logging
# =============================================================================================================


# Defining function to generate open response using OpenAI API
def open_response(usersaid):
    # Setting API key for OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Checking if API key is present
    if not openai.api_key:
        return "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."

    # Checking if user input is valid
    if not usersaid or not usersaid.strip():
        return "Invalid input. Please provide a valid question or statement."

    # Generating response using OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=usersaid,
        max_tokens=500,  # 2048 is max
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Returning response text
    if response.choices[0].text:
        return response.choices[0].text.strip()
    else:
        return "Sorry, I couldn't understand your question or statement."
# =============================================================================================================
