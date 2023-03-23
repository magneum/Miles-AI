# import os
import requests
from flask import Flask
from colorama import Fore, Style
# os.system("python3 -m app")

# api_key = "860b08c7c72841e6a8a53a3c3cfa8ec6"
# response = requests.get(
#     f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}")
# data = response.json()
# articles = data["articles"]
# for article in articles:
#     print(article["title"])


# api_url = f"http://localhost:3000/news"
# response = requests.get(api_url)
# data = response.json()
# print(f"> {Fore.GREEN}TITLE: {Style.RESET_ALL}" + data["title"])
# print(f"> {Fore.GREEN}AUTHOR: {Style.RESET_ALL}" + data["author"])
# print(f"> {Fore.GREEN}DESCRIPTION: {Style.RESET_ALL}" + data["description"])
# print(f"> {Fore.GREEN}PUBLISHEDAT: {Style.RESET_ALL}" + data["publishedAt"])
# print(f"> {Fore.GREEN}CONTENT: {Style.RESET_ALL}" + data["content"])
# print(f"> {Fore.GREEN}URL: {Style.RESET_ALL}" + data["url"])
# print(f"> {Fore.GREEN}IMAGE: {Style.RESET_ALL}" + data["urlToImage"])


api_url = f"http://localhost:3000/weather"
response = requests.get(api_url)
data = response.json()
