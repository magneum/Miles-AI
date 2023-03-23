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


api_url = f"http://localhost:3000/news"
response = requests.get(api_url)
data = response.json()

Title = data["title"]
Author = data["author"]
Description = data["description"]
Url = data["url"]
Image = data["urlToImage"]
PublishedAt = data["publishedAt"]
Content = data["content"]


print(f"{Fore.GREEN}TITLE: {Style.RESET_ALL}{Title}")
print(f"{Fore.GREEN}AUTHOR: {Style.RESET_ALL}{Author}")
print(f"{Fore.GREEN}DESCRIPTION: {Style.RESET_ALL}{Description}")
print(f"{Fore.GREEN}URL: {Style.RESET_ALL}{Url}")
print(f"{Fore.GREEN}IMAGE: {Style.RESET_ALL}{Image}")
print(f"{Fore.GREEN}PUBLISHEDAT: {Style.RESET_ALL}{PublishedAt}")
print(f"{Fore.GREEN}CONTENT: {Style.RESET_ALL}{Content}")
