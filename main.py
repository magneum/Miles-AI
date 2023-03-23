from colorama import Fore, Style
from termcolor import cprint


cprint("ҠΛI: ", "green", "on_grey", attrs=["bold"])
cprint(": ", "white", "on_grey", attrs=[])


cprint("ҠΛI: ", "red", "on_grey", attrs=["bold"])
cprint(": ", "white", "on_grey", attrs=[])


cprint("ҠΛI: ", "yellow", "on_grey", attrs=["bold"])
cprint(": ", "white", "on_grey", attrs=[])


cprint("ҠΛI: ", "blue", "on_grey", attrs=["bold"])
cprint(": ", "white", "on_grey", attrs=[])

print(f"{Fore.BLUE}Hello, {Style.RESET_ALL} guys. {Fore.RED} I should be red.")
