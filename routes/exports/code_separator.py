import random
from colorama import Fore, Style


def code_separator(section_name):
    separator_width = 50
    separator_char = "*"
    section_label = f" {section_name} "
    section_label_width = separator_width - 2
    section_label_padding = (section_label_width - len(section_label)) // 2
    separator_line = separator_char * separator_width
    available_colors = [
        Fore.RED,
        Fore.GREEN,
        Fore.YELLOW,
        Fore.BLUE,
        Fore.MAGENTA,
        Fore.CYAN,
        Fore.WHITE,
    ]
    random_color = random.choice(available_colors)
    section_label_line = (
        separator_char
        + " " * section_label_padding
        + f"{random_color}{Style.BRIGHT}{section_name}{Style.RESET_ALL}"
        + " " * section_label_padding
        + separator_char
    )

    print(separator_line)
    print(section_label_line)
    print(separator_line)
