import re
import math


# Regular expressions for different types of calculations
def calculate(input_string):
    # Define the regular expression pattern for each calculation type
    regex_calculations = [
        # Addition
        {
            "regex": r"what is (?P<first_num>\d+) plus (?P<second_num>\d+)",
            "function": lambda match: int(match.group("first_num")) + int(match.group("second_num")),
            "message": "The sum is"
        },
        # Subtraction
        {
            "regex": r"what is (?P<first_num>\d+) minus (?P<second_num>\d+)",
            "function": lambda match: int(match.group("first_num")) - int(match.group("second_num")),
            "message": "The difference is"
        },
        # Multiplication
        {
            "regex": r"what is (?P<first_num>\d+) times (?P<second_num>\d+)",
            "function": lambda match: int(match.group("first_num")) * int(match.group("second_num")),
            "message": "The product is"
        },
        # Pythagorean theorem
        {
            "regex": r"hypotenuse of triangle with sides (?P<side1>\d+(\.\d+)?), (?P<side2>\d+(\.\d+)?)",
            "function": lambda match: math.sqrt(float(match.group("side1")) ** 2 + float(match.group("side2")) ** 2),
            "message": "The hypotenuse of a triangle with sides"
        },
        # Geometry
        {
            "regex": r"sum of angles in (?P<shape>triangle|quadrilateral|pentagon|hexagon|heptagon|octagon|nonagon|decagon)",
            "function": lambda match: {
                "triangle": 180,
                "quadrilateral": 360,
                "pentagon": 540,
                "hexagon": 720,
                "heptagon": 900,
                "octagon": 1080,
                "nonagon": 1260,
                "decagon": 1440
            }.get(match.group("shape"), None),
            "message": "The sum of angles in a"
        },
        # Exponentiation
        {
            "regex": r"^(?P<base>-?\d+\.?\d*)\s*\^\s*(?P<exponent>-?\d+\.?\d*)$",
            "function": lambda match: float(match.group("base")) ** float(match.group("exponent")),
            "message": "The result of exponentiation is"
        },
        # Square root
        {
            "regex": r"^sqrt\((?P<number>-?\d+\.?\d*)\)$",
            "function": lambda match: math.sqrt(float(match.group("number"))),
            "message": "The square root is"
        },
        # Cube root
        {
            "regex": r"^cbrt\((?P<number>-?\d+\.?\d*)\)$",
            "function": lambda match: round(float(match.group("number")) ** (1/3), 2),
            "message": "The cube root is"
        },
        # Nth root
        {
            "regex": r"^(?P<root>-?\d+\.?\d*)-th\s*root\s*of\s*(?P<number>-?\d+\.?\d*)$",
            "function": lambda match: float(match.group("number"))**(1/float(match.group("root"))),
            "message": "The nth root is"
        },
        # Logarithm
        {
            "regex": r"^log\s*\((?P<number>-?\d+\.?\d*)\s*,\s*(?P<base>-?\d+\.?\d*)\)$",
            "function": lambda match: math.log(float(match.group("number")), float(match.group("base"))),
            "message": "The logarithm is"
        },
        # Natural logarithm
        {
            "regex": r"^ln\s*\((?P<number>-?\d+\.?\d*)\)$",
            "function": lambda match: math.log(float(match.group("number"))) if float(match.group("number")) > 0 else None,
            "message": "The natural logarithm is",
            "error_message": "Invalid argument for natural logarithm"
        },

        # Factorial
        {
            "regex": r"^(?P<number>-?\d+\.?\d*)\s*!$",
            "function": lambda match: math.factorial(int(match.group("number"))),
            "message": "The factorial is "
        },

        # Percentage calculation
        {
            "regex": r"^(?P<number>-?\d+\.?\d*)\s*%\s*(?P<percentage>-?\d+\.?\d*)$",
            "function": lambda match: float(match.group("number")) * float(match.group("percentage")) / 100
            if float(match.group("percentage")) != 0
            else "Error: Cannot divide by zero.",
            "message": "The result is"
        },
        # Circle area
        {
            "regex": r"^circle\s*area\s*\((?P<radius>-?\d+\.?\d*)\)$",
            "function": lambda match: circle_area(match.group("radius")),
            "message": "The area is"
        },
        # Circle circumference
        {
            "regex": r"^circle\s*(circumference|perimeter)\s*\((?P<radius>-?\d+\.?\d*)\)$",
            "function": lambda match: 2 * 3.14159 * float(match.group("radius")),
            "message": "The circumference of a circle with radius"
        },

        # Pythagorean theorem
        {
            "regex": r"^pythagorean\s*theorem\s*\((?P<leg1>-?\d+\.?\d*)\s*,\s*(?P<leg2>-?\d+\.?\d*)\)$",
            "function": lambda match: (float(match.group("leg1")) ** 2 + float(match.group("leg2")) ** 2) ** 0.5,
            "message": "The hypotenuse is"
        },
        # Triangle area
        {
            "regex": r"^triangle\s*area\s*\((?P<base>-?\d+\.?\d*)\s*,\s*(?P<height>-?\d+\.?\d*)\)$",
            "function": lambda match: 0.5 * float(match.group("base")) * float(match.group("height")),
            "message": "The area is"
        },

        # Triangle semiperimeter
        {
            # Match "triangle semiperimeter" with optional whitespace
            "regex": r"^triangle\s*semiperimeter\s*"
            r"\("  # Match opening parenthesis
            # Match first side (optional minus sign, digits, optional decimal point)
            r"(?P<side1>-?\d+\.?\d*)"
            r"\s*,\s*"  # Match comma with optional whitespace on both sides
            r"(?P<side2>-?\d+\.?\d*)"  # Match second side
            r"\s*,\s*"  # Match comma with optional whitespace on both sides
            r"(?P<side3>-?\d+\.?\d*)"  # Match third side
            r"\)$",  # Match closing parenthesis
            "function": lambda match: (float(match.group("side1")) + float(match.group("side2")) + float(match.group("side3"))) / 2,
            "message": "The semiperimeter of the triangle is"
        },

        # Rectangle area
        {
            "regex": r"^rectangle\s*area\s*\((?P<width>-?\d+\.?\d*)\s*,\s*(?P<height>-?\d+\.?\d*)\)$",
            "function": lambda match: float(match.group("width")) * float(match.group("height")),
            "message": "The area is:",
        },

        # Celsius to Fahrenheit conversion
        {
            "regex": r"^(?P<celsius>-?\d+\.?\d*)\sC\sto\s*F$",
            "function": lambda match: (float(match.group("celsius")) * 1.8) + 32,
            "message": "The temperature in Fahrenheit is"
        },

        # Fahrenheit to Celsius
        {
            "regex": r"^(?P<fahrenheit>-?\d+\.?\d*)\sF\sto\s*C$",
            "function": lambda match: (float(match.group("fahrenheit")) - 32) * 5 / 9,
            "message": "The temperature in Celsius is"
        },
        # Kilometers to miles conversion
        {
            "regex": r"^(?P<kilometers>-?\d+.?\d*)\sKM\sto\s*MILES?$",
            "function": lambda match: float(match.group("kilometers")) / 1.60934,
            "message": "The distance in miles is"
        },
        # Feet to meters conversion
        {
            "regex": r"^(?P<feet>-?\d+\.?\d*)\sFT\sto\s*M$",
            "function": lambda match: float(match.group("feet")) * 0.3048,
            "message": "The value in meters is"
        },
        # Meters to feet conversion
        {
            "regex": r"^(?P<meters>-?\d+\.?\d*)\sM\sto\s*FT$",
            "function": lambda match: float(match.group("meters")) * 3.28084,
            "message": "The length in feet is"
        },
        # Inches to centimeters conversion
        {
            "regex": r"^(?P<inches>-?\d+\.?\d*)\s*(IN|inch(es)?)\s+to\s+(CM|centimeter(s)?)$",
            "function": lambda match: float(match.group("inches")) * 2.54,
            "message": "The length in centimeters is"
        },
        # Centimeters to Inches
        {
            "regex": r"^(?P<centimeters>-?\d+\.?\d*)\s*CM\s*to\s*IN(CH)?$",
            "function": lambda match: float(match.group("centimeters")) / 2.54,
            "message": "The length in inches is:"
        },
        # Pounds to Kilograms
        {
            "regex": r"^(?P<pounds>-?\d+\.?\d*)\s*LB(?:S)?\s*TO\s*KG$",
            "function": lambda match: float(match.group("pounds")) * 0.453592,
            "message": "The weight in kilograms is"
        },
        # Kilograms to pounds
        {
            "regex": r"^(?P<kilograms>-?\d+(?:\.\d+)?)\s*KG(?:\s+TO)?\s+LBS?$",
            "function": lambda match: float(match.group("kilograms")) * 2.20462,
            "message": "The weight in pounds is"
        },
        # Quadratic equation
        {
            "regex": r"^(?P<a>-?\d+\.?\d*)x\^2\s*[+-]\s*(?P<b>-?\d+\.?\d*)x\s*[+-]\s*(?P<c>-?\d+\.?\d*)$",
            "function": lambda match: solve_quadratic_equation(float(match.group('a')), float(match.group('b')), float(match.group('c'))),
            "message": "The solutions are"
        },
        # Slope-intercept form
        {
            "regex": r"^y\s*[=+-]\s*(?P<m>-?\d+\.?\d*)x\s*[+-]\s*(?P<b>-?\d+\.?\d*)$",
            "function": lambda match: f"The equation of the line in slope-intercept form is y = {match.group('m')}x + {match.group('b')}.",
            "message": "Solution is"
        },

        # Standard form
        {
            "regex": r"^(?P<a>-?\d+\.?\d*)x\s*[+\-]\s*(?P<b>-?\d+\.?\d*)y\s*[=]\s*(?P<c>-?\d+\.?\d*)$",
            "function": lambda match: f"The equation is {match.group('a')}x {'+' if float(match.group('b')) >= 0 else '-'} {abs(float(match.group('b'))) if float(match.group('b')) != 1 else ''}y {'+' if float(match.group('c')) >= 0 else '-'} {abs(float(match.group('c')))}",
            "message": "Standard form:"
        },
        # Slope-intercept form
        {
            "regex": r"^y\s*[=]\s*(?P<m>-?\d+\.?\d*)x\s*[+\-]?\s*(?P<b>-?\d+\.?\d*)$",
            "function": lambda match: f"The equation is y = {match.group('m')}x {'+' if float(match.group('b')) >= 0 else '-'} {abs(float(match.group('b')))}",
            "message": "Slope-intercept form:"
        },
        # Point-slope form
        {
            "regex": r"^y\s*[=]\s*(?P<m>-?\d+\.?\d*)x\s*[+\-]\s*(?P<b>-?\d+\.?\d*)$",
            "function": lambda match: f"The equation is y = {match.group('m')}x {'+' if float(match.group('b')) >= 0 else '-'} {abs(float(match.group('b')))}",
            "message": "Point-slope form:"
        },
        # System of equations
        {
            "regex": r"^{\s*(?P<a1>-?\d+\.?\d*)x\s*[+-]\s*(?P<b1>-?\d+\.?\d*)y\s*=\s*(?P<c1>-?\d+\.?\d*),\s*(?P<a2>-?\d+\.?\d*)x\s*[+-]\s*(?P<b2>-?\d+\.?\d*)y\s*=\s*(?P<c2>-?\d+\.?\d*)\s*}$",
            "function": lambda match: solve_system_of_equations(float(match.group("a1")), float(match.group("b1")), float(match.group("c1")),
                                                                float(match.group("a2")), float(match.group("b2")), float(match.group("c2"))),
            "message": "The solution is"
        },

        # Interest calculation
        {
            "regex": r"^(?P<p>-?\d+\.?\d*)\s*\times\s*(?P<r>-?\d+\.?\d*)\s*\times\s*(?P<t>-?\d+\.?\d*)$",
            "function": lambda match: round(float(match.group("p")) * float(match.group("r")) * float(match.group("t")), 2),
            "message": "The simple interest is: "
        },

        # Distance formula
        {
            "regex": r"^distance formula (?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)$",
            "function": lambda match: ((float(match.group('x2')) - float(match.group('x1'))) ** 2 + (float(match.group('y2')) - float(match.group('y1'))) ** 2) ** 0.5,
            "message": "The distance is"
        },

        # Slope formula
        {
            "regex": r"^slope\sformula\s\*\s(?P<x1>-?\d+\.?\d*)\s*,\s*(?P<y1>-?\d+\.?\d*)\s*,\s*(?P<x2>-?\d+\.?\d*)\s*,\s*(?P<y2>-?\d+\.?\d*)\s*$",
            "function": lambda match: (float(match.group("y2")) - float(match.group("y1"))) / (float(match.group("x2")) - float(match.group("x1"))),
            "message": "The slope is"
        },
        # Midpoint formula
        {
            "regex": r"^midpoint\sformula\s\s∗(?P<x1>−?\d+?˙\d∗)\s∗,\s∗(?P<y1>−?\d+?˙\d∗)\s∗,\s∗(?P<x2>−?\d+?˙\d∗)\s∗,\s∗(?P<y2>−?\d+?˙\d∗)\s∗$",
            "function": lambda match: ((float(match.group("x1")) + float(match.group("x2"))) / 2, (float(match.group("y1")) + float(match.group("y2"))) / 2),
            "message": "The midpoint is"
        },
        # Quartic equation
        {
            "regex": r"^(?P<a>-?\d+\.?\d*)x\^4\s*[+-]\s*(?P<b>-?\d+\.?\d*)x\^3\s*[+-]\s*(?P<c>-?\d+\.?\d*)x\^2\s*[+-]\s*(?P<d>-?\d+\.?\d*)x\s*[+-]\s*(?P<e>-?\d+\.?\d*)$",
            "function": lambda match: solve_quartic(float(match.group('a')), float(match.group('b')), float(match.group('c')), float(match.group('d')), float(match.group('e'))),
            "message": "The roots are"
        }
    ]

    # Loop through the regex_calculations list and check if the user input matches any of the regular expressions
    for calculation in regex_calculations:
        match = re.match(calculation["regex"], input_string)
        if match:
            # Call the function associated with the matching regular expression and print the result
            result = calculation["function"](match)
            message = calculation["message"]
            print(f"{message} {result}")
            return result

    # If the user input does not match any of the regular expressions, print an error message
    print("Invalid input format.")
    miles_speaker("Invalid input format.")
    return None


# # =============================================================================================================
# # Define the regular expression pattern
# regex_pythagoras = r"hypotenuse of triangle with sides (?P<side1>\d+(\.\d+)?), (?P<side2>\d+(\.\d+)?)"
# # Match the input string to the regular expression pattern
# match = re.match(regex_pythagoras, usersaid)
# # If the input string matches the pattern, perform the calculation
# if match:
#     # Get the sides from the matched pattern
#     side1 = float(match.group("side1"))
#     side2 = float(match.group("side2"))
#     # Calculate the hypotenuse using the Pythagorean theorem
#     hypotenuse = math.sqrt(side1 ** 2 + side2 ** 2)
#     # Print the result
#     print(
#         f"The hypotenuse of a triangle with sides {side1} and {side2} is {hypotenuse:.2f}.")
#     miles_speaker(
#         f"The hypotenuse of a triangle with sides {side1} and {side2} is {hypotenuse:.2f}.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_rectangle_perimeter = r"perimeter of rectangle with sides (?P<length>\d+(\.\d+)?), (?P<width>\d+(\.\d+)?)"
# # Match the input string to the regular expression pattern
# match = re.match(regex_rectangle_perimeter, usersaid)
# # If the input string matches the pattern, calculate the perimeter
# if match:
#     # Get the length and width from the matched pattern
#     length = float(match.group("length"))
#     width = float(match.group("width"))
#     # Calculate the perimeter of the rectangle
#     perimeter = 2 * (length + width)
#     # Print the result
#     print(
#         f"The perimeter of a rectangle with sides {length} and {width} is {perimeter:.2f}.")
#     miles_speaker(
#         f"The perimeter of a rectangle with sides {length} and {width} is {perimeter:.2f}.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_geometry = r"sum of angles in (?P<shape>triangle|quadrilateral|pentagon|hexagon|heptagon|octagon|nonagon|decagon)"
# # Match the input string to the regular expression pattern
# match = re.match(regex_geometry, usersaid)
# # If the input string matches the pattern, perform the calculation
# if match:
#     # Get the shape from the matched pattern
#     shape = match.group("shape")
#     # Calculate the sum of angles based on the shape
#     if shape == "triangle":
#         sum_angles = 180
#     elif shape == "quadrilateral":
#         sum_angles = 360
#     elif shape == "pentagon":
#         sum_angles = 540
#     elif shape == "hexagon":
#         sum_angles = 720
#     elif shape == "heptagon":
#         sum_angles = 900
#     elif shape == "octagon":
#         sum_angles = 1080
#     elif shape == "nonagon":
#         sum_angles = 1260
#     elif shape == "decagon":
#         sum_angles = 1440
#     # Print the result
#     print(f"The sum of angles in a {shape} is {sum_angles} degrees.")
#     miles_speaker(f"The sum of angles in a {shape} is {sum_angles} degrees.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# regex_addition = r"^(?P<first_number>-?\d+\.?\d*)\s*\+\s*(?P<second_number>-?\d+\.?\d*)$"
# match = re.match(regex_addition, usersaid)
# if match:
#     # Extract the first and second numbers from the match object
#     first_number = float(match.group("first_number"))
#     second_number = float(match.group("second_number"))

#     # Perform the addition calculation
#     result = first_number + second_number

#     # Print the result
#     print(f"{first_number} + {second_number} = {result}")
#     miles_speaker(f"{first_number} + {second_number} = {result}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# regex_subtraction = r"^(?P<first_number>-?\d+\.?\d*)\s*-\s*(?P<second_number>-?\d+\.?\d*)$"
# match = re.match(regex_subtraction, usersaid)
# if match:
#     # Get the numbers from the matched pattern
#     first_number = float(match.group("first_number"))
#     second_number = float(match.group("second_number"))
#     # Call the subtraction function with the numbers
#     result = subtract(first_number, second_number)
#     # Print the result
#     print(f"{first_number} - {second_number} = {result}")
#     miles_speaker(f"{first_number} - {second_number} = {result}")
#     break
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# regex_multiplication = r"^(?P<first_number>-?\d+\.?\d*)\s*\*\s*(?P<second_number>-?\d+\.?\d*)$"
# # find matches
# matches = re.match(regex_multiplication, usersaid)
# # extract values from matches
# if matches:
#     first_number = float(matches.group("first_number"))
#     second_number = float(matches.group("second_number"))
#     # perform calculation
#     result = first_number * second_number
#     # print result
#     print(result)
#     miles_speaker(result)
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# regex_division = r"^(?P<first_number>-?\d+\.?\d*)\s*\/\s*(?P<second_number>-?\d+\.?\d*)$"
# match = re.match(regex_division, usersaid)
# if match:
#     first_number = float(match.group("first_number"))
#     second_number = float(match.group("second_number"))
#     result = first_number / second_number
#     print(result)
#     miles_speaker(result)
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# regex_exponentiation = r"^(?P<base>-?\d+\.?\d*)\s*\^\s*(?P<exponent>-?\d+\.?\d*)$"
# match = re.match(regex_exponentiation, usersaid)
# if match:
#     # Extract the base and exponent from the match object
#     base = float(match.group("base"))
#     exponent = float(match.group("exponent"))
#     # Perform the exponentiation calculation
#     result = base ** exponent
#     # Print the result
#     print(f"{base}^{exponent} = {result}")
#     miles_speaker(f"{base}^{exponent} = {result}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# regex_square_root = r"^sqrt\((?P<number>-?\d+\.?\d*)\)$"
# # Attempt to match user input to square root regex pattern
# match = re.match(regex_square_root, usersaid)
# # If a match is found, perform square root calculation
# if match:
#     # Extract number from match object
#     number = float(match.group("number"))
#     # Calculate square root
#     result = math.sqrt(number)
#     # Print result
#     print(f"sqrt({number}) = {result}")
#     miles_speaker(f"sqrt({number}) = {result}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define the regex pattern for cube root
# regex_cube_root = r"^cbrt\((?P<number>-?\d+\.?\d*)\)$"
# # Check if input matches the cube root pattern
# match = re.match(regex_cube_root, usersaid)
# if match:
#     # Extract the number from the match object
#     number = float(match.group("number"))
#     # Calculate the cube root
#     result = number ** (1/3)
#     # Print the result
#     print(f"The cube root of {number} is {result:.2f}")
#     miles_speaker(f"The cube root of {number} is {result:.2f}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define the regular expression pattern
# regex_n_root = r"^(?P<root>-?\d+\.?\d*)-th\s*root\s*of\s*(?P<number>-?\d+\.?\d*)$"
# # Match the input string with the regular expression pattern
# match = re.match(regex_n_root, usersaid)
# # If there"s a match, extract the root and number values
# if match:
#     root = float(match.group("root"))
#     number = float(match.group("number"))

#     # Calculate the n-th root of the number
#     result = math.pow(number, 1 / root)

#     # Print the result
#     print(f"{root}-th root of {number} = {result}")
#     miles_speaker(f"{root}-th root of {number} = {result}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# regex_logarithm = r"^log\s*\((?P<number>-?\d+\.?\d*)\s*,\s*(?P<base>-?\d+\.?\d*)\)$"
# match = re.match(regex_logarithm, usersaid)
# if match:
#     # Extract the number and base from the match object
#     number = float(match.group("number"))
#     base = float(match.group("base"))
#     # Calculate the logarithm
#     result = math.log(number, base)
#     # Print the result
#     print(f"log({number}, {base}) = {result}")
#     miles_speaker(f"log({number}, {base}) = {result}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define the regex pattern
# regex_natural_logarithm = r"^ln\s*\((?P<number>-?\d+\.?\d*)\)$"
# # Use re.match() to match the pattern to the user input
# match = re.match(regex_natural_logarithm, usersaid)
# # If there"s a match, extract the number and compute the natural logarithm
# if match:
#     number = float(match.group("number"))
#     result = math.log(number)
#     print(f"ln({number}) = {result}")
#     miles_speaker(f"ln({number}) = {result}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define the regex pattern for factorial calculation
# regex_factorial = r"^(?P<number>-?\d+\.?\d*)\s*!$"
# # Check if the user input matches the factorial regex pattern
# match = re.match(regex_factorial, usersaid)
# if match:
#     # Extract the number from the match object
#     number = int(float(match.group("number")))  # float to handle decimals
#     # Calculate the factorial of the number
#     result = 1
#     for i in range(1, number + 1):
#         result *= i
#     # Print the result
#     print(f"{number}! = {result}")
#     miles_speaker(f"{number}! = {result}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define regex pattern for percentage calculation
# regex_percentage = r"^(?P<number>-?\d+\.?\d*)\s*%\s*(?P<percentage>-?\d+\.?\d*)$"
# # Match user input with the regex pattern
# match = re.match(regex_percentage, usersaid)
# if match:
#     # Extract the number and percentage from the match object
#     number = float(match.group("number"))
#     percentage = float(match.group("percentage"))
#     # Calculate the result
#     result = number * (percentage / 100)
#     # Print the result
#     print(f"{number}% of {percentage} = {result}")
#     miles_speaker(f"{number}% of {percentage} = {result}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define regex pattern for circle area calculation
# regex_circle_area = r"^circle\s*area\s*\((?P<radius>-?\d+\.?\d*)\)$"
# # Match the user input with the regex pattern
# match = re.match(regex_circle_area, usersaid)
# # If the input matches the pattern, perform the calculation
# if match:
#     radius = float(match.group("radius"))
#     area = 3.14 * radius**2
#     print(f"The area of the circle with radius {radius} is {area:.2f}")
#     miles_speaker(f"The area of the circle with radius {radius} is {area:.2f}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define regex pattern for circle circumference calculation
# regex_circle_circumference = r"^circle\s*(circumference|perimeter)\s*\((?P<radius>-?\d+\.?\d*)\)$"
# # Match input string against pattern
# match = re.match(regex_circle_circumference, usersaid)
# # If input matches pattern
# if match:
#     # Get the radius from the matched pattern
#     radius = float(match.group("radius"))
#     # Calculate the circumference
#     circumference = 2 * 3.14159 * radius
#     # Print the result
#     print(
#         f"The circumference of the circle with radius {radius} is {circumference:.2f}.")
#     miles_speaker(
#         f"The circumference of the circle with radius {radius} is {circumference:.2f}.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define the regex pattern
# regex_pythagorean_theorem = r"^pythagorean\s*theorem\s*\((?P<leg1>-?\d+\.?\d*)\s*,\s*(?P<leg2>-?\d+\.?\d*)\)$"
# # Use re.match to see if the input matches the pattern
# match = re.match(regex_pythagorean_theorem, usersaid)
# # If there is a match, extract the values and calculate the hypotenuse
# if match:
#     # Get the leg lengths from the match object
#     leg1 = float(match.group("leg1"))
#     leg2 = float(match.group("leg2"))
#     # Calculate the hypotenuse
#     hypotenuse = (leg1 ** 2 + leg2 ** 2) ** 0.5
#     # Print the result
#     print(
#         f"The hypotenuse of the right triangle with legs {leg1} and {leg2} is {hypotenuse}.")
#     miles_speaker(
#         f"The hypotenuse of the right triangle with legs {leg1} and {leg2} is {hypotenuse}.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define the regular expression pattern
# regex_triangle_area = r"^triangle\s*area\s*\((?P<base>-?\d+\.?\d*)\s*,\s*(?P<height>-?\d+\.?\d*)\)$"
# # Try to match the user input with the regular expression pattern
# match = re.match(regex_triangle_area, usersaid)
# # If the user input matches the pattern, extract the base and height values and calculate the area
# if match:
#     # Extract the base and height values from the match object
#     base = float(match.group("base"))
#     height = float(match.group("height"))
#     # Calculate the area of the triangle
#     area = 0.5 * base * height
#     # Print the result
#     print(
#         f"The area of the triangle with base {base} and height {height} is {area}")
#     miles_speaker(
#         f"The area of the triangle with base {base} and height {height} is {area}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_triangle_semiperimeter = r"^triangle\s*semiperimeter\s*\((?P<side1>-?\d+\.?\d*)\s*,\s*(?P<side2>-?\d+\.?\d*)\s*,\s*(?P<side3>-?\d+\.?\d*)\)$"
# # Attempt to match the user input against the regular expression pattern
# match = re.match(regex_triangle_semiperimeter, usersaid)
# # If a match is found, perform the calculation and print the result
# if match:
#     # Extract the side lengths from the match object
#     side1 = float(match.group("side1"))
#     side2 = float(match.group("side2"))
#     side3 = float(match.group("side3"))
#     # Calculate the semiperimeter
#     semiperimeter = (side1 + side2 + side3) / 2
#     # Print the result
#     print(
#         f"The semiperimeter of the triangle with side lengths {side1}, {side2}, and {side3} is {semiperimeter}.")
#     miles_speaker(
#         f"The semiperimeter of the triangle with side lengths {side1}, {side2}, and {side3} is {semiperimeter}.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define regex pattern for rectangle area calculation
# regex_rectangle_area = r"^rectangle\s*area\s*\((?P<width>-?\d+\.?\d*)\s*,\s*(?P<height>-?\d+\.?\d*)\)$"
# # Attempt to match regex pattern against user input
# match = re.match(regex_rectangle_area, usersaid)
# # If the regex pattern matches, perform the calculation
# if match:
#     # Extract the width and height from the match object
#     width = float(match.group("width"))
#     height = float(match.group("height"))
#     # Calculate the rectangle area
#     area = width * height
#     # Print the result
#     print(
#         f"The area of the rectangle with width {width} and height {height} is {area}.")
#     miles_speaker(
#         f"The area of the rectangle with width {width} and height {height} is {area}.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regex pattern for Celsius to Fahrenheit conversion
# regex_celsius_to_fahrenheit = r"^(?P<celsius>-?\d+.?\d*)\sC\sto\s*F$"
# # Match the user input with the regex pattern
# match = re.match(regex_celsius_to_fahrenheit, usersaid)
# # If the input matches the pattern, perform the conversion and print the result
# if match:
#     # Get the Celsius temperature from the matched pattern
#     celsius = float(match.group("celsius"))
#     # Convert Celsius to Fahrenheit
#     fahrenheit = celsius * 1.8 + 32
#     # Print the result
#     print(f"{celsius} C to F = {fahrenheit} F")
#     miles_speaker(f"{celsius} C to F = {fahrenheit} F")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define the regular expression pattern for Fahrenheit to Celsius conversion
# regex_fahrenheit_to_celsius = r"^(?P<fahrenheit>-?\d+.?\d*)\sF\sto\s*C$"
# # Match the user input with the regex pattern
# match = re.match(regex_fahrenheit_to_celsius, usersaid)
# # If the input matches the regular expression pattern, extract the Fahrenheit value and calculate Celsius
# if match:
#     # Extract the Fahrenheit value from the match object
#     fahrenheit = float(match.group("fahrenheit"))
#     # Calculate Celsius
#     celsius = (fahrenheit - 32) * 5/9
#     # Print the result
#     print(f"{fahrenheit} F = {celsius:.2f} C")
#     miles_speaker(f"{fahrenheit} F = {celsius:.2f} C")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define regex pattern for kilometers to miles conversion
# regex_kilometers_to_miles = r"^(?P<kilometers>-?\d+.?\d*)\sKM\sto\s*MILES?$"
# # Match the user input against the regex pattern
# match = re.match(regex_kilometers_to_miles, usersaid)
# # Check if the pattern matches
# if match:
#     # Extract the kilometers value from the match object
#     kilometers = float(match.group("kilometers"))
#     # Convert kilometers to miles
#     miles = kilometers * 0.621371
#     # Print the result
#     print(f"{kilometers} kilometers is equal to {miles} miles")
#     miles_speaker(f"{kilometers} kilometers is equal to {miles} miles")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define the regular expression pattern
# regex_miles_to_kilometers = r"^(?P<miles>-?\d+\.?\d*)\sMILES?\sto\s*KM$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_miles_to_kilometers, usersaid)
# # If the input string matches the pattern, perform the conversion
# if match:
#     # Get the miles from the matched pattern
#     miles = float(match.group("miles"))
#     # Perform the conversion from miles to kilometers
#     kilometers = miles * 1.60934
#     # Print the result
#     print(f"{miles} miles is equal to {kilometers:.2f} kilometers.")
#     miles_speaker(f"{miles} miles is equal to {kilometers:.2f} kilometers.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")
# # =============================================================================================================
# # Define the regular expression pattern
# regex_feet_to_meters = r"^(?P<feet>-?\d+\.?\d*)\sFT\sto\s*M$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_feet_to_meters, usersaid)
# # If the input string matches the pattern, perform the conversion
# if match:
#     # Get the feet from the matched pattern
#     feet = float(match.group("feet"))
#     # Perform the conversion from feet to meters
#     meters = feet * 0.3048
#     # Print the result
#     print(f"{feet} feet is equal to {meters:.2f} meters.")
#     miles_speaker(f"{feet} feet is equal to {meters:.2f} meters.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_meters_to_feet = r"^(?P<meters>-?\d+\.?\d*)\sM\sto\s*FT$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_meters_to_feet, usersaid)
# # If the input string matches the pattern, perform the conversion
# if match:
#     # Get the meters from the matched pattern
#     meters = float(match.group("meters"))
#     # Perform the conversion from meters to feet
#     feet = meters * 3.28084
#     # Print the result
#     print(f"{meters} meters is equal to {feet:.2f} feet.")
#     miles_speaker(f"{meters} meters is equal to {feet:.2f} feet.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_inches_to_centimeters = r"^(?P<inches>-?\d+\.?\d*)\sIN(CH)?\sto\s*CM$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_inches_to_centimeters, usersaid)
# # If the input string matches the pattern, perform the conversion
# if match:
#     # Get the inches from the matched pattern
#     inches = float(match.group("inches"))
#     # Perform the conversion from inches to centimeters
#     centimeters = inches * 2.54
#     # Print the result
#     print(f"{inches} inches is equal to {centimeters:.2f} centimeters.")
#     miles_speaker(
#         f"{inches} inches is equal to {centimeters:.2f} centimeters.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_centimeters_to_inches = r"^(?P<centimeters>-?\d+\.?\d*)\sCM\sto\s*IN(CH)?$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_centimeters_to_inches, usersaid)
# # If the input string matches the pattern, perform the conversion
# if match:
#     # Get the centimeters from the matched pattern
#     centimeters = float(match.group("centimeters"))
#     # Perform the conversion from centimeters to inches
#     inches = centimeters / 2.54
#     # Print the result
#     print(f"{centimeters} centimeters is equal to {inches:.2f} inches.")
#     miles_speaker(
#         f"{centimeters} centimeters is equal to {inches:.2f} inches.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_pounds_to_kilograms = r"^(?P<pounds>-?\d+\.?\d*)\sLB(S)?\sto\s*KG$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_pounds_to_kilograms, usersaid)
# # If the input string matches the pattern, perform the conversion
# if match:
#     # Get the pounds from the matched pattern
#     pounds = float(match.group("pounds"))
#     # Perform the conversion from pounds to kilograms
#     kilograms = pounds * 0.453592
#     # Print the result
#     print(f"{pounds} pounds is equal to {kilograms:.2f} kilograms.")
#     miles_speaker(f"{pounds} pounds is equal to {kilograms:.2f} kilograms.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_kilograms_to_pounds = r"^(?P<kilograms>-?\d+\.?\d*)\sKG\sto\s*LB(S)?$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_kilograms_to_pounds, usersaid)
# # If the input string matches the pattern, perform the conversion
# if match:
#     # Get the kilograms from the matched pattern
#     kilograms = float(match.group("kilograms"))
#     # Perform the conversion from kilograms to pounds
#     pounds = kilograms * 2.20462
#     # Print the result
#     print(f"{kilograms} kilograms is equal to {pounds:.2f} pounds.")
#     miles_speaker(f"{kilograms} kilograms is equal to {pounds:.2f} pounds.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_quadratic_equation = r"^(?P<a>-?\d+.?\d*)x\^2\s*[+-]\s*(?P<b>-?\d+.?\d*)x\s*[+-]\s*(?P<c>-?\d+.?\d*)$"

# # Match the input string to the regular expression pattern
# match = re.match(regex_quadratic_equation, usersaid)

# # If the input string matches the pattern, solve the quadratic equation
# if match:
#     # Get the coefficients a, b, and c from the matched pattern
#     a = float(match.group("a"))
#     b = float(match.group("b"))
#     c = float(match.group("c"))

#     # Calculate the discriminant
#     discriminant = b**2 - 4*a*c

#     # If the discriminant is positive, there are two real roots
#     if discriminant > 0:
#         root1 = (-b + math.sqrt(discriminant)) / (2*a)
#         root2 = (-b - math.sqrt(discriminant)) / (2*a)
#         print(f"The roots are {root1:.2f} and {root2:.2f}.")
#         miles_speaker(f"The roots are {root1:.2f} and {root2:.2f}.")
#     # If the discriminant is zero, there is one real root
#     elif discriminant == 0:
#         root = -b / (2*a)
#         print(f"The root is {root:.2f}.")
#         miles_speaker(f"The root is {root:.2f}.")
#     # If the discriminant is negative, there are two complex roots
#     else:
#         real_part = -b / (2*a)
#         imaginary_part = math.sqrt(-discriminant) / (2*a)
#         print(
#             f"The roots are {real_part:.2f} + {imaginary_part:.2f}i and {real_part:.2f} - {imaginary_part:.2f}i.")
#         miles_speaker(
#             f"The roots are {real_part:.2f} + {imaginary_part:.2f}i and {real_part:.2f} - {imaginary_part:.2f}i.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_slope_intercept_form = r"^y\s*[=-]\s*(?P<m>-?\d+\.?\d*)x\s*[+-]\s*(?P<b>-?\d+\.?\d*)$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_slope_intercept_form, usersaid)
# # If the input string matches the pattern, extract the slope and y-intercept
# if match:
#     # Get the slope and y-intercept from the matched pattern
#     slope = float(match.group("m"))
#     y_intercept = float(match.group("b"))
#     # Print the slope-intercept equation
#     print(
#         f"The slope-intercept form of the equation is y = {slope:.2f}x + {y_intercept:.2f}")
#     # Convert to point-slope form
#     point_slope_form = f"y - {y_intercept:.2f} = {slope:.2f}(x - 0)"
#     # Print the point-slope form equation
#     print(f"The point-slope form of the equation is {point_slope_form}")
#     # Convert to standard form
#     standard_form = f"{slope:.2f}x - y + {y_intercept:.2f} = 0"
#     # Print the standard form equation
#     print(f"The standard form of the equation is {standard_form}")
#     miles_speaker(
#         f"The slope-intercept form of the equation is y = {slope:.2f}x + {y_intercept:.2f}. The point-slope form of the equation is {point_slope_form}. The standard form of the equation is {standard_form}.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_point_slope_form = r"^y\s*[=]\s*(?P<m>-?\d+\.?\d*)x\s*[+\-]\s*(?P<b>-?\d+\.?\d*)$"

# # Match the input string to the regular expression pattern
# match = re.match(regex_point_slope_form, usersaid)

# # If the input string matches the pattern, perform the calculation
# if match:
#     # Get the values for m and b from the matched pattern
#     m = float(match.group("m"))
#     b = float(match.group("b"))

#     # Print the equation in slope-intercept form
#     print(f"y = {m}x + {b}")
#     miles_speaker(f"y = {m}x + {b}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_system_of_equations = r"^{\s*(?P<a1>-?\d+.?\d*)x\s*[+-]\s*(?P<b1>-?\d+.?\d*)y\s*[=-]\s*(?P<c1>-?\d+.?\d*),\s*(?P<a2>-?\d+.?\d*)x\s*[+-]\s*(?P<b2>-?\d+.?\d*)y\s*[=-]\s*(?P<c2>-?\d+.?\d*)\s*}$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_system_of_equations, usersaid)
# # If the input string matches the pattern, solve the system of equations
# if match:
#     # Get the coefficients from the matched pattern
#     a1 = float(match.group("a1"))
#     b1 = float(match.group("b1"))
#     c1 = float(match.group("c1"))
#     a2 = float(match.group("a2"))
#     b2 = float(match.group("b2"))
#     c2 = float(match.group("c2"))

#     # Solve the system of equations
#     x = (c1*b2 - c2*b1) / (a1*b2 - a2*b1)
#     y = (a1*c2 - a2*c1) / (a1*b2 - a2*b1)

#     # Print the solution
#     print(f"The solution is x = {x:.2f} and y = {y:.2f}.")
#     miles_speaker(f"The solution is x = {x:.2f} and y = {y:.2f}.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_interest = r"^(?P<p>-?\d+\.?\d*)\s*\times\s*(?P<r>-?\d+\.?\d*)\s*\times\s*(?P<t>-?\d+\.?\d*)$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_interest, usersaid)
# # If the input string matches the pattern, perform the calculation
# if match:
#     # Get the values from the matched pattern
#     p = float(match.group("p"))
#     r = float(match.group("r"))
#     t = float(match.group("t"))
#     # Perform the interest calculation
#     interest = p * r * t / 100
#     # Print the result
#     print(f"The interest is {interest:.2f}")
#     miles_speaker(f"The interest is {interest:.2f}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_distance_formula = r"^distance\sformula\s*(?P<x1>-?\d+\.?\d*)\s*,\s*(?P<y1>-?\d+\.?\d*)\s*,\s*(?P<x2>-?\d+\.?\d*)\s*,\s*(?P<y2>-?\d+\.?\d*)\s*$"

# # Match the input string to the regular expression pattern
# match = re.match(regex_distance_formula, usersaid)

# # If the input string matches the pattern, perform the calculation
# if match:
#     # Extract the coordinates from the match object
#     x1, y1, x2, y2 = map(float, match.groups())
#     # Calculate the distance using the distance formula
#     distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
#     # Print the result
#     print(
#         f"The distance between ({x1}, {y1}) and ({x2}, {y2}) is {distance:.2f}")
#     miles_speaker(
#         f"The distance between ({x1}, {y1}) and ({x2}, {y2}) is {distance:.2f}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_slope_formula = r"^slope\sformula\s∗(?P<x1>-?\d+\.?\d*)\s*,\s*(?P<y1>-?\d+\.?\d*)\s*,\s*(?P<x2>-?\d+\.?\d*)\s*,\s*(?P<y2>-?\d+\.?\d*)\s*$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_slope_formula, usersaid)
# # If the input string matches the pattern, perform the calculation
# if match:
#     # Get the coordinates from the matched pattern
#     x1 = float(match.group("x1"))
#     y1 = float(match.group("y1"))
#     x2 = float(match.group("x2"))
#     y2 = float(match.group("y2"))
#     # Calculate the slope using the coordinates
#     slope = (y2 - y1) / (x2 - x1)
#     # Print the result
#     print(
#         f"The slope of the line passing through ({x1}, {y1}) and ({x2}, {y2}) is {slope:.2f}.")
#     miles_speaker(
#         f"The slope of the line passing through ({x1}, {y1}) and ({x2}, {y2}) is {slope:.2f}.")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_midpoint_formula = r"^midpoint\sformula\s\s∗(?P<x1>−?\d+?˙\d∗)\s∗,\s∗(?P<y1>−?\d+?˙\d∗)\s∗,\s∗(?P<x2>−?\d+?˙\d∗)\s∗,\s∗(?P<y2>−?\d+?˙\d∗)\s∗\s∗(?P<x1>−?\d+?˙\d∗)\s∗,\s∗(?P<y1>−?\d+?˙\d∗)\s∗,\s∗(?P<x2>−?\d+?˙\d∗)\s∗,\s∗(?P<y2>−?\d+?˙\d∗)\s∗$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_midpoint_formula, usersaid)
# # If the input string matches the pattern, perform the calculation
# if match:
#     # Get the coordinates of the endpoints of the line segment
#     x1 = float(match.group("x1"))
#     y1 = float(match.group("y1"))
#     x2 = float(match.group("x2"))
#     y2 = float(match.group("y2"))
#     # Calculate the midpoint of the line segment
#     midpoint_x = (x1 + x2) / 2
#     midpoint_y = (y1 + y2) / 2
#     # Print the result
#     print(
#         f"The midpoint of the line segment with endpoints ({x1}, {y1}) and ({x2}, {y2}) is ({midpoint_x}, {midpoint_y}).")
#     miles_speaker(
#         f"The midpoint of the line segment with endpoints ({x1}, {y1}) and ({x2}, {y2}) is ({midpoint_x}, {midpoint_y}).")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
# # Define the regular expression pattern
# regex_quartic_equation = r"^(?P<a>-?\d+.?\d*)x\^4\s*[+-]\s*(?P<b>-?\d+.?\d*)x\^3\s*[+-]\s*(?P<c>-?\d+.?\d*)x\^2\s*[+-]\s*(?P<d>-?\d+.?\d*)x\s*[+-]\s*(?P<e>-?\d+.?\d*)$"
# # Match the input string to the regular expression pattern
# match = re.match(regex_quartic_equation, usersaid)
# # If the input string matches the pattern, solve the equation
# if match:
#     # Extract the coefficients from the match object
#     a = float(match.group("a"))
#     b = float(match.group("b"))
#     c = float(match.group("c"))
#     d = float(match.group("d"))
#     e = float(match.group("e"))
#     # Solve the quartic equation using numpy.roots()
#     roots = np.roots([a, b, c, d, e])
#     # Print the roots
#     print(
#         f"The roots of the quartic equation {a}x^4 + {b}x^3 + {c}x^2 + {d}x + {e} are: {roots}")
#     miles_speaker(
#         f"The roots of the quartic equation {a}x^4 + {b}x^3 + {c}x^2 + {d}x + {e} are: {roots}")
# else:
#     print("Invalid input format.")
#     miles_speaker("Invalid input format.")

# # =============================================================================================================
