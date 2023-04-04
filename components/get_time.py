import datetime


def get_time(request):
    current_time = datetime.datetime.now()
    if "hour" in request and "minute" in request:
        return current_time.strftime("%I:%M %p")
    elif "hour" in request:
        return current_time.strftime("%I %p")
    elif "minute" in request:
        return current_time.strftime("%M minutes past %I %p")
    else:
        return "Sorry, I didn't understand your request. Please try again."
