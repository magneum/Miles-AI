import datetime
# =============================================================================================================


def get_date(user_input):
    today = datetime.date.today()

    if 'today' in user_input:
        return today.strftime("%B %d, %Y")
    elif 'tomorrow' in user_input:
        tomorrow = today + datetime.timedelta(days=1)
        return tomorrow.strftime("%B %d, %Y")
    elif 'yesterday' in user_input:
        yesterday = today - datetime.timedelta(days=1)
        return yesterday.strftime("%B %d, %Y")
    else:
        try:
            date = datetime.datetime.strptime(user_input, '%Y-%m-%d').date()
            return date.strftime("%B %d, %Y")
        except ValueError:
            return "Sorry, I don't understand. Please enter a valid date in YYYY-MM-DD format or ask for today, tomorrow, or yesterday."
# =============================================================================================================
