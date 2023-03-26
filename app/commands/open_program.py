import os


def open_program(program_name):
    try:
        if not os.path.isfile(program_name):
            raise ValueError(f"{program_name} is not a valid file path.")

        os.startfile(program_name)

    except ValueError as ve:
        error_msg = f"An error occurred while opening {program_name}: {str(ve)}"
        print(error_msg)
        raven_speaker(error_msg)

    except Exception as e:
        error_msg = f"An error occurred while opening {program_name}: {str(e)}"
        print(error_msg)
        raven_speaker(error_msg)
