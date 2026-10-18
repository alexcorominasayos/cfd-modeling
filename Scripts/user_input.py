"""Questions asked to the user. Every answer is validated before it is returned."""

from config import FLUIDS, SPECIAL_CASES, make_normal_case

MAX_ATTEMPTS = 3


def ask_choice(question, valid_answers):
    """Repeat the question until the answer is one of valid_answers.

    The answer is converted to lowercase, so 'Water' and 'water' are the same.
    The program exits if the user fails MAX_ATTEMPTS times or there is no input.
    """
    for _ in range(MAX_ATTEMPTS):
        try:
            answer = input(question).strip().lower()
        except EOFError:
            raise SystemExit("No input available. Exiting.")
        if answer in valid_answers:
            return answer
        print(f"Invalid option. Choose one of: {', '.join(valid_answers)}")
    raise SystemExit("Too many invalid answers. Exiting.")


def ask_case():
    """Ask what the user wants to see and return the matching case dictionary.

    Water and lava also offer a special case. In the normal case the program
    asks if the temperature must include the viscous heating term.
    """
    fluid_name = ask_choice(f"Choose fluid ({', '.join(FLUIDS)}): ", list(FLUIDS))

    special_case = SPECIAL_CASES.get(fluid_name)
    if special_case is not None:
        answer = ask_choice(
            f"Show the special case '{special_case['label']}' instead of the normal one? (y/n): ",
            ["y", "n"],
        )
        if answer == "y":
            return special_case

    answer = ask_choice("Include the viscous heating term? (y/n): ", ["y", "n"])
    return make_normal_case(fluid_name, answer == "y")
