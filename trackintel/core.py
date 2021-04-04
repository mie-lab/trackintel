from trackintel import __version__


def print_version():
    """Print the framework version."""

    print(
        "This is trackintel v"
        + str(__version__.__version__)
        + ". You can find more information "
        + "under https://github.com/mie-lab/trackintel. Thank you for using it!"
    )
