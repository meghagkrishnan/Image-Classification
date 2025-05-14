import os


try:
    SRCPATH: str = os.path.dirname(os.path.abspath(__file__))
    HEADPATH: str = os.path.split(SRCPATH)[0]

except NameError:
    # If we execute __init__.py (e.g. in setup.py) __file__ is not set
    SRCPATH = None
    HEADPATH = None