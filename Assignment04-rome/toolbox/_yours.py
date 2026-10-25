"""Your 9 functions live in the notebook, not here. The toolbox finds them there by name, at the moment it needs them,
so a check always uses the version you ran last (or ref.<name> if you borrowed it: tfidf_matrix = ref.tfidf_matrix)."""

import sys


class NotDefinedYet(NameError):
    """Raised when the toolbox needs one of your functions and the notebook has not defined it yet."""


def yours(name):
    """Your function called name, as defined in the notebook (the __main__ namespace)."""
    f = getattr(sys.modules.get("__main__"), name, None)
    if not callable(f):
        raise NotDefinedYet(f"not defined yet: {name}. Run the cell that defines it.")
    return f
