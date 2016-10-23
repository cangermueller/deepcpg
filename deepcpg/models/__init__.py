from .utils import *

from . import dna
from . import cpg
from . import joint

from ..utils import get_from_module


def get_class(name):
    _name = name.lower()
    if _name.startswith('dna'):
        return get_from_module(name, vars(dna))
    elif _name.startswith('cpg'):
        return get_from_module(name, vars(cpg))
    else:
        return get_from_module(name, vars(joint))
