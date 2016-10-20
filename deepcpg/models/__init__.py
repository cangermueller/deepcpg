from .utils import *

from . import dna
from . import cpg
from . import joint


def get_class(name):
    _name = name.lower()
    if _name == 'dna01':
        return dna.Dna01
    elif _name == 'dna02':
        return dna.Dna02
    elif _name == 'dna03':
        return dna.Dna03
    elif _name == 'dna04':
        return dna.Dna04


    elif _name == 'cpg01':
        return cpg.Cpg01
    elif _name == 'cpg02':
        return cpg.Cpg02
    elif _name == 'cpg03':
        return cpg.Cpg03
    elif _name == 'cpg04':
        return cpg.Cpg04
    elif _name == 'joint01':
        return joint.Joint01
    else:
        raise ValueError('Invalid model "%s"!' % name)
