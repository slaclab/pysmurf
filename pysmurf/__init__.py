'''
init for root python directory
'''
from .base.smurf_control import SmurfControl

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
