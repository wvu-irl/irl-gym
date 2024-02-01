"""
This module contains the model for Stickbug (and the adjudication system)
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from copy import deepcopy
import logging

class Adjudicator:
    def __init__(self):
        pass

class Stickbug:
    def __init__(self, base, arms):
        pass