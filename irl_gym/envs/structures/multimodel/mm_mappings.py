"""
This module contains the GridworldEnv for discrete path planning
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

#assumption mappings

#foreach assumption in model to next, find mapping

#Should let mappings define between different models

#treat mappings as a graph, so each model should have a list of other models for which it retains a mapping


#How to map observations between full and partial observability? need access to a state function...?

#How to map actions between different models? Need to decide whether to map action down to nearest action or policy

#states map to states
