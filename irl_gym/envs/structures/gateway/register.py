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

from typing import Any, Callable, Iterable, Sequence



def register():
    # For now let external envs register with their own register function. Users should already be familiar with these if they want to use a specific standard. 
    # just register internal models here.
    pass

def make(id: str | EnvSpec,
         max_episode_steps: int | None = None,
         disable_env_checker: bool | None = None,
         **kwargs: Any):
    
    #call appropriate make functions. -> encode gym type using keys. 
    pass


def parse_id():
    #parse id into components
    pass