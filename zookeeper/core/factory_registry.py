"""Keep track of user-defined @factory classes.

This is defined here only to avoid import loops.
"""

from typing import Dict, Set, Type

FACTORY_REGISTRY: Dict[Type, Set] = {}
