"""Context-local managed inputs for explicitly requested scientific replay."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field

from config.crm_schema import CURRENT_SCHEMA_VERSION
from config.scientific_identity import digest


@dataclass
class RecordedDependencies:
    reference_library: dict
    custom_contributors: dict
    profile_defaults: dict = field(default_factory=dict)
    database: dict | None = field(default=None, init=False, repr=False)
    generation: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.reference_library, dict) or self.reference_library.get('version') != CURRENT_SCHEMA_VERSION:
            raise ValueError('Recorded reference library requires the current explicit schema')
        if not isinstance(self.reference_library.get('elements'), dict):
            raise ValueError('Recorded reference library requires an elements object')
        if not isinstance(self.custom_contributors, dict):
            raise ValueError('Recorded custom contributors must be an explicit object')
        if not isinstance(self.profile_defaults, dict) or any(
            not isinstance(values, dict) or any(type(value) is not bool for value in values.values())
            for values in self.profile_defaults.values()
        ):
            raise ValueError('Recorded profile defaults must contain explicit Boolean states')
        self.reference_library = deepcopy(self.reference_library)
        self.custom_contributors = deepcopy(self.custom_contributors)
        self.profile_defaults = deepcopy(self.profile_defaults)
        self.generation = -1 - int(digest(self.reference_library), 16)


_active = ContextVar('traceiso_recorded_dependencies', default=None)


def current_dependencies():
    return _active.get()


@contextmanager
def use_recorded_dependencies(dependencies):
    """Isolate both inputs and derived CRM cache from other calls/sessions."""
    local = RecordedDependencies(dependencies.reference_library, dependencies.custom_contributors, dependencies.profile_defaults)
    token = _active.set(local)
    try:
        yield
    finally:
        _active.reset(token)


def recorded_library_generation(dependencies):
    # Negative digest keys cannot collide with the installed library's positive
    # monotonic generations, including across nested replay contexts.
    return dependencies.generation
