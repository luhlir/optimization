from enum import Enum
from backtrackSearch import BacktrackSearch
from decayingSearch import DecayingSearch
from fixedSearch import FixedSearch
from fullSearch import FullSearch
from strongBacktrackSearch import StrongBacktrackSearch


class LineSearchEnum(Enum):
    BACKTRACK_SEARCH = BacktrackSearch
    DECAYING_SEARCH = DecayingSearch
    FIXED_SEARCH = FixedSearch
    FULL_SEARCH = FullSearch
    STRONG_BACKTRACK_SEARCH = StrongBacktrackSearch
