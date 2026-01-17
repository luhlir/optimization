from enum import Enum
from bracketMinimum import BracketMinimum
from goldenSectionSearch import GoldenSectionSearch
from quadraticFitSearch import QuadraticFitSearch


class BracketEnum(Enum):
    BRACKET_MINIMUM = BracketMinimum
    GOLDEN_SECTION_SEARCH = GoldenSectionSearch
    QUADRATIC_FIT_SEARCH = QuadraticFitSearch
