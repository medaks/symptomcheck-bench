from enum import Enum


class Lang(Enum):
    UNKNOWN = 0
    ENGLISH = 1
    SLOVENE = 2
    FRENCH = 3
    DUTCH = 4
    ITALIAN = 5
    SPANISH = 6
    GERMAN = 7
    RUSSIAN = 8
    PORTUGESE = 9
    POLISH = 10
    HUNGARIAN = 11


class Role(Enum):
    ASSISTANT = "ASSISTANT"
    FUNCTION = "FUNCTION"
    SYSTEM = "SYSTEM"
    USER = "USER"
