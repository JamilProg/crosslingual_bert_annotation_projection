from dataclasses import dataclass


@dataclass
class ParsingException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
