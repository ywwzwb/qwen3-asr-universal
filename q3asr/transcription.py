"""转录引擎抽象接口(v2 可换 llama.cpp 原生引擎)。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AlignItem:
    text: str
    start: float
    end: float


@dataclass
class TranscribeResult:
    text: str = ""
    alignment: list[AlignItem] | None = None
    performance: dict = field(default_factory=dict)


class TranscriptionEngine(ABC):
    @abstractmethod
    def transcribe(self, audio_file: str, language=None, context="",
                   start_second=0.0, duration=None, temperature=0.4) -> TranscribeResult:
        ...
