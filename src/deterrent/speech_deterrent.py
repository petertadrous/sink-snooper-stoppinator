import json
from pathlib import Path
import random
from typing import Optional
import traceback

import pyttsx3
from langchain_ollama import OllamaLLM

from src.deterrent._deterrent import Deterrent
from src.utils.logger import logger

with open(Path("assets/phrases.json")) as f:
    PHRASE_LOOKUP = json.load(f)

ALL_PHRASES = list(
    set(str.lower(p) for phrases in PHRASE_LOOKUP.values() for p in phrases)
)


class SpeechProvider:
    def __init__(self, category: str = "any") -> None:
        if category == "any":
            logger.debug("No category selected, allowing any category")
            self.category = None
            self.phrases = ALL_PHRASES
        else:
            self.category = str.lower(category)
            logger.debug(f"Selected category: {self.category}")
            self.phrases = list(set(PHRASE_LOOKUP[self.category]))

    def get_phrase(self) -> str:
        return random.choice(self.phrases)


class SpeechDeterrent(Deterrent):
    def __init__(
        self,
        creative: bool = False,
        category: str = "any",
        voice: Optional[str] = "com.apple.voice.compact.en-US.Samantha",
    ) -> None:
        self.engine: Optional[pyttsx3.Engine] = None
        self.llm = None
        self.creative = creative
        self.phrase_provider = None
        self.category = category
        self.prompt = None
        self.voice = voice

    def setup(self):
        """Initialize text-to-speech engine and load needed resources."""
        try:
            self.engine = pyttsx3.init()
            self._select_voice()  # Initialize with default voice
            logger.debug("Speech engine initialized")
            self.engine.startLoop(False)

            if self.creative:
                # Initialize LLM for creative mode
                logger.debug("Using creative mode")
                self.llm = OllamaLLM(model="HammerAI/openhermes-2.5-mistral")  # type: ignore
                with open("assets/creative_prompt.txt", "r") as f:
                    self.prompt = f.read()
            else:
                logger.debug("Using basic mode")
                self.phrase_provider = SpeechProvider(category=self.category)

        except Exception as e:
            logger.error(f"Failed to initialize SpeechDeterrent: {e}")
            self.cleanup()
            raise

    def _select_voice(self) -> str:
        """Select a suitable voice for the text-to-speech engine."""
        if not self.engine:
            raise RuntimeError("Engine not initialized")

        voices = self.engine.getProperty("voices")

        # If a specific voice is requested, try to use it
        if self.voice:
            for voice in voices:
                if voice.id == self.voice:
                    self.engine.setProperty("voice", voice.id)
                    return voice.id

        # Fall back to finding any English voice if specified voice not found
        english_voices = [
            v for v in voices if "en_US" in v.languages or "en-US" in v.languages
        ]

        if not english_voices:
            logger.error("No suitable English voices found. Available voices:")
            for voice in voices:
                logger.error(f" - {voice.id} ({voice.name})")
            raise RuntimeError("No suitable English voices found")

        selected_voice = english_voices[0]
        self.engine.setProperty("voice", selected_voice.id)
        return selected_voice.id

    def _run_basic(self, duration: float):
        """Run basic mode with predefined phrases."""
        if self.phrase_provider is None:
            raise RuntimeError("Phrase provider not initialized")
        if not self.engine:
            raise RuntimeError("Speech engine not initialized")

        try:
            phrase = self.phrase_provider.get_phrase()
            logger.debug(f"Basic mode phrase: {phrase}")
            self.engine.say(phrase)
            self.engine.iterate()
        except Exception as e:
            logger.error(f"Error getting/speaking phrase: {e}")
            logger.debug(traceback.format_exc())
            self.cleanup()
            raise

    def _run_creative(self, duration: float):
        """Run creative mode using LLM."""
        if not self.llm or not self.prompt or not self.engine:
            raise RuntimeError("LLM components not initialized")

        try:
            response = self.llm.invoke(self.prompt)
            logger.debug(f"Creative mode phrase: {response}")
            self.engine.say(response)
            self.engine.iterate()
        except Exception as e:
            logger.error(f"Error getting/speaking creative response: {e}")
            logger.debug(traceback.format_exc())
            self.cleanup()
            raise

    def activate(self, duration: float):
        """Activates the deterrent for the specified duration."""
        if self.creative:
            self._run_creative(duration)
        else:
            self._run_basic(duration)

    def cleanup(self):
        """Clean up speech engine resources"""
        if self.llm:
            self.llm = None
        if self.engine:
            try:
                self.engine.stop()
            except Exception as e:
                logger.error(f"Error during speech engine cleanup: {e}")
                logger.debug(traceback.format_exc())
            self.engine = None
