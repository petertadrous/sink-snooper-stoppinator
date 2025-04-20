import json
from pathlib import Path
import random
import math
from typing import Optional

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
        category: str = "any",
        creative: bool = False,
        voice: Optional[str] = "com.apple.voice.compact.en-US.Samantha",
    ) -> None:
        self.category = category
        self.creative = creative
        self.llm = None
        self.engine = None
        self.provider = None
        if voice:
            self._voice_selection = voice
        else:
            self._voice_selection = "random"

    def setup(self):
        """
        Initializes the SpeechDeterrent, including text-to-speech engine and optional LLM.
        """
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("voice", self._select_voice())
            # self.engine.setProperty("rate", 150)
            self.provider = SpeechProvider(category=self.category)
            self.engine.startLoop(False)

            if self.creative:
                logger.debug("Using creative mode")
                self.llm = OllamaLLM(model="HammerAI/openhermes-2.5-mistral")  # type: ignore
                with open("assets/creative_prompt.txt", "r") as f:
                    self.prompt = f.read()
        except Exception as e:
            logger.error(f"Failed to initialize SpeechDeterrent: {e}")
            raise

    def _select_voice(self):
        """
        Selects a suitable voice for the text-to-speech engine.
        """
        if self._voice_selection != "random":
            return self._voice_selection
        assert self.engine is not None, "Engine not initialized"
        voices = self.engine.getProperty("voices")
        voices = [voice for voice in voices if "en_US" in voice.languages]
        voices = [
            voice
            for voice in voices
            if voice.gender in ["VoiceGenderMale", "VoiceGenderFemale"]
        ]
        if not voices:
            raise RuntimeError("No suitable voices found for en_US language.")
        voice = random.choice(voices).id
        logger.debug(f"Selected voice: {voice}")
        return voice

    def _activate_basic(self, duration: float) -> None:
        """
        Activates the basic deterrent mode by speaking phrases.
        """
        assert self.engine is not None, "Engine not initialized"
        assert self.provider is not None, "Provider not initialized"
        phrases_to_say = math.ceil(duration)
        for _ in range(phrases_to_say):
            try:
                phrase = self.provider.get_phrase()
                logger.debug(f"Phrase: {phrase}")
                self.engine.say(phrase)
                self.engine.iterate()
            except Exception as e:
                logger.error(f"Error during basic activation: {e}")

    def _activate_creative(self, duration: float):
        """
        Activates the creative deterrent mode using the LLM.
        """
        assert self.llm is not None, "LLM not initialized"
        assert self.engine is not None, "Engine not initialized"
        if not self.llm:
            raise RuntimeError("LLM is not initialized for creative mode.")

        try:
            response = self.llm.invoke(self.prompt)
            logger.debug(f"Response: {response}")
            self.engine.say(response)
            self.engine.iterate()
        except Exception as e:
            logger.error(f"Error during creative activation: {e}")

    def activate(self, duration: float):
        """
        Activates the deterrent for the specified duration.
        """
        try:
            if self.creative:
                self._activate_creative(duration)
            else:
                self._activate_basic(duration)
        except Exception as e:
            logger.error(f"Failed to activate SpeechDeterrent: {e}")

    def cleanup(self) -> None:
        """
        Cleans up resources used by the SpeechDeterrent.
        """
        try:
            if self.engine:
                self.engine.stop()
            self.engine = None
            self.provider = None
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
