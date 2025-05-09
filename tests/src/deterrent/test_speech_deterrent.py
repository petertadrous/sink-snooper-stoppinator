import pytest
from unittest.mock import patch, MagicMock
from src.deterrent.speech_deterrent import SpeechDeterrent, SpeechProvider


@pytest.fixture
def speech_deterrent():
    return SpeechDeterrent(category="any", creative=False)


@pytest.fixture
def basic_speech():
    return SpeechDeterrent(category="any", creative=False)


@pytest.fixture
def creative_speech():
    return SpeechDeterrent(category="any", creative=True)


def _mock_engine_with_voice():
    mock_engine = MagicMock()
    mock_voice = MagicMock()
    mock_voice.languages = ["en_US"]
    mock_voice.id = "en_US_voice"
    mock_engine.getProperty.return_value = [mock_voice]
    return mock_engine


def test_speech_deterrent_setup(speech_deterrent):
    with patch("src.deterrent.speech_deterrent.pyttsx3.init") as mock_init:
        mock_engine = _mock_engine_with_voice()
        mock_init.return_value = mock_engine
        speech_deterrent.setup()
        mock_init.assert_called_once()
        mock_engine.setProperty.assert_called()


def test_speech_deterrent_activate_basic(speech_deterrent):
    with (
        patch(
            "src.deterrent.speech_deterrent.SpeechProvider.get_phrase",
            return_value="mock_phrase",
        ) as mock_get_phrase,
        patch("src.deterrent.speech_deterrent.pyttsx3.init") as mock_init,
    ):
        mock_engine = _mock_engine_with_voice()
        mock_init.return_value = mock_engine
        speech_deterrent.setup()
        speech_deterrent.activate(duration=1.5)
        mock_get_phrase.assert_called()
        mock_engine.say.assert_called_with("mock_phrase")


def test_speech_deterrent_cleanup(speech_deterrent):
    with patch("src.deterrent.speech_deterrent.pyttsx3.init") as mock_init:
        mock_engine = _mock_engine_with_voice()
        mock_init.return_value = mock_engine
        speech_deterrent.setup()
        speech_deterrent.cleanup()
        mock_engine.stop.assert_called_once()


def test_select_voice_success(monkeypatch, basic_speech):
    mock_engine = _mock_engine_with_voice()
    basic_speech.engine = mock_engine
    assert basic_speech._select_voice() == "en_US_voice"


def test_select_voice_failure(monkeypatch, basic_speech):
    mock_engine = MagicMock()
    mock_voice = MagicMock()
    mock_voice.languages = ["fr_FR"]
    mock_engine.getProperty.return_value = [mock_voice]
    basic_speech.engine = mock_engine
    with pytest.raises(RuntimeError):
        basic_speech._select_voice()


def test_setup_creative(monkeypatch, creative_speech):
    with patch("src.deterrent.speech_deterrent.pyttsx3.init") as mock_init:
        mock_engine = _mock_engine_with_voice()
        mock_init.return_value = mock_engine
        with patch("src.deterrent.speech_deterrent.OllamaLLM") as mock_llm:
            mock_llm.return_value = MagicMock()
            creative_speech.setup()
            assert creative_speech.llm is not None
            assert creative_speech.engine is not None


def test_activate_creative(monkeypatch, creative_speech):
    with (
        patch("src.deterrent.speech_deterrent.pyttsx3.init") as mock_init,
        patch("src.deterrent.speech_deterrent.OllamaLLM") as mock_llm,
    ):
        mock_engine = _mock_engine_with_voice()
        mock_init.return_value = mock_engine
        mock_llm_inst = MagicMock()
        mock_llm_inst.invoke.return_value = "creative response"
        mock_llm.return_value = mock_llm_inst
        creative_speech.setup()
        creative_speech.engine = mock_engine
        creative_speech.llm = mock_llm_inst
        creative_speech.prompt = "prompt"
        creative_speech.activate(1.0)
        mock_llm_inst.invoke.assert_called()
        mock_engine.say.assert_called_with("creative response")


def test_activate_basic_error(monkeypatch, basic_speech):
    with patch("src.deterrent.speech_deterrent.pyttsx3.init") as mock_init:
        mock_engine = _mock_engine_with_voice()
        mock_init.return_value = mock_engine
        basic_speech.setup()
        basic_speech.engine = mock_engine
        basic_speech.provider = MagicMock()
        basic_speech.provider.get_phrase.side_effect = Exception("fail")
        basic_speech.activate(1.0)  # Should log error, not raise


def test_cleanup_error(monkeypatch, basic_speech):
    with patch("src.deterrent.speech_deterrent.pyttsx3.init") as mock_init:
        mock_engine = _mock_engine_with_voice()
        mock_engine.stop.side_effect = Exception("fail")
        mock_init.return_value = mock_engine
        basic_speech.setup()
        basic_speech.engine = mock_engine
        basic_speech.cleanup()  # Should log error, not raise


def test_speech_provider_any():
    p = SpeechProvider(category="any")
    assert p.category is None
    assert isinstance(p.get_phrase(), str)


def test_speech_provider_specific():
    p = SpeechProvider(category="asian")
    assert p.category == "asian"
    assert isinstance(p.get_phrase(), str)
