"""Stream-oriented transcription backends for OpenAI and Microsoft Speech."""

from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass
import io
import json
import re
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlsplit
import wave

import httpx
import openai

from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_PROMPT
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryError,
    HomeAssistantError,
)
from homeassistant.helpers.config_entry_oauth2_flow import OAuth2Session
from homeassistant.helpers.httpx_client import get_async_client

from . import create_deployment_client
from .auth import async_get_inference_session
from .const import (
    AUTH_ENTRA_ID,
    CONF_AUTH_TYPE,
    CONF_CHAT_MODEL,
    CONF_MAI_MODEL,
    CONF_SPEECH_ENDPOINT,
    CONF_STT_BACKEND,
    DEFAULT_STT_PROMPT,
    MAI_MODELS,
    RECOMMENDED_MAI_MODEL,
    RECOMMENDED_STT_MODEL,
    STT_BACKEND_MAI_FILE,
    STT_BACKEND_OPENAI,
)

if TYPE_CHECKING:
    from . import OpenAIConfigEntry

_SPEECH_API_VERSION = "2025-10-15"
_MAX_SPEECH_AUDIO_BYTES = 250 * 1024 * 1024
_SPEECH_HOST = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.cognitiveservices\.azure\.com"
)


@dataclass(frozen=True)
class AudioCapabilities:
    """Audio accepted by a particular transcription backend."""

    languages: tuple[str, ...]
    formats: tuple[stt.AudioFormats, ...]
    codecs: tuple[stt.AudioCodecs, ...]
    bit_rates: tuple[stt.AudioBitRates, ...]
    sample_rates: tuple[stt.AudioSampleRates, ...]
    channels: tuple[stt.AudioChannels, ...]


OPENAI_CAPABILITIES = AudioCapabilities(
    languages=(
        "af-ZA",
        "ar-SA",
        "hy-AM",
        "az-AZ",
        "be-BY",
        "bs-BA",
        "bg-BG",
        "ca-ES",
        "zh-CN",
        "hr-HR",
        "cs-CZ",
        "da-DK",
        "nl-NL",
        "en-US",
        "et-EE",
        "fi-FI",
        "fr-FR",
        "gl-ES",
        "de-DE",
        "el-GR",
        "he-IL",
        "hi-IN",
        "hu-HU",
        "is-IS",
        "id-ID",
        "it-IT",
        "ja-JP",
        "kn-IN",
        "kk-KZ",
        "ko-KR",
        "lv-LV",
        "lt-LT",
        "mk-MK",
        "ms-MY",
        "mr-IN",
        "mi-NZ",
        "ne-NP",
        "no-NO",
        "fa-IR",
        "pl-PL",
        "pt-PT",
        "ro-RO",
        "ru-RU",
        "sr-RS",
        "sk-SK",
        "sl-SI",
        "es-ES",
        "sw-KE",
        "sv-SE",
        "fil-PH",
        "ta-IN",
        "th-TH",
        "tr-TR",
        "uk-UA",
        "ur-PK",
        "vi-VN",
        "cy-GB",
    ),
    formats=(stt.AudioFormats.WAV, stt.AudioFormats.OGG),
    codecs=(stt.AudioCodecs.PCM, stt.AudioCodecs.OPUS),
    bit_rates=(
        stt.AudioBitRates.BITRATE_8,
        stt.AudioBitRates.BITRATE_16,
        stt.AudioBitRates.BITRATE_24,
        stt.AudioBitRates.BITRATE_32,
    ),
    sample_rates=(
        stt.AudioSampleRates.SAMPLERATE_8000,
        stt.AudioSampleRates.SAMPLERATE_11000,
        stt.AudioSampleRates.SAMPLERATE_16000,
        stt.AudioSampleRates.SAMPLERATE_18900,
        stt.AudioSampleRates.SAMPLERATE_22000,
        stt.AudioSampleRates.SAMPLERATE_32000,
        stt.AudioSampleRates.SAMPLERATE_37800,
        stt.AudioSampleRates.SAMPLERATE_44100,
        stt.AudioSampleRates.SAMPLERATE_48000,
    ),
    channels=(stt.AudioChannels.CHANNEL_MONO, stt.AudioChannels.CHANNEL_STEREO),
)

_MAI_15_LANGUAGES = (
    "ar-SA",
    "as-IN",
    "bg-BG",
    "bn-IN",
    "ca-ES",
    "cs-CZ",
    "da-DK",
    "de-DE",
    "el-GR",
    "en-US",
    "es-ES",
    "et-EE",
    "fi-FI",
    "fr-FR",
    "gu-IN",
    "hi-IN",
    "hu-HU",
    "id-ID",
    "it-IT",
    "ja-JP",
    "kn-IN",
    "ko-KR",
    "lt-LT",
    "ml-IN",
    "mr-IN",
    "nb-NO",
    "nl-NL",
    "or-IN",
    "pa-IN",
    "pl-PL",
    "pt-PT",
    "ro-RO",
    "ru-RU",
    "sk-SK",
    "sl-SI",
    "sv-SE",
    "ta-IN",
    "te-IN",
    "th-TH",
    "tr-TR",
    "uk-UA",
    "vi-VN",
    "zh-CN",
)
_MAI_2_LANGUAGES = (
    *_MAI_15_LANGUAGES,
    "af-ZA",
    "az-AZ",
    "bs-BA",
    "fa-IR",
    "fil-PH",
    "gl-ES",
    "he-IL",
    "hy-AM",
    "is-IS",
    "kk-KZ",
    "lv-LV",
    "mk-MK",
    "ms-MY",
    "ne-NP",
    "sw-KE",
    "ur-PK",
    "yue-HK",
)


class TranscriptionError(HomeAssistantError):
    """A transcription request or response could not be processed."""


class TranscriptionBackend(Protocol):
    """Consume live audio without imposing buffering on future transports."""

    @property
    def capabilities(self) -> AudioCapabilities:
        """Return the supported audio metadata."""
        ...

    async def async_transcribe(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> str:
        """Return the final transcript for an utterance."""
        ...


def validate_speech_endpoint(endpoint: str) -> str:
    """Accept only resource-root public Azure Speech endpoints for bearer tokens."""
    parsed = urlsplit(endpoint)
    if not (
        parsed.scheme == "https"
        and parsed.hostname
        and _SPEECH_HOST.fullmatch(parsed.hostname)
        and parsed.port in (None, 443)
        and parsed.username is None
        and parsed.password is None
        and parsed.path in ("", "/")
        and not parsed.query
        and not parsed.fragment
    ):
        raise ValueError("Expected an HTTPS Azure Speech resource endpoint")
    return f"https://{parsed.hostname}/"


async def _async_file_audio(
    metadata: stt.SpeechMetadata,
    stream: AsyncIterable[bytes],
    *,
    max_bytes: int | None = None,
) -> bytes:
    """Collect file-based audio and add Home Assistant's missing WAV header."""
    audio = bytearray()
    async for chunk in stream:
        if max_bytes is not None and len(audio) + len(chunk) + 44 > max_bytes:
            raise TranscriptionError("Audio exceeds Azure Speech's file size limit")
        audio.extend(chunk)
    if metadata.format != stt.AudioFormats.WAV:
        return bytes(audio)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(metadata.channel.value)
        wav.setsampwidth(metadata.bit_rate.value // 8)
        wav.setframerate(metadata.sample_rate.value)
        wav.writeframes(audio)
    return buffer.getvalue()


class OpenAIFileTranscriptionBackend:
    """Preserve the existing Azure OpenAI deployment transcription request."""

    capabilities = OPENAI_CAPABILITIES

    def __init__(
        self, client: openai.AsyncOpenAI, deployment: str, prompt: str
    ) -> None:
        """Initialize deployment routing and existing optional prompt."""
        self._client = create_deployment_client(client, deployment)
        self._deployment = deployment
        self._prompt = prompt

    async def async_transcribe(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> str:
        """Collect and transcribe a file using the existing request parameters."""
        audio = await _async_file_audio(metadata, stream)
        response = await self._client.audio.transcriptions.create(
            model=self._deployment,
            file=(f"a.{metadata.format.value}", audio),
            response_format="json",
            language=metadata.language.split("-")[0],
            prompt=self._prompt,
        )
        return response.text


class MAIFileTranscriptionBackend:
    """Transcribe WAV files through Azure Speech using renewable Entra tokens."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        session: OAuth2Session,
        endpoint: str,
        model: str,
    ) -> None:
        """Validate the destination and supported model before reading audio."""
        self._url = f"{validate_speech_endpoint(endpoint)}speechtotext/transcriptions:transcribe"
        if model not in MAI_MODELS:
            raise ValueError("Unsupported MAI file transcription model")
        self._client = client
        self._session = session
        self._model = model
        self.capabilities = AudioCapabilities(
            languages=_MAI_2_LANGUAGES
            if model == "MAI-Transcribe-2"
            else _MAI_15_LANGUAGES,
            formats=(stt.AudioFormats.WAV,),
            codecs=(stt.AudioCodecs.PCM,),
            bit_rates=(stt.AudioBitRates.BITRATE_16,),
            sample_rates=(stt.AudioSampleRates.SAMPLERATE_16000,),
            channels=(stt.AudioChannels.CHANNEL_MONO,),
        )

    async def async_transcribe(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> str:
        """Upload audio and return validated, ordered combined phrases."""
        capabilities = self.capabilities
        if (
            metadata.format not in capabilities.formats
            or metadata.codec not in capabilities.codecs
            or metadata.bit_rate not in capabilities.bit_rates
            or metadata.sample_rate not in capabilities.sample_rates
            or metadata.channel not in capabilities.channels
            or metadata.language not in capabilities.languages
        ):
            raise TranscriptionError("Unsupported audio metadata for MAI transcription")
        audio = await _async_file_audio(
            metadata, stream, max_bytes=_MAX_SPEECH_AUDIO_BYTES
        )
        await self._session.async_ensure_token_valid()
        definition = {
            "enhancedMode": {"enabled": True, "model": self._model},
            "locales": [metadata.language.split("-")[0]],
        }
        try:
            response = await self._client.post(
                self._url,
                params={"api-version": _SPEECH_API_VERSION},
                headers={
                    "Authorization": f"Bearer {self._session.token['access_token']}"
                },
                files={
                    "audio": ("a.wav", audio, "audio/wav"),
                    "definition": (None, json.dumps(definition), "application/json"),
                },
                timeout=60.0,
                follow_redirects=False,
            )
        except httpx.RequestError as err:
            raise TranscriptionError("Unable to connect to Azure Speech") from err
        if response.status_code == 401:
            raise ConfigEntryAuthFailed("Azure Speech rejected the Entra token")
        if response.status_code == 403:
            raise TranscriptionError(
                "Azure Speech access denied; grant Cognitive Services Speech User "
                "access on the Speech resource"
            )
        if response.status_code != 200:
            raise TranscriptionError(
                f"Azure Speech transcription failed with HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except ValueError as err:
            raise TranscriptionError("Azure Speech returned invalid JSON") from err
        if not isinstance(payload, dict) or not isinstance(
            phrases := payload.get("combinedPhrases"), list
        ):
            raise TranscriptionError("Azure Speech returned invalid combined phrases")
        if any(
            not isinstance(phrase, dict) or not isinstance(phrase.get("text"), str)
            for phrase in phrases
        ):
            raise TranscriptionError("Azure Speech returned invalid transcript text")
        return " ".join(phrase["text"] for phrase in phrases)


async def async_create_transcription_backend(
    hass: HomeAssistant, entry: OpenAIConfigEntry, subentry: ConfigSubentry
) -> TranscriptionBackend:
    """Resolve the configured transport without changing the OpenAI runtime client."""
    options = subentry.data
    backend = options.get(CONF_STT_BACKEND, STT_BACKEND_OPENAI)
    if backend == STT_BACKEND_OPENAI:
        return OpenAIFileTranscriptionBackend(
            entry.runtime_data,
            options.get(CONF_CHAT_MODEL, RECOMMENDED_STT_MODEL),
            options.get(CONF_PROMPT, DEFAULT_STT_PROMPT),
        )
    if backend != STT_BACKEND_MAI_FILE:
        raise ConfigEntryError("Unsupported speech-to-text backend")
    if entry.data.get(CONF_AUTH_TYPE) != AUTH_ENTRA_ID:
        raise ConfigEntryError("MAI transcription requires Microsoft Entra ID")
    try:
        endpoint = validate_speech_endpoint(options.get(CONF_SPEECH_ENDPOINT, ""))
        model = options.get(CONF_MAI_MODEL, RECOMMENDED_MAI_MODEL)
        if model not in MAI_MODELS:
            raise ValueError("Unsupported MAI file transcription model")
    except ValueError as err:
        raise ConfigEntryError("Invalid MAI model or Azure Speech endpoint") from err
    session = await async_get_inference_session(hass, entry)
    return MAIFileTranscriptionBackend(get_async_client(hass), session, endpoint, model)
