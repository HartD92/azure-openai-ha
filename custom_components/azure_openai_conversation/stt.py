"""Speech to text support for Azure OpenAI."""

from __future__ import annotations

from collections.abc import AsyncIterable
import logging
from typing import TYPE_CHECKING, override

from openai import OpenAIError

from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    OAuth2TokenRequestConnectionError,
    OAuth2TokenRequestError,
    OAuth2TokenRequestReauthError,
)
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .entity import AzureOpenAIBaseLLMEntity
from .transcription import (
    TranscriptionBackend,
    TranscriptionError,
    async_create_transcription_backend,
)

if TYPE_CHECKING:
    from . import OpenAIConfigEntry

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up STT entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "stt":
            continue

        async_add_entities(
            [
                AzureOpenAISTTEntity(
                    config_entry,
                    subentry,
                    await async_create_transcription_backend(
                        hass, config_entry, subentry
                    ),
                )
            ],
            config_subentry_id=subentry.subentry_id,
        )


class AzureOpenAISTTEntity(stt.SpeechToTextEntity, AzureOpenAIBaseLLMEntity):
    """Azure OpenAI Speech to text entity."""

    def __init__(
        self,
        entry: OpenAIConfigEntry,
        subentry: ConfigSubentry,
        backend: TranscriptionBackend,
    ) -> None:
        """Bind an STT entity to a stream-oriented transcription backend."""
        super().__init__(entry, subentry)
        self._backend = backend

    @property
    @override
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return list(self._backend.capabilities.languages)

    @property
    @override
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return a list of supported formats."""
        return list(self._backend.capabilities.formats)

    @property
    @override
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return a list of supported codecs."""
        return list(self._backend.capabilities.codecs)

    @property
    @override
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return a list of supported bit rates."""
        return list(self._backend.capabilities.bit_rates)

    @property
    @override
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return a list of supported sample rates."""
        return list(self._backend.capabilities.sample_rates)

    @property
    @override
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return a list of supported channels."""
        return list(self._backend.capabilities.channels)

    @override
    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Process an audio stream to STT service."""
        try:
            text = await self._backend.async_transcribe(metadata, stream)
        except OAuth2TokenRequestReauthError:
            # OAuth2Session already starts reauthentication for rejected refresh grants.
            _LOGGER.error(
                "STT OAuth token refresh was rejected; Microsoft sign-in is required"
            )
        except ConfigEntryAuthFailed as err:
            _LOGGER.error("STT authentication failed: %s", err)
            self.entry.async_start_reauth(self.hass)
        except OAuth2TokenRequestError, OAuth2TokenRequestConnectionError:
            _LOGGER.error("Unable to renew the STT authentication token")
        except TranscriptionError as err:
            _LOGGER.error("Error during STT: %s", err)
        except OpenAIError:
            _LOGGER.exception("Error during STT")
        else:
            if text:
                return stt.SpeechResult(
                    text,
                    stt.SpeechResultState.SUCCESS,
                )
            _LOGGER.warning("STT returned an empty transcript")

        return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
