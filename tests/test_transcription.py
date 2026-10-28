"""Verify file transports and the stream-oriented STT backend boundary."""

from collections.abc import AsyncGenerator
import asyncio
from email import policy
from email.parser import BytesParser
import io
from importlib import import_module
import json
from unittest.mock import patch
import wave

import httpx
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from respx import MockRouter

from homeassistant.components import stt
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryError
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers import entity_registry as er

from custom_components.azure_openai_conversation import (
    async_create_entry_client,
    create_client,
)
from custom_components.azure_openai_conversation.auth import (
    EntraOAuth2Implementation,
    async_get_inference_session,
)
from custom_components.azure_openai_conversation.const import (
    CONF_MAI_MODEL,
    CONF_SPEECH_ENDPOINT,
    CONF_STT_BACKEND,
    DOMAIN,
    STT_BACKEND_MAI_FILE,
)
from custom_components.azure_openai_conversation.stt import AzureOpenAISTTEntity
from custom_components.azure_openai_conversation.transcription import (
    MAIFileTranscriptionBackend,
    OPENAI_CAPABILITIES,
    OpenAIFileTranscriptionBackend,
    TranscriptionError,
    async_create_transcription_backend,
    validate_speech_endpoint,
)

from .const import ENDPOINT, TENANT_ID

SPEECH_ENDPOINT = "https://test-account.cognitiveservices.azure.com/"
TRANSCRIBE_URL = f"{SPEECH_ENDPOINT}speechtotext/transcriptions:transcribe"
PCM_AUDIO = b"\0\0" * 32


@pytest.fixture
def metadata() -> stt.SpeechMetadata:
    """Provide the native Assist mono PCM format without real audio recordings."""
    return stt.SpeechMetadata(
        language="en-US",
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )


async def audio() -> AsyncGenerator[bytes]:
    """Yield multiple chunks of synthetic audio."""
    yield PCM_AUDIO[:32]
    yield PCM_AUDIO[32:]


@pytest.fixture
def mai_entry(hass: HomeAssistant, oauth_data: dict) -> MockConfigEntry:
    """Create an Entra entry with one manually configured MAI STT subentry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=oauth_data,
        version=2,
        minor_version=2,
        subentries_data=[
            {
                "subentry_type": "stt",
                "title": "MAI transcription",
                "unique_id": None,
                "data": {
                    CONF_STT_BACKEND: STT_BACKEND_MAI_FILE,
                    CONF_MAI_MODEL: "MAI-Transcribe-2",
                    CONF_SPEECH_ENDPOINT: SPEECH_ENDPOINT,
                },
            }
        ],
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
async def mai_backend(
    hass: HomeAssistant,
    mai_entry: MockConfigEntry,
    oauth_implementation: EntraOAuth2Implementation,
) -> MAIFileTranscriptionBackend:
    """Use a native renewable OAuth session and Home Assistant HTTP client."""
    return MAIFileTranscriptionBackend(
        get_async_client(hass),
        await async_get_inference_session(hass, mai_entry),
        SPEECH_ENDPOINT,
        "MAI-Transcribe-2",
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://test-account.cognitiveservices.azure.com",
        "https://attacker.example",
        "https://test-account.cognitiveservices.azure.com.attacker.example",
        "https://test-account.openai.azure.com",
        "https://test-account.services.ai.azure.com",
        "https://user:password@test-account.cognitiveservices.azure.com",
        "https://test-account.cognitiveservices.azure.com:8443",
        "https://test-account.cognitiveservices.azure.com:invalid",
        "https://test-account.cognitiveservices.azure.com/path",
        "https://test-account.cognitiveservices.azure.com?query=1",
        "https://test-account.cognitiveservices.azure.com#fragment",
        "https://.cognitiveservices.azure.com",
        "https://bad%20host.cognitiveservices.azure.com",
        "https://-invalid.cognitiveservices.azure.com",
        "https://nested.resource.cognitiveservices.azure.com",
        "",
    ],
)
def test_invalid_speech_endpoint(endpoint: str) -> None:
    """Never send bearer credentials to unsupported or nonroot endpoints."""
    with pytest.raises(ValueError):
        validate_speech_endpoint(endpoint)


def test_speech_endpoint_normalization() -> None:
    """Normalize HTTPS resource roots without guessing a different hostname."""
    assert validate_speech_endpoint(SPEECH_ENDPOINT[:-1]) == SPEECH_ENDPOINT
    assert (
        validate_speech_endpoint(
            "https://TEST-ACCOUNT.cognitiveservices.azure.com:443/"
        )
        == SPEECH_ENDPOINT
    )


async def test_mai_request_and_transcript(
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
    respx_mock: MockRouter,
) -> None:
    """Post exact Speech fields, ordered text, and a valid WAV with a bearer token."""
    route = respx_mock.post(TRANSCRIBE_URL).respond(
        json={"combinedPhrases": [{"text": "Turn on"}, {"text": "the lights."}]}
    )
    assert (
        await mai_backend.async_transcribe(metadata, audio()) == "Turn on the lights."
    )
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer inference-token"
    assert "api-key" not in request.headers
    assert "Ocp-Apim-Subscription-Key" not in request.headers
    assert dict(request.url.params) == {"api-version": "2025-10-15"}
    assert request.extensions["timeout"]["read"] == 60.0
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {request.headers['Content-Type']}\r\n\r\n".encode()
        + request.content
    )
    parts = {
        part.get_param("name", header="content-disposition"): part
        for part in message.iter_parts()
    }
    assert set(parts) == {"audio", "definition"}
    assert json.loads(parts["definition"].get_payload(decode=True)) == {
        "enhancedMode": {"enabled": True, "model": "MAI-Transcribe-2"},
        "locales": ["en"],
    }
    with wave.open(io.BytesIO(parts["audio"].get_payload(decode=True))) as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.readframes(wav.getnframes()) == PCM_AUDIO


@pytest.mark.parametrize("audio_format", [stt.AudioFormats.WAV, stt.AudioFormats.OGG])
async def test_openai_request_unchanged(
    hass: HomeAssistant,
    metadata: stt.SpeechMetadata,
    respx_mock: MockRouter,
    audio_format: stt.AudioFormats,
) -> None:
    """The legacy transport retains deployment routing and every request option."""
    metadata.format = audio_format
    client = create_client(hass, ENDPOINT, "existing-key")
    await hass.async_add_executor_job(client.platform_headers)
    backend = OpenAIFileTranscriptionBackend(client, "listen", "My prompt")
    route = respx_mock.post(
        f"{ENDPOINT}openai/deployments/listen/audio/transcriptions"
    ).respond(json={"text": "Lights on"})
    assert await backend.async_transcribe(metadata, audio()) == "Lights on"
    request = route.calls[0].request
    assert request.url.params["api-version"] == "2025-04-01-preview"
    assert request.headers["Authorization"] == "Bearer existing-key"
    for field, value in (
        ("model", "listen"),
        ("response_format", "json"),
        ("language", "en"),
        ("prompt", "My prompt"),
    ):
        assert f'name="{field}"\r\n\r\n{value}'.encode() in request.content
    assert (b"RIFF" in request.content) is (audio_format == stt.AudioFormats.WAV)
    if audio_format == stt.AudioFormats.OGG:
        assert PCM_AUDIO in request.content


async def test_mai_and_openai_share_token_refresh(
    hass: HomeAssistant,
    mai_entry: MockConfigEntry,
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
    respx_mock: MockRouter,
) -> None:
    """Both transports share the OAuth refresh lock and read the rotated token."""
    client = await async_create_entry_client(hass, mai_entry)
    await hass.async_add_executor_job(client.platform_headers)
    speech = respx_mock.post(TRANSCRIBE_URL).respond(
        json={"combinedPhrases": [{"text": "Hello"}]}
    )
    models = respx_mock.get(f"{ENDPOINT}openai/v1/models").respond(
        json={"object": "list", "data": []}
    )
    hass.config_entries.async_update_entry(
        mai_entry,
        data={**mai_entry.data, "token": {**mai_entry.data["token"], "expires_at": 0}},
    )
    with patch.object(
        EntraOAuth2Implementation,
        "_token_request",
        return_value={
            "access_token": "rotated-token",
            "expires_in": 3600,
            "refresh_token": "rotated-refresh",
        },
    ) as refresh:
        await asyncio.gather(
            mai_backend.async_transcribe(metadata, audio()), client.models.list()
        )
    refresh.assert_awaited_once()
    assert speech.calls[0].request.headers["Authorization"] == "Bearer rotated-token"
    assert models.calls[0].request.headers["Authorization"] == "Bearer rotated-token"
    assert mai_entry.data["token"]["refresh_token"] == "rotated-refresh"


@pytest.mark.parametrize("status", [302, 400, 403, 429, 500])
async def test_mai_http_errors_do_not_fallback(
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
    respx_mock: MockRouter,
    status: int,
) -> None:
    """Do not follow redirects, strip enhanced mode, or try API keys on errors."""
    route = respx_mock.post(TRANSCRIBE_URL).respond(
        status,
        headers={"Location": "https://attacker.example/"},
        json={"error": {"message": "sensitive service detail"}},
    )
    with pytest.raises(TranscriptionError) as err:
        await mai_backend.async_transcribe(metadata, audio())
    assert "sensitive service detail" not in str(err.value)
    assert len(route.calls) == 1
    if status == 403:
        assert "Cognitive Services Speech User" in str(err.value)


async def test_mai_timeout(
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
    respx_mock: MockRouter,
) -> None:
    """Surface connection failures without substituting a successful transcript."""
    respx_mock.post(TRANSCRIBE_URL).mock(side_effect=httpx.ReadTimeout("timeout"))
    with pytest.raises(TranscriptionError, match="Unable to connect"):
        await mai_backend.async_transcribe(metadata, audio())


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"combinedPhrases": {}},
        {"combinedPhrases": [None]},
        {"combinedPhrases": [{}]},
        {"combinedPhrases": [{"text": 1}]},
    ],
)
async def test_mai_invalid_response(
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
    respx_mock: MockRouter,
    payload: object,
) -> None:
    """Reject malformed response structures explicitly."""
    respx_mock.post(TRANSCRIBE_URL).respond(content=json.dumps(payload))
    with pytest.raises(TranscriptionError, match="invalid"):
        await mai_backend.async_transcribe(metadata, audio())


async def test_mai_invalid_json(
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
    respx_mock: MockRouter,
) -> None:
    """Distinguish invalid JSON from an empty valid transcript."""
    respx_mock.post(TRANSCRIBE_URL).respond(content=b"not json")
    with pytest.raises(TranscriptionError, match="invalid JSON"):
        await mai_backend.async_transcribe(metadata, audio())


@pytest.mark.parametrize("status", [200, 401])
async def test_mai_entity_error_and_reauth(
    hass: HomeAssistant,
    mai_entry: MockConfigEntry,
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
    respx_mock: MockRouter,
    caplog: pytest.LogCaptureFixture,
    status: int,
) -> None:
    """Empty results fail; a rejected bearer token requests native reauthentication."""
    respx_mock.post(TRANSCRIBE_URL).respond(status, json={"combinedPhrases": []})
    entity = AzureOpenAISTTEntity(
        mai_entry, next(iter(mai_entry.subentries.values())), mai_backend
    )
    entity.hass = hass
    with patch.object(mai_entry, "async_start_reauth") as reauth:
        result = await entity.async_process_audio_stream(metadata, audio())
    assert result.result is stt.SpeechResultState.ERROR
    assert result.text is None
    assert reauth.call_count == (1 if status == 401 else 0)
    if status == 401:
        assert "Azure Speech rejected the Entra token (HTTP 401)" in caplog.text
        assert "expired" not in caplog.text


@pytest.mark.parametrize(
    "model,has_afrikaans",
    [
        ("MAI-Transcribe-1.5", False),
        ("MAI-Transcribe-2", True),
    ],
)
async def test_mai_capabilities(
    hass: HomeAssistant,
    mai_entry: MockConfigEntry,
    mai_backend: MAIFileTranscriptionBackend,
    model: str,
    has_afrikaans: bool,
) -> None:
    """Advertise only supported formats and model-specific languages."""
    backend = MAIFileTranscriptionBackend(
        get_async_client(hass),
        await async_get_inference_session(hass, mai_entry),
        SPEECH_ENDPOINT,
        model,
    )
    caps = backend.capabilities
    assert ("af-ZA" in caps.languages) is has_afrikaans
    assert caps.formats == (stt.AudioFormats.WAV,)
    assert caps.codecs == (stt.AudioCodecs.PCM,)
    assert caps.bit_rates == (stt.AudioBitRates.BITRATE_16,)
    assert caps.sample_rates == (stt.AudioSampleRates.SAMPLERATE_16000,)
    assert caps.channels == (stt.AudioChannels.CHANNEL_MONO,)


async def test_mai_rejects_audio_metadata(
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
) -> None:
    """Do not send unsupported OGG input to an endpoint that accepts WAV files."""
    metadata.format = stt.AudioFormats.OGG
    with pytest.raises(TranscriptionError, match="Unsupported audio metadata"):
        await mai_backend.async_transcribe(metadata, audio())


async def test_mai_enforces_file_size(
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
) -> None:
    """Check the file limit while collecting chunks, including the WAV header."""
    with (
        patch(
            "custom_components.azure_openai_conversation.transcription._MAX_SPEECH_AUDIO_BYTES",
            64,
        ),
        pytest.raises(TranscriptionError, match="file size limit"),
    ):
        await mai_backend.async_transcribe(metadata, audio())


@pytest.mark.parametrize("backend", [STT_BACKEND_MAI_FILE, "mai_realtime", "unknown"])
async def test_no_api_key_or_unimplemented_backend(
    hass: HomeAssistant, backend: str
) -> None:
    """Reject forged configuration instead of exposing unfinished streaming or keys."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"api_key": "must-not-be-used"},
        subentries_data=[
            {
                "subentry_type": "stt",
                "title": "STT",
                "unique_id": None,
                "data": {CONF_STT_BACKEND: backend},
            }
        ],
    )
    entry.add_to_hass(hass)
    with pytest.raises(ConfigEntryError):
        await async_create_transcription_backend(
            hass, entry, next(iter(entry.subentries.values()))
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (CONF_SPEECH_ENDPOINT, "https://attacker.example"),
        (CONF_MAI_MODEL, "MAI-Transcribe-2-Streaming"),
    ],
)
async def test_invalid_configuration_precedes_authentication(
    hass: HomeAssistant,
    mai_entry: MockConfigEntry,
    field: str,
    value: str,
) -> None:
    """Reject invalid destinations and file models before obtaining credentials."""
    subentry = next(iter(mai_entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        mai_entry, subentry, data={**subentry.data, field: value}
    )
    with (
        patch(
            "custom_components.azure_openai_conversation.transcription.async_get_inference_session"
        ) as authenticate,
        pytest.raises(ConfigEntryError, match="Invalid MAI"),
    ):
        await async_create_transcription_backend(hass, mai_entry, subentry)
    authenticate.assert_not_called()


async def test_entity_passes_stream_without_buffering(
    mai_entry: MockConfigEntry, metadata: stt.SpeechMetadata
) -> None:
    """A future realtime backend receives the original live audio iterable."""
    stream = audio()

    class LiveBackend:
        """Exercise the backend protocol without a realtime implementation."""

        capabilities = OPENAI_CAPABILITIES

        async def async_transcribe(self, received_metadata, received_stream) -> str:
            """Consume chunks at the transport, not in the Home Assistant entity."""
            assert received_stream is stream
            assert received_metadata is metadata
            assert await anext(received_stream) == PCM_AUDIO[:32]
            assert await anext(received_stream) == PCM_AUDIO[32:]
            return "Streamed"

    entity = AzureOpenAISTTEntity(
        mai_entry, next(iter(mai_entry.subentries.values())), LiveBackend()
    )
    result = await entity.async_process_audio_stream(metadata, stream)
    assert result.text == "Streamed"
    assert result.result is stt.SpeechResultState.SUCCESS


async def test_audio_cancellation_propagates(
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
    respx_mock: MockRouter,
) -> None:
    """Cancellation remains cancellation, rather than a failed or partial transcript."""

    async def cancelled_audio() -> AsyncGenerator[bytes]:
        """Stop an in-progress utterance before it can be uploaded."""
        yield PCM_AUDIO[:32]
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await mai_backend.async_transcribe(metadata, cancelled_audio())
    assert not respx_mock.calls


@pytest.mark.parametrize("status", [400, 503])
async def test_runtime_token_failure(
    hass: HomeAssistant,
    mai_entry: MockConfigEntry,
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
    aioclient_mock,
    respx_mock: MockRouter,
    caplog: pytest.LogCaptureFixture,
    status: int,
) -> None:
    """Revoked grants request reauth; transient refresh errors fail without uploading."""
    import_module(f"custom_components.{DOMAIN}.config_flow")
    hass.config_entries.async_update_entry(
        mai_entry,
        data={**mai_entry.data, "token": {**mai_entry.data["token"], "expires_at": 0}},
    )
    aioclient_mock.post(
        f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token",
        status=status,
        json={"error": "invalid_grant" if status == 400 else "temporarily_unavailable"},
    )
    entity = AzureOpenAISTTEntity(
        mai_entry, next(iter(mai_entry.subentries.values())), mai_backend
    )
    entity.hass = hass
    with patch.object(mai_entry, "async_start_reauth") as reauth:
        result = await entity.async_process_audio_stream(metadata, audio())
    assert result.result is stt.SpeechResultState.ERROR
    assert reauth.call_count == (1 if status == 400 else 0)
    assert not respx_mock.calls
    if status == 400:
        assert "STT OAuth token refresh was rejected" in caplog.text
        assert "Azure Speech rejected" not in caplog.text


@pytest.mark.usefixtures("native_exposed_entities")
async def test_mai_platform_setup_and_reload(
    hass: HomeAssistant,
    mai_entry: MockConfigEntry,
    oauth_implementation: EntraOAuth2Implementation,
    metadata: stt.SpeechMetadata,
    respx_mock: MockRouter,
) -> None:
    """The native STT platform wires MAI, device identity, and unload/reload cleanup."""
    respx_mock.get(f"{ENDPOINT}openai/v1/models").respond(
        json={"object": "list", "data": []}
    )
    route = respx_mock.post(TRANSCRIBE_URL).respond(
        json={"combinedPhrases": [{"text": "Lights on"}]}
    )
    assert await hass.config_entries.async_setup(mai_entry.entry_id)
    await hass.async_block_till_done()
    subentry = next(iter(mai_entry.subentries.values()))
    entity_id = er.async_get(hass).async_get_entity_id(
        "stt", DOMAIN, subentry.subentry_id
    )
    assert entity_id is not None
    entity = stt.async_get_speech_to_text_entity(hass, entity_id)
    assert entity is not None
    assert entity.device_info["model"] == "MAI-Transcribe-2"
    assert entity.supported_formats == [stt.AudioFormats.WAV]
    assert entity.supported_codecs == [stt.AudioCodecs.PCM]
    result = await entity.async_process_audio_stream(metadata, audio())
    assert result.result is stt.SpeechResultState.SUCCESS
    assert result.text == "Lights on"
    first_session = await async_get_inference_session(hass, mai_entry)
    assert await hass.config_entries.async_reload(mai_entry.entry_id)
    await hass.async_block_till_done()
    assert await async_get_inference_session(hass, mai_entry) is not first_session
    assert (
        er.async_get(hass).async_get_entity_id("stt", DOMAIN, subentry.subentry_id)
        == entity_id
    )
    assert len(route.calls) == 1
    assert await hass.config_entries.async_unload(mai_entry.entry_id)
    await hass.async_block_till_done()


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (
            "The principal lacks the required data action for this operation",
            "RBAC data action denied",
        ),
        (
            "The principal does not have authorization to perform this action",
            "RBAC action denied",
        ),
        ("Access denied: invalid audience", "token audience rejected"),
        ("Access denied: audience is incorrect", "token audience rejected"),
        ("The token is expired", "token expired"),
        ("The token has expired", "token expired"),
        (
            "Access denied due to invalid subscription key or wrong API endpoint.",
            "gateway rejected credentials or resource endpoint",
        ),
    ],
)
async def test_speech_auth_failure_has_safe_diagnostics(
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
    respx_mock: MockRouter,
    message: str,
    reason: str,
) -> None:
    """Keep useful Azure categories and request IDs without exposing raw service text."""
    request_id = "12345678-1234-1234-1234-123456789abc"
    respx_mock.post(TRANSCRIBE_URL).respond(
        401,
        json={
            "error": {
                "code": "Unauthorized",
                "message": f"{message} sensitive-service-text",
            }
        },
        headers={"apim-request-id": request_id},
    )
    with pytest.raises(ConfigEntryAuthFailed) as err:
        await mai_backend.async_transcribe(metadata, audio())
    diagnostic = str(err.value)
    assert "HTTP 401" in diagnostic
    assert "Azure code Unauthorized" in diagnostic
    assert f"request ID {request_id}" in diagnostic
    assert reason in diagnostic
    assert "sensitive-service-text" not in diagnostic


@pytest.mark.parametrize(
    ("body", "headers", "expected"),
    [
        (b"secret-html", {}, "non-JSON error response"),
        (
            b'{"error":{"code":"secret-code","message":"secret-message"}}',
            {"apim-request-id": "secret-header"},
            "HTTP 401",
        ),
        (b'{"error":"secret"}', {}, "HTTP 401"),
        (b"null", {}, "HTTP 401"),
    ],
)
async def test_speech_auth_failure_does_not_log_untrusted_details(
    mai_backend: MAIFileTranscriptionBackend,
    metadata: stt.SpeechMetadata,
    respx_mock: MockRouter,
    body: bytes,
    headers: dict[str, str],
    expected: str,
) -> None:
    """Malformed and unexpected error payloads remain failures without leaking data."""
    respx_mock.post(TRANSCRIBE_URL).respond(401, content=body, headers=headers)
    with pytest.raises(ConfigEntryAuthFailed) as err:
        await mai_backend.async_transcribe(metadata, audio())
    assert expected in str(err.value)
    assert "secret" not in str(err.value)
