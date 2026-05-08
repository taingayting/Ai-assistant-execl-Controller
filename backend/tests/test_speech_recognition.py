"""Tests for speech_recognition.py.

Covers:
  - Language code normalisation (_to_bcp47)
  - Google Cloud STT REST backend (mocked — no real API calls)
  - Google Cloud STT SDK backend (mocked)
  - Groq Whisper fallback (mocked)
  - Public API (transcribe_audio_file) cascade logic
  - Error handling
"""
import base64
import io
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, mock_open

# Ensure backend/ is importable regardless of working directory.
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import speech_recognition as sr_module


class TestToBcp47(unittest.TestCase):
    """Unit tests for the language-code normalisation helper."""

    def test_full_tag_passthrough(self):
        """BCP-47 tags that already contain a hyphen are returned as-is."""
        self.assertEqual(sr_module._to_bcp47('en-US'), 'en-US')
        self.assertEqual(sr_module._to_bcp47('cmn-Hans-CN'), 'cmn-Hans-CN')
        self.assertEqual(sr_module._to_bcp47('th-TH'), 'th-TH')

    def test_iso_english(self):
        self.assertEqual(sr_module._to_bcp47('en'), 'en-US')

    def test_iso_chinese(self):
        self.assertEqual(sr_module._to_bcp47('zh'), 'cmn-Hans-CN')

    def test_iso_thai(self):
        self.assertEqual(sr_module._to_bcp47('th'), 'th-TH')

    def test_iso_japanese(self):
        self.assertEqual(sr_module._to_bcp47('ja'), 'ja-JP')

    def test_case_insensitive(self):
        self.assertEqual(sr_module._to_bcp47('EN'), 'en-US')
        self.assertEqual(sr_module._to_bcp47('ZH'), 'cmn-Hans-CN')

    def test_empty_returns_default(self):
        self.assertEqual(sr_module._to_bcp47(''), 'en-US')
        self.assertEqual(sr_module._to_bcp47(None), 'en-US')

    def test_unknown_iso_code(self):
        """Unknown 2-letter codes fall back to 'en-US'."""
        result = sr_module._to_bcp47('xx')
        self.assertEqual(result, 'en-US')


class TestGoogleCloudBackend(unittest.TestCase):
    """Unit tests for _transcribe_google_cloud (Google STT mocked)."""

    def _make_mock_speech_module(self, transcript_text: str):
        """Build a minimal fake `google.cloud.speech` module."""
        mock_speech = MagicMock()

        # Mock response with a single result
        alt = MagicMock()
        alt.transcript = transcript_text
        result = MagicMock()
        result.alternatives = [alt]
        response = MagicMock()
        response.results = [result]

        mock_client = MagicMock()
        mock_client.recognize.return_value = response
        mock_speech.SpeechClient.return_value = mock_client

        # AudioEncoding enum
        enc = MagicMock()
        enc.WEBM_OPUS = 'WEBM_OPUS'
        enc.LINEAR16 = 'LINEAR16'
        enc.FLAC = 'FLAC'
        enc.OGG_OPUS = 'OGG_OPUS'
        enc.MP3 = 'MP3'
        mock_speech.RecognitionConfig.AudioEncoding = enc
        mock_speech.RecognitionConfig.return_value = MagicMock()
        mock_speech.RecognitionAudio.return_value = MagicMock()

        return mock_speech, mock_client

    def test_successful_transcription(self):
        """Returns transcribed text when Google STT succeeds."""
        mock_speech, mock_client = self._make_mock_speech_module('Add row Tom 90')

        with patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': '/fake/key.json'}), \
             patch.dict(sys.modules, {'google.cloud.speech': mock_speech,
                                      'google.cloud': MagicMock(speech=mock_speech)}), \
             patch('builtins.open', mock_open(read_data=b'fake_audio_bytes')):
            result = sr_module._transcribe_google_cloud('/fake/audio.webm', language='en')

        self.assertEqual(result, 'Add row Tom 90')
        mock_client.recognize.assert_called_once()

    def test_returns_none_without_credentials(self):
        """Returns None (triggers fallback) when credentials env var is missing."""
        env = {k: v for k, v in os.environ.items() if k != 'GOOGLE_APPLICATION_CREDENTIALS'}
        with patch.dict(os.environ, env, clear=True):
            result = sr_module._transcribe_google_cloud('/fake/audio.webm', language='en')
        self.assertIsNone(result)

    def test_returns_none_when_library_missing(self):
        """Returns None gracefully when google-cloud-speech is not installed."""
        with patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': '/fake/key.json'}), \
             patch.dict(sys.modules, {'google.cloud.speech': None}):
            result = sr_module._transcribe_google_cloud('/fake/audio.webm', language='en')
        self.assertIsNone(result)

    def test_empty_results_returns_empty_string(self):
        """Returns empty string (not None) when STT response has no results."""
        mock_speech, mock_client = self._make_mock_speech_module('')
        mock_client.recognize.return_value.results = []

        with patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': '/fake/key.json'}), \
             patch.dict(sys.modules, {'google.cloud.speech': mock_speech,
                                      'google.cloud': MagicMock(speech=mock_speech)}), \
             patch('builtins.open', mock_open(read_data=b'fake_audio_bytes')):
            result = sr_module._transcribe_google_cloud('/fake/audio.webm', language='en')

        self.assertEqual(result, '')

    def test_api_exception_returns_none(self):
        """Returns None (not a raised exception) on API errors."""
        mock_speech, mock_client = self._make_mock_speech_module('')
        mock_client.recognize.side_effect = Exception('API error: 503')

        with patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': '/fake/key.json'}), \
             patch.dict(sys.modules, {'google.cloud.speech': mock_speech,
                                      'google.cloud': MagicMock(speech=mock_speech)}), \
             patch('builtins.open', mock_open(read_data=b'fake_audio_bytes')):
            result = sr_module._transcribe_google_cloud('/fake/audio.webm', language='en')

        self.assertIsNone(result)

    def test_language_code_passed_to_config(self):
        """The BCP-47 language code is forwarded to RecognitionConfig."""
        mock_speech, mock_client = self._make_mock_speech_module('สวัสดี')
        captured = {}

        def capture_config(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        mock_speech.RecognitionConfig.side_effect = capture_config

        with patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': '/fake/key.json'}), \
             patch.dict(sys.modules, {'google.cloud.speech': mock_speech,
                                      'google.cloud': MagicMock(speech=mock_speech)}), \
             patch('builtins.open', mock_open(read_data=b'fake_audio_bytes')):
            sr_module._transcribe_google_cloud('/fake/audio.webm', language='th')

        self.assertEqual(captured.get('language_code'), 'th-TH')


class TestGroqFallback(unittest.TestCase):
    """Unit tests for _transcribe_groq (Groq Whisper mocked)."""

    def test_returns_none_without_api_key(self):
        with patch.object(sr_module, '_GROQ_KEY', ''):
            result = sr_module._transcribe_groq('/fake/audio.webm', language='en')
        self.assertIsNone(result)

    def test_successful_transcription(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'text': 'Create Excel with headers Name Score'}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(sr_module, '_GROQ_KEY', 'fake-key'), \
             patch('requests.post', return_value=mock_resp) as mock_post, \
             patch('builtins.open', mock_open(read_data=b'audio')):
            result = sr_module._transcribe_groq('/fake/audio.webm', language='en')

        self.assertEqual(result, 'Create Excel with headers Name Score')
        mock_post.assert_called_once()

    def test_strips_bcp47_to_iso_for_groq(self):
        """Groq API receives 2-letter ISO code, not full BCP-47 tag."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'text': 'สวัสดี'}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(sr_module, '_GROQ_KEY', 'fake-key'), \
             patch('requests.post', return_value=mock_resp) as mock_post, \
             patch('builtins.open', mock_open(read_data=b'audio')):
            sr_module._transcribe_groq('/fake/audio.webm', language='th-TH')

        _call_data = mock_post.call_args
        sent_data = _call_data.kwargs.get('data') or _call_data[1].get('data', {})
        self.assertEqual(sent_data.get('language'), 'th')

    def test_http_error_returns_none(self):
        """Returns None on HTTP error (not raised)."""
        import requests as req_lib
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_lib.exceptions.HTTPError('400')

        with patch.object(sr_module, '_GROQ_KEY', 'fake-key'), \
             patch('requests.post', return_value=mock_resp), \
             patch('builtins.open', mock_open(read_data=b'audio')):
            result = sr_module._transcribe_groq('/fake/audio.webm', language='en')

        self.assertIsNone(result)


class TestGoogleRestBackend(unittest.TestCase):
    """Unit tests for _transcribe_google_rest (REST API with API key, mocked)."""

    def _make_mock_response(self, transcript: str):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            'results': [{'alternatives': [{'transcript': transcript, 'confidence': 0.98}]}]
        }
        return mock_resp

    def test_returns_none_without_api_key(self):
        with patch.object(sr_module, '_GOOGLE_API_KEY', ''):
            result = sr_module._transcribe_google_rest('/fake/audio.webm', language='en')
        self.assertIsNone(result)

    def test_successful_transcription_english(self):
        mock_resp = self._make_mock_response('Add row Tom 90')
        with patch.object(sr_module, '_GOOGLE_API_KEY', 'fake-api-key'), \
             patch('requests.post', return_value=mock_resp) as mock_post, \
             patch('builtins.open', mock_open(read_data=b'audio_bytes')):
            result = sr_module._transcribe_google_rest('/fake/audio.webm', language='en')
        self.assertEqual(result, 'Add row Tom 90')
        mock_post.assert_called_once()

    def test_api_key_sent_as_query_param(self):
        """API key must be in the query params, not in the JSON body."""
        mock_resp = self._make_mock_response('hello')
        with patch.object(sr_module, '_GOOGLE_API_KEY', 'my-secret-key'), \
             patch('requests.post', return_value=mock_resp) as mock_post, \
             patch('builtins.open', mock_open(read_data=b'audio_bytes')):
            sr_module._transcribe_google_rest('/fake/audio.webm', language='en')
        call_kwargs = mock_post.call_args.kwargs
        self.assertEqual(call_kwargs.get('params', {}).get('key'), 'my-secret-key')

    def test_language_code_normalised_to_bcp47(self):
        """Short ISO code 'th' must be expanded to 'th-TH' in the config."""
        mock_resp = self._make_mock_response('สวัสดี')
        captured_json = {}

        def capture_post(url, **kwargs):
            captured_json.update(kwargs.get('json', {}))
            return mock_resp

        with patch.object(sr_module, '_GOOGLE_API_KEY', 'fake-key'), \
             patch('requests.post', side_effect=capture_post), \
             patch('builtins.open', mock_open(read_data=b'audio_bytes')):
            sr_module._transcribe_google_rest('/fake/audio.webm', language='th')

        self.assertEqual(captured_json['config']['languageCode'], 'th-TH')

    def test_audio_sent_as_base64(self):
        """Audio file content must be base64-encoded in the JSON body."""
        raw_bytes = b'fake_audio_content'
        expected_b64 = base64.b64encode(raw_bytes).decode('utf-8')
        mock_resp = self._make_mock_response('test')
        captured_json = {}

        def capture_post(url, **kwargs):
            captured_json.update(kwargs.get('json', {}))
            return mock_resp

        with patch.object(sr_module, '_GOOGLE_API_KEY', 'fake-key'), \
             patch('requests.post', side_effect=capture_post), \
             patch('builtins.open', mock_open(read_data=raw_bytes)):
            sr_module._transcribe_google_rest('/fake/audio.wav', language='en')

        self.assertEqual(captured_json['audio']['content'], expected_b64)

    def test_empty_results_returns_empty_string(self):
        """Empty `results` array → empty string (not None, not error)."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {'results': []}
        with patch.object(sr_module, '_GOOGLE_API_KEY', 'fake-key'), \
             patch('requests.post', return_value=mock_resp), \
             patch('builtins.open', mock_open(read_data=b'audio')):
            result = sr_module._transcribe_google_rest('/fake/audio.webm', language='en')
        self.assertEqual(result, '')

    def test_http_error_returns_none(self):
        """HTTP errors (e.g. 403 invalid key) return None to trigger fallback."""
        import requests as req_lib
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_lib.exceptions.HTTPError('403')
        with patch.object(sr_module, '_GOOGLE_API_KEY', 'bad-key'), \
             patch('requests.post', return_value=mock_resp), \
             patch('builtins.open', mock_open(read_data=b'audio')):
            result = sr_module._transcribe_google_rest('/fake/audio.webm', language='en')
        self.assertIsNone(result)

    def test_chinese_language(self):
        mock_resp = self._make_mock_response('你好')
        captured_json = {}

        def capture_post(url, **kwargs):
            captured_json.update(kwargs.get('json', {}))
            return mock_resp

        with patch.object(sr_module, '_GOOGLE_API_KEY', 'fake-key'), \
             patch('requests.post', side_effect=capture_post), \
             patch('builtins.open', mock_open(read_data=b'audio')):
            result = sr_module._transcribe_google_rest('/fake/audio.webm', language='zh')

        self.assertEqual(result, '你好')
        self.assertEqual(captured_json['config']['languageCode'], 'cmn-Hans-CN')


class TestTranscribeAudioFile(unittest.TestCase):
    """Integration tests for the public API (cascade logic)."""

    def test_uses_google_rest_first(self):
        """REST backend result is returned without calling SDK or Groq."""
        with patch.object(sr_module, '_transcribe_google_rest', return_value='hello world') as mock_r, \
             patch.object(sr_module, '_transcribe_google_cloud') as mock_sdk, \
             patch.object(sr_module, '_transcribe_groq') as mock_groq:
            result = sr_module.transcribe_audio_file('/fake/audio.webm', language='en-US')

        self.assertEqual(result, 'hello world')
        mock_r.assert_called_once_with('/fake/audio.webm', language='en-US')
        mock_sdk.assert_not_called()
        mock_groq.assert_not_called()

    def test_falls_back_to_sdk_when_rest_unavailable(self):
        """SDK backend is called when REST returns None."""
        with patch.object(sr_module, '_transcribe_google_rest', return_value=None), \
             patch.object(sr_module, '_transcribe_google_cloud', return_value='sdk result') as mock_sdk, \
             patch.object(sr_module, '_transcribe_groq') as mock_groq:
            result = sr_module.transcribe_audio_file('/fake/audio.webm', language='en')

        self.assertEqual(result, 'sdk result')
        mock_sdk.assert_called_once()
        mock_groq.assert_not_called()

    def test_falls_back_to_groq_when_google_unavailable(self):
        """Groq Whisper is called when both Google backends return None."""
        with patch.object(sr_module, '_transcribe_google_rest', return_value=None), \
             patch.object(sr_module, '_transcribe_google_cloud', return_value=None), \
             patch.object(sr_module, '_transcribe_groq', return_value='add row Alice 95'):
            result = sr_module.transcribe_audio_file('/fake/audio.webm', language='en')

        self.assertEqual(result, 'add row Alice 95')

    def test_error_message_when_all_backends_fail(self):
        """Returns a [transcription_error] string when all three backends fail."""
        with patch.object(sr_module, '_transcribe_google_rest', return_value=None), \
             patch.object(sr_module, '_transcribe_google_cloud', return_value=None), \
             patch.object(sr_module, '_transcribe_groq', return_value=None):
            result = sr_module.transcribe_audio_file('/fake/audio.webm', language='en')

        self.assertTrue(result.startswith('[transcription_error]'))

    def test_unhandled_exception_caught(self):
        """An unexpected exception is caught; result is an error string."""
        with patch.object(sr_module, '_transcribe_google_rest', side_effect=RuntimeError('crash')), \
             patch.object(sr_module, '_transcribe_google_cloud', return_value=None), \
             patch.object(sr_module, '_transcribe_groq', return_value=None):
            result = sr_module.transcribe_audio_file('/fake/audio.webm', language='en')
        self.assertTrue(result.startswith('[transcription_error]'))

    def test_chinese_language_passed_correctly(self):
        """Full BCP-47 Chinese code is forwarded to REST backend."""
        with patch.object(sr_module, '_transcribe_google_rest', return_value='你好') as mock_r, \
             patch.object(sr_module, '_transcribe_google_cloud') as mock_sdk:
            result = sr_module.transcribe_audio_file('/fake/audio.webm', language='cmn-Hans-CN')

        mock_r.assert_called_once_with('/fake/audio.webm', language='cmn-Hans-CN')
        mock_sdk.assert_not_called()
        self.assertEqual(result, '你好')

    def test_thai_language_passthrough(self):
        """Thai BCP-47 code is forwarded without modification."""
        with patch.object(sr_module, '_transcribe_google_rest', return_value='เพิ่มแถว') as mock_r:
            result = sr_module.transcribe_audio_file('/fake/audio.webm', language='th-TH')

        mock_r.assert_called_once_with('/fake/audio.webm', language='th-TH')
        self.assertEqual(result, 'เพิ่มแถว')


if __name__ == '__main__':
    unittest.main()
