import io
from unittest.mock import patch

import app as app_module


def _make_fake_wav():
    # Minimal WAV header + silence payload so Flask will accept the uploaded file.
    return io.BytesIO(b"RIFF----WAVEfmt " + (b"\x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data----"))


def test_upload_audio_create_and_auto_execute(client, tmp_path):
    # Patch transcription + LLM router to return a known action
    with patch('app.transcribe_audio_file') as mock_transcribe, \
         patch('agent.route_llm', return_value={"type": "create_excel", "headers": ["Name", "Score"]}):
        mock_transcribe.return_value = 'create excel with headers "Name,Score"'

        data = {
            'auto_execute': 'true',
            'filename': 'test_audio_demo.xlsx',
        }
        file_stream = (io.BytesIO(b'fake'), 'sample.wav')
        resp = client.post('/api/upload-audio', data={
            'file': (file_stream[0], file_stream[1]),
            'auto_execute': data['auto_execute'],
            'filename': data['filename'],
        }, content_type='multipart/form-data')

        assert resp.status_code == 200
        j = resp.get_json()
        assert j['success'] is True
        assert 'action' in j
        assert j['action'].get('type') == 'create_excel'
        assert 'exec_result' in j
        assert j['exec_result'].get('message') == 'created'


def test_upload_audio_add_row(client):
    # Test that the LLM parser is called with the transcript
    with patch('app.transcribe_audio_file') as mock_transcribe, \
         patch('agent.route_llm', return_value={"type": "add_row", "values": ["Tom", 90]}):
        mock_transcribe.return_value = 'add row Tom ninety'
        file_stream = (io.BytesIO(b'fake2'), 'sample2.wav')
        resp = client.post('/api/upload-audio', data={
            'file': (file_stream[0], file_stream[1]),
            'auto_execute': 'false',
        }, content_type='multipart/form-data')

        assert resp.status_code == 200
        j = resp.get_json()
        assert j['success'] is True
        assert j['action'].get('type') == 'add_row'
