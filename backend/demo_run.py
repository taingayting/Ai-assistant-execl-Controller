"""Simple end-to-end demo runner that exercises parse -> execute -> download.

This script assumes the Flask server is running on http://127.0.0.1:5000
and will POST example commands to the `/api/parse` and `/api/execute` endpoints.

It does not require audio files; instead it uses textual commands to validate
the JSON parsing and Excel generation.
"""
import requests
import json

BASE = 'http://127.0.0.1:5000'

def post_parse(text, use_llm=False):
    r = requests.post(BASE + '/api/parse', json={'text': text, 'use_llm': use_llm}, timeout=10)
    r.raise_for_status()
    return r.json()

def post_execute(action, filename='demo.xlsx'):
    r = requests.post(BASE + '/api/execute', json={'action': action, 'filename': filename}, timeout=10)
    r.raise_for_status()
    return r.json()

def main():
    tests = [
        'create excel with headers "Name,Score"',
        'add row Tom ninety',
        'add column Score',
        'delete the third row'
    ]

    for t in tests:
        print('\n----')
        print('Command:', t)
        resp = post_parse(t, use_llm=False)
        print('Parse response:', json.dumps(resp, indent=2))
        action = resp.get('action')
        if action and action.get('action') != 'unknown' and action.get('type') != 'unknown':
            print('Executing action...')
            exe = post_execute(action, filename='demo.xlsx')
            print('Execute response:', json.dumps(exe, indent=2))
        else:
            print('No executable action parsed; suggestions:', resp.get('suggestions'))

if __name__ == '__main__':
    main()
