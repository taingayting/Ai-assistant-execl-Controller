import urllib.request
import json
import sys
import os
from time import sleep

BASE = 'http://localhost:5000'

def post_json(path, payload):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def main():
    # 1) Parse
    cmd = 'Add row Tom 90'
    print('Parsing command:', cmd)
    parsed = post_json('/api/parse', {'text': cmd, 'use_llm': False})
    print('Parse response:', json.dumps(parsed, indent=2))

    action = parsed.get('action')
    if not action:
        print('No action parsed, exiting')
        return

    # 2) Execute
    filename = 'data.xlsx'
    print('Executing action on', filename)
    exec_resp = post_json('/api/execute', {'action': action, 'filename': filename})
    print('Execute response:', json.dumps(exec_resp, indent=2))

    # 3) Wait a moment for file to be written then read it
    filepath = exec_resp.get('filepath')
    if not filepath:
        filepath = os.path.join(os.path.dirname(__file__), 'files', filename)
    print('Reading file at', filepath)

    # Give filesystem a moment
    sleep(0.5)

    try:
        from openpyxl import load_workbook
        wb = load_workbook(filepath)
        ws = wb.active
        rows = list(ws.values)
        print('Workbook rows:')
        for r in rows:
            print(r)
    except Exception as e:
        print('Failed to read workbook:', e)


if __name__ == '__main__':
    main()
