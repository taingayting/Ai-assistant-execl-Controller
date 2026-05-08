import urllib.request
import json
import os
from time import sleep

BASE = 'http://localhost:5000'

def post_json(path, payload):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def main():
    filename = 'data.xlsx'
    file_path = os.path.join(os.path.dirname(__file__), 'files', filename)
    # 1) Create workbook with headers
    print('1) Creating workbook with headers [Name, Score]')
    resp = post_json('/api/execute', {'action': {'type': 'create_excel', 'headers': ['Name', 'Score']}, 'filename': filename})
    print('->', resp)

    # 2) Add two rows
    print('2) Adding row: Alice 95')
    resp = post_json('/api/execute', {'action': {'type': 'add_row', 'values': ['Alice', '95']}, 'filename': filename})
    print('->', resp)

    print('3) Adding row: Bob 88')
    resp = post_json('/api/execute', {'action': {'type': 'add_row', 'values': ['Bob', '88']}, 'filename': filename})
    print('->', resp)

    # 4) Add column
    print('4) Adding column: Comments with default N/A')
    resp = post_json('/api/execute', {'action': {'type': 'add_column', 'name': 'Comments', 'default': 'N/A'}, 'filename': filename})
    print('->', resp)

    # 5) Delete row 3 (header is row1, Alice row2, Bob row3) -> delete Bob
    print('5) Deleting row 3')
    resp = post_json('/api/execute', {'action': {'type': 'delete_row', 'index': 3}, 'filename': filename})
    print('->', resp)

    # Give filesystem time
    sleep(0.5)

    # 6) Read workbook and print rows
    try:
        from openpyxl import load_workbook
        wb = load_workbook(file_path)
        ws = wb.active
        print('\nFinal workbook rows:')
        for row in ws.values:
            print(row)
    except Exception as e:
        print('Failed to read workbook:', e)


if __name__ == '__main__':
    main()
