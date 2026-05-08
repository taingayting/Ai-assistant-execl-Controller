"""Quick smoke test — hit every major endpoint to verify the server works."""
import requests
import json

base = "http://127.0.0.1:5000"

def p(label, r):
    print(f"\n{'='*60}")
    print(f" {label}  |  HTTP {r.status_code}")
    print(f"{'='*60}")
    try:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(r.text[:300])

# 1 — Frontend loads
r = requests.get(base)
print(f"\n[1] Frontend: HTTP {r.status_code}, has title = {'Smart Office' in r.text}")

# 2 — Parse: create excel
r2 = requests.post(f"{base}/api/parse", json={"text": "create excel with headers Name, Score"})
p("2  Parse 'create excel'", r2)
action_create = r2.json().get("action", {})

# 3 — Execute: create excel
r3 = requests.post(f"{base}/api/execute", json={"action": action_create, "filename": "smoke.xlsx"})
p("3  Execute create_excel", r3)

# 4 — Parse: add row
r4 = requests.post(f"{base}/api/parse", json={"text": "add row Tom 90"})
p("4  Parse 'add row Tom 90'", r4)
action_add = r4.json().get("action", {})

# 5 — Execute: add row
r5 = requests.post(f"{base}/api/execute", json={"action": action_add, "filename": "smoke.xlsx"})
p("5  Execute add_row", r5)

# 6 — Parse: delete row (destructive — should ask confirmation)
r6 = requests.post(f"{base}/api/parse", json={"text": "delete row 1"})
p("6  Parse 'delete row 1'", r6)
action_del = r6.json().get("action", {})

r7 = requests.post(f"{base}/api/execute", json={"action": action_del, "filename": "smoke.xlsx"})
p("7  Execute delete_row (expects confirmation)", r7)

# 8 — Download
r8 = requests.get(f"{base}/files/smoke.xlsx")
print(f"\n[8] Download: HTTP {r8.status_code}, size = {len(r8.content)} bytes")

# 9 — Backups info
r9 = requests.get(f"{base}/api/backups", params={"filename": "smoke.xlsx"})
p("9  Backups info", r9)

print("\n" + "="*60)
print(" SMOKE TEST COMPLETE")
print("="*60)
