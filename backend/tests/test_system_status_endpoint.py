"""Tests for /api/system_status monitoring endpoint."""

from memory import memory


def test_system_status_empty_state(client):
    memory.clear()

    resp = client.get('/api/system_status')
    assert resp.status_code == 200
    body = resp.get_json()

    assert set(body.keys()) == {
        'active_file',
        'active_sheet',
        'recent_commands',
        'memory_state',
        'available_workflows',
    }
    assert body['active_file'] is None
    assert body['active_sheet'] is None
    assert body['recent_commands'] == []
    assert body['memory_state']['recent_commands'] == []
    assert isinstance(body['available_workflows'], list)
    assert 'create_sales_report' in body['available_workflows']


def test_system_status_populated_state(client):
    memory.clear()
    memory.set_active_file('monitor.xlsx')
    memory.set_active_sheet('Sheet1')
    memory.add_command({'type': 'create_excel'})

    resp = client.get('/api/system_status')
    assert resp.status_code == 200
    body = resp.get_json()

    assert body['active_file'] == 'monitor.xlsx'
    assert body['active_sheet'] == 'Sheet1'
    assert body['recent_commands']
    assert body['recent_commands'][-1]['type'] == 'create_excel'
    assert body['memory_state']['active_file'] == 'monitor.xlsx'
    assert body['memory_state']['active_sheet'] == 'Sheet1'
