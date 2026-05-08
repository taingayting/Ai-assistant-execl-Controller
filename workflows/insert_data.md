# Workflow: Insert Data

> **Workflow ID:** `insert_data`  
> **Category:** data_entry  
> **Version:** 1.0

## Objective
Insert data into the active workbook — either by appending a new row or writing to a specific cell.

## Preconditions
- An Excel file must be open.
- Memory must contain `active_file`.
- For cell writing, the cell reference must be valid A1 notation.

## Trigger Phrases
- "Add row Tom ninety"
- "Insert Tom, 90 into the sheet"
- "Write Hello in cell A1"
- "Put 100 in B5"
- "Add a new row with values Apple, 50, 3.99"

## Inputs

### For Row Insertion
| Parameter | Type      | Required | Description                                      |
|-----------|-----------|----------|--------------------------------------------------|
| values    | list[any] | Yes      | Values for each column in the new row            |
| filename  | string    | No       | Target file; defaults to `active_file` in memory |

### For Cell Writing
| Parameter | Type   | Required | Description                                      |
|-----------|--------|----------|--------------------------------------------------|
| cell      | string | Yes      | Cell reference in A1 notation (e.g. `"B5"`)      |
| value     | any    | Yes      | Value to write                                   |
| filename  | string | No       | Target file; defaults to `active_file` in memory |

## Steps

### Row Insertion
1. **Agent** extracts `values` list from the user command.
2. **Memory** resolves `active_file`.
3. **Validator** checks:
   - `values` is a non-empty list.
   - The active file exists.
4. **Tool** → `add_row(path, values)`
5. **Feedback** → Confirm row was appended.

### Cell Writing
1. **Agent** extracts `cell` reference and `value`.
2. **Memory** resolves `active_file`.
3. **Validator** checks:
   - `cell` is valid A1-style reference.
   - The active file exists.
4. **Tool** → `write_cell(path, cell, value)`
5. **Feedback** → Confirm value was written to the cell.

## Tool Calls

### Row
```json
{
  "type": "add_row",
  "values": ["Tom", 90]
}
```

### Cell
```json
{
  "type": "write_cell",
  "cell": "B5",
  "value": 100
}
```

## Expected Results

### Row
```json
{
  "message": "row_added",
  "values": ["Tom", 90]
}
```

### Cell
```json
{
  "message": "cell_written",
  "cell": "B5",
  "value": 100,
  "speech_text": "Wrote 100 to B5."
}
```

## Error Handling
- If `values` is empty or not a list, ask the user to provide data.
- If `cell` reference is invalid, ask for a valid A1-style reference.
- If no active file is set, ask the user to open or create a file first.
- Numeric strings are automatically coerced to numbers.

## Memory Updates
- `recent_commands` → append `insert_data`.
