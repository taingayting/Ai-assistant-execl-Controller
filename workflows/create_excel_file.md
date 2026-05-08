# Workflow: Create Excel File

> **Workflow ID:** `create_excel_file`  
> **Category:** file_management  
> **Version:** 1.0

## Objective
Create a new Excel workbook, optionally with predefined column headers.

## Preconditions
- A valid `filename` must be provided or inferred from the user command.
- The `files/` directory must be writable.

## Trigger Phrases
- "Create a new Excel file"
- "Make a new spreadsheet"
- "Create Excel with headers Name, Score"
- "Start a new workbook called sales.xlsx"

## Inputs
| Parameter  | Type       | Required | Description                              |
|------------|------------|----------|------------------------------------------|
| filename   | string     | Yes      | Name for the new file (e.g. `sales.xlsx`)|
| headers    | list[str]  | No       | Column headers for the first row         |

## Steps

1. **Agent** extracts `filename` and optional `headers` from the user command.
2. **Memory** stores `active_file = filename`.
3. **Validator** checks:
   - `filename` ends with `.xlsx`.
   - `filename` does not contain path traversal characters.
   - File does not already exist (or user confirms overwrite).
4. **Tool** → `create_excel(path, headers)`
5. **Feedback** → Confirm creation to the user.

## Tool Calls
```json
{
  "type": "create_excel",
  "headers": ["Name", "Score"]
}
```

## Expected Result
```json
{
  "message": "created",
  "path": "files/sales.xlsx"
}
```

## Error Handling
- If the file already exists, prompt for confirmation before overwriting.
- If `filename` is missing, ask the user to specify a name.

## Memory Updates
- `active_file` → set to the newly created filename.
- `active_sheet` → set to `Sheet1` (default).
