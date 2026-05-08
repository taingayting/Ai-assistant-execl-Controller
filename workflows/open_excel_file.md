# Workflow: Open Excel File

> **Workflow ID:** `open_excel_file`  
> **Category:** file_management  
> **Version:** 1.0

## Objective
Open an existing Excel workbook and set it as the active file for subsequent commands.

## Preconditions
- The target file must exist in the `files/` directory.
- The file must be a valid Excel workbook (`.xlsx`, `.xls`, `.csv`).

## Trigger Phrases
- "Open sales.xlsx"
- "Load the Excel file"
- "Switch to report.xlsx"
- "Use the file data.xlsx"

## Inputs
| Parameter | Type   | Required | Description                            |
|-----------|--------|----------|----------------------------------------|
| filename  | string | Yes      | Name of the file to open               |

## Steps

1. **Agent** extracts `filename` from the user command.
2. **Validator** checks:
   - `filename` ends with a supported extension (`.xlsx`, `.xls`, `.csv`).
   - File exists in the `files/` directory.
3. **Tool** → `ensure_workbook(path)` to verify the file is a valid workbook.
4. **Memory** stores `active_file = filename` and reads sheet names.
5. **Feedback** → Confirm the file is open and report sheet names and row count.

## Tool Calls
```json
{
  "type": "open_file",
  "filename": "sales.xlsx"
}
```

## Expected Result
```json
{
  "message": "file_opened",
  "path": "files/sales.xlsx",
  "sheets": ["Sheet1"],
  "rows": 10
}
```

## Error Handling
- If the file does not exist, inform the user and suggest creating it.
- If the file is corrupted, return an error and suggest uploading again.

## Memory Updates
- `active_file` → set to the opened filename.
- `active_sheet` → set to the workbook's active sheet.
