# Workflow: Add Column

> **Workflow ID:** `add_column`  
> **Category:** structure  
> **Version:** 1.0

## Objective
Add a new column to the active Excel workbook with a specified header name and optional default value.

## Preconditions
- An Excel file must be open.
- Memory must contain `active_file`.
- The column name must not already exist in the workbook.

## Trigger Phrases
- "Add a column called Score"
- "Add column Price"
- "Insert a new column named Category"
- "Add column Status with default value Pending"

## Inputs
| Parameter     | Type   | Required | Description                                      |
|---------------|--------|----------|--------------------------------------------------|
| name          | string | Yes      | Header name for the new column                   |
| default_value | any    | No       | Default value to fill in existing rows            |
| filename      | string | No       | Target file; defaults to `active_file` in memory |

## Steps

1. **Agent** extracts `name` and optional `default_value` from the user command.
2. **Memory** resolves `active_file` if `filename` is not provided.
3. **Validator** checks:
   - `name` is a non-empty string.
   - A column with the same name does not already exist.
   - The active file exists.
4. **Tool** → `add_column(path, column_name, default_value)`
5. **Feedback** → Confirm the column was added and its position (e.g., column D).

## Tool Calls
```json
{
  "type": "add_column",
  "name": "Score",
  "default": null
}
```

## Expected Result
```json
{
  "message": "column_added",
  "column": "Score"
}
```

## Error Handling
- If `name` is missing, ask the user for the column name.
- If no active file is set, ask the user to open or create a file first.
- If the column already exists, inform the user and ask how to proceed.

## Memory Updates
- `recent_commands` → append `add_column`.
