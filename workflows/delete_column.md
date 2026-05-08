# Workflow: Delete Column

> **Workflow ID:** `delete_column`  
> **Category:** structure  
> **Version:** 1.0

## Objective
Delete a column from the active Excel workbook by letter or index. This is a destructive action that requires user confirmation.

## Preconditions
- An Excel file must be open.
- Memory must contain `active_file`.
- The target column must exist in the workbook.
- User confirmation is required (destructive action).

## Trigger Phrases
- "Delete column B"
- "Remove the second column"
- "Drop column Score"
- "Delete the Price column"

## Inputs
| Parameter | Type         | Required | Description                                        |
|-----------|--------------|----------|----------------------------------------------------|
| column    | string / int | Yes      | Column letter (e.g. `"B"`) or 1-based index        |
| filename  | string       | No       | Target file; defaults to `active_file` in memory   |

## Steps

1. **Agent** extracts `column` identifier from the user command.
2. **Memory** resolves `active_file` if `filename` is not provided.
3. **Validator** checks:
   - Column identifier is valid (letter or positive integer).
   - Column exists in the workbook.
4. **Confirmation** → Because this is a destructive action, prompt the user:
   _"Are you sure you want to delete column B? It contains 10 rows of data."_
5. **Backup** → Create a rotating backup of the file before deletion.
6. **Tool** → `delete_column(path, col)`
7. **Feedback** → Confirm deletion and number of cells removed.

## Tool Calls
```json
{
  "type": "delete_column",
  "column": "B"
}
```

## Expected Result
```json
{
  "message": "column_deleted",
  "column": "B",
  "rows_affected": 10,
  "speech_text": "column B deleted (10 cells removed)"
}
```

## Error Handling
- If column is out of range, inform the user of the valid range.
- If user cancels confirmation, abort without changes.
- If deletion fails, restore from backup automatically.

## Memory Updates
- `recent_commands` → append `delete_column`.
- Backup path stored for undo support.
