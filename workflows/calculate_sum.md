# Workflow: Calculate Sum

> **Workflow ID:** `calculate_sum`  
> **Category:** calculation  
> **Version:** 1.0

## Objective
Calculate the sum of numeric values in a specified row or column of the active workbook.

## Preconditions
- An Excel file must be open.
- Memory must contain `active_file`.
- The target row must contain numeric data.

## Trigger Phrases
- "Calculate the sum of row 3"
- "What is the total of row 5?"
- "Sum up row two"
- "Add up all values in row 4"

## Inputs
| Parameter | Type   | Required | Description                                      |
|-----------|--------|----------|--------------------------------------------------|
| row       | int    | Yes      | 1-based row index to sum                         |
| filename  | string | No       | Target file; defaults to `active_file` in memory |

## Steps

1. **Agent** extracts the `row` index from the user command.
2. **Memory** resolves `active_file` if `filename` is not provided.
3. **Validator** checks:
   - `row` is a positive integer.
   - `row` does not exceed the workbook's max row.
   - The active file exists.
4. **Tool** → `calculate_row_sum(path, index)`
5. **Feedback** → Speak the sum result to the user.

## Tool Calls
```json
{
  "type": "calculate_row_sum",
  "row": 3
}
```

## Expected Result
```json
{
  "message": "row_sum",
  "sum": 270.0,
  "speech_text": "The sum of row 3 is 270.0."
}
```

## Error Handling
- If `row` is out of range, inform the user of the valid range.
- If the row contains no numeric values, report a sum of 0.
- If no active file is set, ask the user to open or create a file first.

## Memory Updates
- `recent_commands` → append `calculate_sum`.
