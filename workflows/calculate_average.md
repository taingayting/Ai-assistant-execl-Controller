# Workflow: Calculate Average

> **Workflow ID:** `calculate_average`  
> **Category:** calculation  
> **Version:** 1.0

## Objective
Calculate the average of numeric values in a specified column of the active workbook.

## Preconditions
- An Excel file must be open.
- Memory must contain `active_file`.
- The target column must contain numeric data.

## Trigger Phrases
- "Calculate average for column two"
- "What is the average of column B?"
- "Get the mean of the Score column"
- "Average column 3"

## Inputs
| Parameter | Type         | Required | Description                                        |
|-----------|--------------|----------|----------------------------------------------------|
| column    | string / int | Yes      | Column letter (e.g. `"B"`) or 1-based index        |
| filename  | string       | No       | Target file; defaults to `active_file` in memory   |

## Steps

1. **Agent** extracts the `column` identifier from the user command.
2. **Memory** resolves `active_file` if `filename` is not provided.
3. **Validator** checks:
   - Column identifier is valid.
   - Column exists and contains numeric data.
   - The active file exists.
4. **Tool** → `calculate_average(path, column)`
5. **Feedback** → Speak the average result to the user.

## Tool Calls
```json
{
  "type": "calculate_average",
  "column": 2
}
```

## Expected Result
```json
{
  "message": "column_average",
  "average": 85.5,
  "count": 10,
  "speech_text": "The average of column B is 85.5 across 10 values."
}
```

## Error Handling
- If column is out of range, inform the user of the valid range.
- If the column contains no numeric values, report that no average can be computed.
- If no active file is set, ask the user to open or create a file first.

## Memory Updates
- `recent_commands` → append `calculate_average`.
