# Workflow: Format Cells

> **Workflow ID:** `format_cells`  
> **Category:** formatting  
> **Version:** 1.0

## Objective
Apply formatting (bold, colors, font size, number format) to a range of cells in the active workbook.

## Preconditions
- An Excel file must be open.
- Memory must contain `active_file`.
- At least one formatting option must be specified.

## Trigger Phrases
- "Make the header row bold"
- "Format A1 to E1 with blue background"
- "Set font size to 14 for row 1"
- "Apply currency format to column D"
- "Color the headers white on blue"

## Inputs
| Parameter   | Type   | Required | Description                                         |
|-------------|--------|----------|-----------------------------------------------------|
| range       | string | Yes      | Cell range in A1 notation (e.g. `"A1:E1"`)          |
| bold        | bool   | No       | Apply bold font                                     |
| italic      | bool   | No       | Apply italic font                                   |
| font_size   | int    | No       | Font size in points                                 |
| font_color  | string | No       | Font color as hex (e.g. `"FFFFFF"`)                 |
| fill_color  | string | No       | Background fill color as hex (e.g. `"4472C4"`)      |
| num_format  | string | No       | Number format string (e.g. `"$#,##0.00"`)           |
| filename    | string | No       | Target file; defaults to `active_file` in memory    |

## Steps

1. **Agent** extracts `range` and formatting options from the user command.
2. **Memory** resolves `active_file` if `filename` is not provided.
3. **Validator** checks:
   - `range` is a valid cell range.
   - At least one formatting option is provided.
   - The active file exists.
4. **Tool** → `format_cells(path, range, bold, italic, font_size, font_color, fill_color, num_format)`
5. **Feedback** → Confirm formatting was applied.

## Tool Calls
```json
{
  "type": "format_cells",
  "range": "A1:E1",
  "bold": true,
  "fill_color": "4472C4",
  "font_color": "FFFFFF"
}
```

## Expected Result
```json
{
  "message": "cells_formatted",
  "range": "A1:E1",
  "speech_text": "Formatted cells A1 to E1 with bold and blue background."
}
```

## Error Handling
- If `range` is missing or invalid, ask the user to specify a cell range.
- If no formatting options are given, ask what formatting to apply.
- If no active file is set, ask the user to open or create a file first.

## Memory Updates
- `recent_commands` → append `format_cells`.
