# Workflow: Create Sales Report

> **Workflow ID:** `create_sales_report`  
> **Category:** reporting  
> **Version:** 1.0

## Objective
Create a complete sales report workbook with standard columns, formatted headers, and summary formulas.

## Preconditions
- The `files/` directory must be writable.
- No conflicting file with the same name should exist (or user confirms overwrite).

## Trigger Phrases
- "Create a sales report"
- "Make a new sales spreadsheet"
- "Generate a sales report with totals"
- "Build a revenue report"

## Inputs
| Parameter  | Type       | Required | Description                                     |
|------------|------------|----------|-------------------------------------------------|
| filename   | string     | No       | Defaults to `sales_report.xlsx`                 |
| headers    | list[str]  | No       | Defaults to standard sales columns (see below)  |

### Default Headers
`["Product", "Quantity", "Unit Price", "Total", "Date"]`

## Steps

1. **Agent** extracts optional `filename` or uses default `sales_report.xlsx`.
2. **Memory** stores `active_file = filename`.
3. **Tool** → `create_excel(path, headers)` — create workbook with sales headers.
4. **Tool** → `format_cells(path, range="A1:E1", bold=true, fill_color="4472C4", font_color="FFFFFF")` — format header row.
5. **Tool** → `add_column(path, "Revenue Summary")` — add summary column if requested.
6. **Feedback** → Confirm report creation with column names.

## Tool Calls (Sequential)
```json
[
  {
    "type": "create_excel",
    "headers": ["Product", "Quantity", "Unit Price", "Total", "Date"]
  },
  {
    "type": "format_cells",
    "range": "A1:E1",
    "bold": true,
    "fill_color": "4472C4",
    "font_color": "FFFFFF"
  }
]
```

## Expected Result
```json
{
  "message": "sheet_created_and_row_added",
  "headers": ["Product", "Quantity", "Unit Price", "Total", "Date"],
  "speech_text": "Sales report created with 5 columns."
}
```

## Error Handling
- If a file with the same name exists, ask user to confirm overwrite or choose a different name.
- If headers are invalid, fall back to the default set.

## Memory Updates
- `active_file` → set to `sales_report.xlsx` (or custom name).
- `active_sheet` → set to `Sheet1`.
- `recent_commands` → append `create_sales_report`.
