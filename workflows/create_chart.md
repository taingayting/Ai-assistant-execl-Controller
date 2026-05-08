# Workflow: Create Chart

> **Workflow ID:** `create_chart`  
> **Category:** visualization  
> **Version:** 1.0

## Objective
Create a chart (bar, line, pie, etc.) from data in the active workbook and embed it in the sheet.

## Preconditions
- An Excel file must be open.
- Memory must contain `active_file`.
- The data range must contain numeric data suitable for charting.

## Trigger Phrases
- "Create a bar chart from the data"
- "Make a pie chart of column B"
- "Add a line chart for sales data"
- "Chart the revenue by product"
- "Create a graph from A1 to C10"

## Inputs
| Parameter   | Type   | Required | Description                                               |
|-------------|--------|----------|-----------------------------------------------------------|
| chart_type  | string | No       | Chart type: `bar`, `line`, `pie`, `scatter`. Default: `bar` |
| data_range  | string | Yes      | Data range in A1 notation (e.g. `"A1:C10"`)                |
| title       | string | No       | Chart title                                                |
| x_axis      | string | No       | Label for the X axis                                       |
| y_axis      | string | No       | Label for the Y axis                                       |
| position    | string | No       | Cell where the chart is placed (e.g. `"F2"`). Default: auto |
| filename    | string | No       | Target file; defaults to `active_file` in memory           |

## Steps

1. **Agent** extracts `chart_type`, `data_range`, and optional labels from the user command.
2. **Memory** resolves `active_file` if `filename` is not provided.
3. **Validator** checks:
   - `data_range` is a valid cell range with data.
   - `chart_type` is a supported type.
   - The active file exists.
4. **Tool** → `create_chart(path, chart_type, data_range, title, x_axis, y_axis, position)`
5. **Feedback** → Confirm chart creation and its location in the sheet.

## Tool Calls
```json
{
  "type": "create_chart",
  "chart_type": "bar",
  "data_range": "A1:C10",
  "title": "Sales Report",
  "x_axis": "Product",
  "y_axis": "Revenue",
  "position": "F2"
}
```

## Expected Result
```json
{
  "message": "chart_created",
  "chart_type": "bar",
  "data_range": "A1:C10",
  "position": "F2",
  "speech_text": "Bar chart created from A1 to C10."
}
```

## Error Handling
- If `data_range` is missing, attempt to auto-detect the data area or ask the user.
- If `chart_type` is unsupported, default to `bar` and inform the user.
- If the data range is empty or contains no numeric data, inform the user.
- If no active file is set, ask the user to open or create a file first.

## Memory Updates
- `recent_commands` → append `create_chart`.
