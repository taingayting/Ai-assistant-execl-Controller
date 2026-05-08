# Demo commands for end-to-end testing

Use these text commands with `/api/parse` (or speak them into the UI) to validate flows.

Examples:

- "create excel with headers \"Name,Score\""
  - Expected JSON: {"action":"create_table","headers":["Name","Score"]}

- "add row Tom ninety"
  - Expected JSON: {"action":"add_row","values":["Tom",90]}

- "add column Score"
  - Expected JSON: {"action":"add_column","name":"Score"}

- "delete the third row"
  - Expected JSON: {"action":"delete_row","index":3}

- "calculate average for column two"
  - Expected JSON: {"action":"calculate_average","column":2}

You can POST these to `/api/parse` as JSON: `{"text": "add row Tom ninety", "use_llm": false}`
