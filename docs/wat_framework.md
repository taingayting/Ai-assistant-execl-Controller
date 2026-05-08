4️⃣ docs/wat_framework.md
# WAT Framework Implementation

Sara is built using the **WAT Architecture**.

WAT stands for:


Workflows
Agents
Tools


---

# Workflows

Workflows define structured procedures for tasks.

Example workflow:


create_sales_report.md


Example steps:

1. Create Excel file
2. Add sheet
3. Add columns
4. Format header

Workflows ensure predictable execution.

---

# Agents

Agents interpret user commands and decide which workflow to execute.

Responsibilities:

- natural language understanding
- workflow selection
- parameter extraction
- memory lookup

Agents rely on LLM reasoning.

---

# Tools

Tools execute deterministic operations.

Examples:


create_excel_file()
add_column()
insert_data()
calculate_sum()
create_chart()


Tools do not perform reasoning.

They only execute predefined actions.

---

# Why WAT Architecture?

WAT improves:

- reliability
- maintainability
- modularity
- debugging

Instead of allowing the AI to generate actions dynamically, workflows def: