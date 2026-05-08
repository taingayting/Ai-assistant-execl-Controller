2️⃣ AI_CONTEXT.md (VERY IMPORTANT FOR COPILOT)
# AI Context — Sara Project

This file provides **context for AI coding assistants** working on the Sara project.

---

# Project Purpose

Sara is a **voice-controlled Excel automation assistant**.

Users speak or type commands, and Sara performs Excel operations automatically.

Example:

User:  
"Create a sales report and calculate total revenue."

Sara will:

1. Understand the command
2. Select a workflow
3. Execute deterministic Excel tools
4. Return results

---

# Core Architecture

Sara uses a **WAT + Memory architecture**.


User
↓
Agent
↓
Memory
↓
Workflow
↓
Tools
↓
Excel


---

# Important Design Rule


LLM = Planning
Tools = Execution


The LLM **decides what to do**, but **tools perform the actual operations**.

The Agent must **never manipulate Excel directly**.

Excel operations must always be executed through deterministic tools.

---

# Main Components

## Agent

Responsible for:

- Understanding user intent
- Selecting workflows
- Extracting parameters
- Coordinating execution

---

## Memory

Stores context between commands.

Example:


active_file = sales.xlsx
active_sheet = Sheet1
recent_commands = [...]


Memory allows users to run multi-step commands.

---

## Workflows

Workflows define structured task procedures.

They are stored in:


/workflows


Example workflow:


create_sales_report.md


Each workflow defines:

- objective
- inputs
- tool calls
- execution steps

---

## Tools

Tools perform deterministic Excel operations.

Examples:


create_excel_file()
add_column()
insert_data()
calculate_sum()
create_chart()


Tools are implemented in Python.

---

# Development Guidelines

When implementing features:

1. Agent decides workflow
2. Workflow defines steps
3. Tools execute Excel operations
4. Memory stores context

Never skip layers.

---