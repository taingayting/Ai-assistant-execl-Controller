# Sara on the web

This folder runs Sara's real backend (`../backend`) as a website with Streamlit.
Visitors can speak or type a command, see the sheet change, and download the `.xlsx`.

- Entry point: `web/streamlit_app.py`
- Packages: `web/requirements.txt`
- Python: 3.12
- Optional secret: `GROQ_API_KEY` (Streamlit → App settings → Secrets) lets Sara understand free-form sentences. Without it, Sara uses her rule parser, which understands all the example commands.

Run it on your computer:

```
pip install -r web/requirements.txt
streamlit run web/streamlit_app.py
```
