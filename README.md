## AI Application Development Code Review Assistant

An end-to-end Streamlit application that uses multiple specialized AI agents (CodeAnalysis, BugDetection, BestPractice) and the DeepSeek API to generate a comprehensive code review for uploaded source files, guided by an uploaded standards document.

### Features
- Upload a code file (Python, JavaScript, TypeScript, Java, or text)
- Upload a standards document (PDF/DOC/DOCX/TXT)
- One-click AI review combining:
  - Structure, readability, and style analysis vs. standards
  - Bug and anti-pattern detection with explanations
  - Best-practice suggestions with short references
- Markdown report with a final Code Quality Score out of 100
- Download the report as .txt or .pdf
- Expanders for raw JSON per agent

### Architecture
- `app.py`: Streamlit UI
- `deepseek_handler.py`: Thin DeepSeek API client with relaxed JSON parsing
- `agents/`: Agent modules
  - `code_analysis.py`
  - `bug_detection.py`
  - `best_practice.py`
- `review_manager.py`: Orchestrates agents and builds the final report
  - Optional CrewAI orchestration with LangChain `ChatOpenAI` if installed

### Setup
1. Python 3.9+
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set your DeepSeek API key in the environment:
```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

#### Optional configuration (OpenAI-compatible endpoints)
If your provider exposes an OpenAI-compatible API (like your reference snippet), configure via environment variables:

```bash
# Endpoint and model
export DEEPSEEK_BASE_URL="https://genailab.tcs.in"    # /v1 will be auto-appended unless disabled
export DEEPSEEK_MODEL="azure_ai/genailab-maas-DeepSeek-V3-0324"

# TLS verification (similar to httpx.Client(verify=False))
export DEEPSEEK_VERIFY_SSL=false

# If your endpoint is NOT OpenAI-compatible, disable auto /v1 append
export DEEPSEEK_OPENAI_COMPATIBLE=false

# Optional: add custom headers as JSON
export DEEPSEEK_EXTRA_HEADERS='{"x-tenant":"my-tenant"}'

# Optional timeouts/retries
export DEEPSEEK_TIMEOUT_SECONDS=60
export DEEPSEEK_MAX_RETRIES=2
```

Programmatic overrides are also supported:

```python
from deepseek_handler import DeepSeekClient

client = DeepSeekClient(
    api_key="<YOUR_KEY>",
    base_url="https://genailab.tcs.in",  # auto adds /v1 by default
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    verify_ssl=False,                     # like httpx.Client(verify=False)
    # openai_compatible=True,            # set False if your endpoint is not OpenAI-style
    # default_headers={"x-tenant": "my-tenant"},
)
```

### Run
```bash
streamlit run app.py
```

Open the printed local URL in your browser.

### Usage
1. Upload your code file (e.g., `main.py`).
2. Upload a standards document (e.g., `pep8_guide.pdf`).
3. Click "Generate Review". The app will analyze and render an AI-generated report plus raw JSON.
4. Download as `.txt` (always) or `.pdf` (if `fpdf2` installed).

### Notes
- The app attempts PDF text extraction via PyMuPDF first, then pdfplumber. For DOC/DOCX it uses `python-docx`.
- The DeepSeek model used is `deepseek-coder` by default; change in `deepseek_handler.py` if needed.
- If API returns non-JSON, the client tries relaxed extraction from fenced blocks or brace spans.
- If `crewai` and `langchain-openai` are installed, the app will try CrewAI-based orchestration automatically. Disable with `USE_CREWAI=false`.

### Extending
- Add more agents in `agents/` and wire them in `review_manager.py`.
- Enhance scoring in `ReviewManager._compute_quality_score` to fit your org.
- Improve prompts to specialize by language or framework.

### CrewAI mode (optional)
Set env for OpenAI-compatible usage and enable CrewAI orchestration (default true when deps exist):

```bash
export USE_CREWAI=true
export DEEPSEEK_API_KEY=your_key
export DEEPSEEK_BASE_URL=https://genailab.tcs.in
export DEEPSEEK_MODEL=azure_ai/genailab-maas-DeepSeek-V3-0324
export DEEPSEEK_VERIFY_SSL=false
```

### Security
- Uploaded files are processed in-memory only.
- Do not log sensitive code or keys. Ensure `DEEPSEEK_API_KEY` is kept secret.
