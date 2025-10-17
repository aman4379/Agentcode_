import os
import io
from typing import Optional

import streamlit as st

from review_manager import ReviewManager


st.set_page_config(page_title="AI Code Review Assistant", layout="wide")


def _read_uploaded_file(uploaded) -> str:
    if not uploaded:
        return ""
    try:
        content = uploaded.read()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return str(content)
    except Exception:
        return ""


def _extract_text_from_pdf_or_doc(uploaded) -> str:
    if not uploaded:
        return ""
    name = (uploaded.name or "").lower()
    # Prefer PyMuPDF (fitz) if installed
    try:
        if name.endswith(".pdf"):
            import fitz  # PyMuPDF

            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            texts = []
            for page in doc:
                texts.append(page.get_text())
            return "\n".join(texts)
    except Exception:
        pass

    # Fallback to pdfplumber for PDF
    try:
        if name.endswith(".pdf"):
            import pdfplumber

            uploaded.seek(0)
            with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
                texts = []
                for page in pdf.pages:
                    texts.append(page.extract_text() or "")
                return "\n".join(texts)
    except Exception:
        pass

    # For doc/docx we can try python-docx if available
    try:
        if name.endswith(".docx") or name.endswith(".doc"):
            from docx import Document

            uploaded.seek(0)
            document = Document(io.BytesIO(uploaded.read()))
            return "\n".join(p.text or "" for p in document.paragraphs)
    except Exception:
        pass

    # As a last resort, try to read as text
    uploaded.seek(0)
    return _read_uploaded_file(uploaded)


st.title("AI Application Development Code Review Assistant")
st.caption("Upload code and standards, then generate an AI-driven review.")

api_key_present = bool(os.getenv("DEEPSEEK_API_KEY"))
if not api_key_present:
    st.warning("DEEPSEEK_API_KEY not found in environment. Set it before generating reviews.")

col1, col2 = st.columns(2)
with col1:
    code_file = st.file_uploader("Upload a code file (Python, JS, Java, etc.)", type=["py", "js", "ts", "java", "txt"])  # allow .txt for demos
with col2:
    standards_file = st.file_uploader("Upload coding standards (PDF/DOC/DOCX/TXT)", type=["pdf", "doc", "docx", "txt"])  # allow txt for demos

review_button = st.button("Generate Review", type="primary", disabled=not api_key_present)

if review_button:
    if not code_file or not standards_file:
        st.error("Please upload both a code file and a standards document.")
    else:
        with st.spinner("Analyzing with AI agents..."):
            try:
                # Read inputs
                code_text = _read_uploaded_file(code_file)
                standards_text = _extract_text_from_pdf_or_doc(standards_file)
                filename = code_file.name

                manager = ReviewManager()
                result = manager.analyze(code_text=code_text, filename=filename, standards_text=standards_text)

                st.success("Review complete.")
                st.subheader("AI Review Report")
                st.markdown(result["report_markdown"])  # Render markdown

                # Download buttons
                text_bytes = result["report_markdown"].encode("utf-8")
                st.download_button(
                    label="Download Report (.txt)",
                    data=text_bytes,
                    file_name=f"review_{filename}.txt",
                    mime="text/plain",
                )

                pdf_bytes = manager.export_report_pdf(result["report_markdown"]) or b""
                if pdf_bytes:
                    st.download_button(
                        label="Download Report (.pdf)",
                        data=pdf_bytes,
                        file_name=f"review_{filename}.pdf",
                        mime="application/pdf",
                    )
                else:
                    st.info("PDF export dependency not installed. Install 'fpdf' to enable PDF downloads.")

                # Expanders for raw JSON sections
                with st.expander("Raw JSON: Style & Analysis"):
                    st.json(result.get("code", {}))
                with st.expander("Raw JSON: Bugs"):
                    st.json(result.get("bugs", {}))
                with st.expander("Raw JSON: Best Practices"):
                    st.json(result.get("best_practices", {}))

            except Exception as e:
                st.exception(e)
