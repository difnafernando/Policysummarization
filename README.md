# Policy Summarisation and Adaptation System

**General Sir John Kotelawala Defence University**
BSc Applied Data Science Communication | Assignment I – Intake 41 | LB3114

---

## Overview

The Policy Summarisation and Adaptation System is a web-based application built with Streamlit that allows users to upload or paste government and institutional policy documents, generate structured AI-powered summaries, and produce scenario-based adapted policy drafts.

The system combines traditional NLP preprocessing techniques (TF-based extractive summarisation) with a large language model (LLaMA 3 via Groq API) to produce high-quality, formal policy outputs.

---

## Features

- Paste or upload `.txt` policy documents
- Automatic word and sentence count statistics
- TF-based extractive NLP preprocessing
- AI-generated structured policy summaries with three sections:
  - Main Goals
  - Key Measures and Strategies
  - Overall Direction
- Scenario-based adapted policy draft generation
- Four preset scenarios included:
  - Rural Community Adaptation
  - Youth and Education Focus
  - Climate and Environmental Constraints
  - Post-Crisis Economic Recovery
- Custom scenario input support
- Download summaries and drafts as `.txt` files
- Clean formatted output with no markdown symbols

---

## Project Structure

```
📁 project_folder/
   ├── nlp.py              # Main Streamlit application
   ├── style.css           # Custom CSS stylesheet
   ├── requirements.txt    # Python dependencies
   └── README.md           # Project documentation
```

---

## Requirements

- Python 3.10 or higher
- Internet connection (for Groq API)
- Groq API key

---

## Installation

**Step 1 — Clone or download the project folder**

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Add your Groq API key**

Open `nlp.py` and replace the API key on this line:
```python
GROQ_API_KEY = "your_groq_api_key_here"
```

You can get a free Groq API key at: https://console.groq.com

**Step 4 — Run the application**
```bash
streamlit run nlp.py
```

**Step 5 — Open in browser**

The app will open automatically at:
```
http://localhost:8501
```

---

## How to Use

### Left Panel — Policy Summarisation
1. Select **Paste text** or **Upload file**
2. Paste or upload your policy document (`.txt` format)
3. Adjust the extractive sentences slider if needed
4. Click **Generate Summary**
5. View the structured summary with bold blue headings
6. Download the summary using the download button

### Right Panel — Scenario-Based Policy Generation
1. Generate a summary first on the left panel
2. Select a preset scenario or enter a custom one
3. Provide a scenario name and description
4. Click **Generate Adapted Policy Draft**
5. View and download the adapted draft
6. Multiple drafts can be generated and compared

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| Streamlit | Web application framework |
| Groq API | LLM inference (LLaMA 3.1 8B) |
| Regex (re) | Text preprocessing and cleaning |
| Collections (Counter) | Term frequency scoring |

---

## NLP Pipeline

```
Raw Policy Text
      ↓
Preprocessing (lowercase, normalize whitespace)
      ↓
Sentence Tokenization (regex-based)
      ↓
TF Scoring (term frequency extractive scoring)
      ↓
Top N Sentences Selected
      ↓
Groq LLM (LLaMA 3.1 8B) → Structured Summary
      ↓
Scenario Input → Adapted Policy Draft
```

---

## Notes

- All outputs are AI-generated and are not official government documents
- The Groq API requires an active internet connection
- The `style.css` file must be in the same folder as `nlp.py`
- Supported input format: plain text (`.txt`) files only

---

## License

This project was developed for academic purposes at the General Sir John Kotelawala Defence University. Not intended for commercial use.
