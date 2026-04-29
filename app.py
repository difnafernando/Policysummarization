import streamlit as st
import re
import io
from groq import Groq
from collections import Counter


# Page Configuration (must be first)

st.set_page_config(
    page_title="Policy Summarisation and Adaptation System",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Load Custom CSS


with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Configuration


GROQ_API_KEY = "ENTER_API_KEY"
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"



# NLP Preprocessing Functions


def preprocess_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def tokenize_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def score_sentences(sentences: list[str]) -> list[tuple[int, float, str]]:
    stopwords = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "is","are","was","were","be","been","being","have","has","had","do",
        "does","did","will","would","could","should","may","might","shall",
        "that","this","these","those","it","its","by","from","as","if",
        "not","no","nor","so","yet","both","either","neither","each",
        "than","such","into","through","during","before","after","above",
        "below","between","out","off","over","under","again","further",
        "then","once","also","about","which","there","their","they","we",
        "you","he","she","who","what","how","when","where","why","all",
        "any","few","more","most","other","some","such","only","own","same",
    }
    all_words = []
    for sent in sentences:
        words = re.findall(r"\b[a-z]{3,}\b", sent.lower())
        all_words.extend([w for w in words if w not in stopwords])

    freq = Counter(all_words)
    max_freq = max(freq.values()) if freq else 1

    scored = []
    for idx, sent in enumerate(sentences):
        words = re.findall(r"\b[a-z]{3,}\b", sent.lower())
        content_words = [w for w in words if w not in stopwords]
        if not content_words:
            scored.append((idx, 0.0, sent))
            continue
        score = sum(freq.get(w, 0) / max_freq for w in content_words) / len(content_words)
        scored.append((idx, score, sent))

    return scored


def extractive_summary(text: str, num_sentences: int = 7) -> str:
    cleaned = preprocess_text(text)
    sentences = tokenize_sentences(cleaned)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    scored = score_sentences(sentences)
    top_sentences = sorted(scored, key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: x[0])
    return " ".join(s for _, _, s in top_sentences)



# Groq API Functions


def generate_ai_summary(policy_text: str) -> str:
    extract = extractive_summary(policy_text, num_sentences=10)
    prompt = f"""You are a professional policy analyst.

Below is an extracted set of key sentences from a government or institutional policy document.

Your task is to produce a concise, structured summary with the following three sections:
1. Main Goals
2. Key Measures and Strategies
3. Overall Direction

STRICT FORMATTING RULES:
- Do NOT use any *, **, #, or markdown symbols anywhere
- Write section titles as plain text only like: "1. Main Goals"
- Each section should be 2-4 sentences in plain paragraph form
- No bullet points, no bold, no italic, no special characters

Extracted policy text:
{extract}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=700,
    )
    return response.choices[0].message.content.strip().replace("*", "")


def generate_scenario_draft(summary: str, scenario_name: str, scenario_description: str) -> str:
    prompt = f"""You are a senior policy consultant.

You have been given a summary of an existing policy document. Your task is to produce a formal adapted policy draft tailored to the specific scenario described below.

The adapted policy must:
- Be written in formal policy language
- Be structured with a brief Introduction, Objectives, Key Provisions, and Implementation Notes
- Clearly reflect the priorities and constraints of the given scenario
- Differ in focus and emphasis from a generic version of the policy
- Do NOT use any *, **, #, or markdown symbols anywhere

Policy Summary:
{summary}

Scenario Name: {scenario_name}
Scenario Description: {scenario_description}

Write the adapted policy draft now.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=900,
    )
    return response.choices[0].message.content.strip().replace("*", "")



# Download Helper


def text_to_download(content: str, filename: str):
    buffer = io.BytesIO(content.encode("utf-8"))
    st.download_button(
        label=f"Download {filename}",
        data=buffer,
        file_name=filename,
        mime="text/plain",
    )


# Page Header

st.title("Policy Summarisation and Adaptation System")
st.caption(
    "General Sir John Kotelawala Defence University | BSc Applied Data Science Communication | "
    "Assignment I – Intake 41 | LB3114"
)
st.markdown("---")


# Session State Initialisation


if "policy_text" not in st.session_state:
    st.session_state["policy_text"] = ""
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "drafts" not in st.session_state:
    st.session_state["drafts"] = {}


# Layout: Two Columns


left_col, right_col = st.columns([1, 1], gap="large")


# LEFT PANEL — Policy Summarisation


with left_col:
    st.subheader("Policy Summarisation")
    st.markdown(
        "Paste or upload a policy document. The system will preprocess the text "
        "and generate a structured summary using NLP and a large language model."
    )

    input_method = st.radio(
        "Input method",
        ["Paste text", "Upload file"],
        horizontal=True,
        key="input_method",
    )

    if input_method == "Paste text":
        policy_input = st.text_area(
            "Policy document text",
            height=300,
            placeholder="Paste the full text of the policy document here...",
            key="paste_input",
        )
        if policy_input:
            st.session_state["policy_text"] = policy_input
    else:
        uploaded_file = st.file_uploader(
            "Upload a .txt file",
            type=["txt"],
            key="file_upload",
        )
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            st.session_state["policy_text"] = content
            st.success(f"File loaded: {uploaded_file.name} ({len(content)} characters)")

    if st.session_state["policy_text"]:
        word_count = len(st.session_state["policy_text"].split())
        sentence_count = len(tokenize_sentences(st.session_state["policy_text"]))
        st.caption(f"Document statistics — Words: {word_count} | Sentences: {sentence_count}")

    num_extract_sentences = st.slider(
        "Extractive sentences passed to AI (preprocessing depth)",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="Controls how many key sentences are extracted before being sent to the AI for summarisation.",
    )

    summarise_btn = st.button("Generate Summary", type="primary", use_container_width=True)

    if summarise_btn:
        if not st.session_state["policy_text"].strip():
            st.warning("Please provide a policy document first.")
        else:
            with st.spinner("Preprocessing and summarising..."):
                extractive_summary(st.session_state["policy_text"], num_sentences=num_extract_sentences)
                ai_summary = generate_ai_summary(st.session_state["policy_text"])
                st.session_state["summary"] = ai_summary

    # --- Display Summary ---
    if st.session_state["summary"]:
        clean_summary = st.session_state["summary"].replace("*", "")
        clean_summary = re.sub(r"^#{1,6}\s*", "", clean_summary, flags=re.MULTILINE)
        clean_summary = re.sub(r"[ \t]+", " ", clean_summary).strip()
        st.session_state["summary"] = clean_summary

        st.markdown("### :blue[Generated Summary]")

        heading_keywords = ["Main Goals", "Key Measures", "Overall Direction"]
        for line in clean_summary.split("\n"):
            line = line.strip()
            if not line:
                st.write("")
            elif any(keyword in line for keyword in heading_keywords):
                st.markdown(f"**:blue[{line}]**")
            else:
                st.write(line)

        st.markdown("---")
        text_to_download(st.session_state["summary"], "policy_summary.txt")

        with st.expander("View extractive preprocessing output"):
            st.text(
                extractive_summary(
                    st.session_state["policy_text"], num_sentences=num_extract_sentences
                )
            )



# RIGHT PANEL — Scenario-Based Policy Generation


with right_col:
    st.subheader("Scenario-Based Policy Generation")
    st.markdown(
        "Define a scenario that changes the focus or constraints of the policy. "
        "The system will generate an adapted policy draft using the summary from the left panel."
    )

    if not st.session_state["summary"]:
        st.info("Generate a policy summary on the left panel before creating scenario drafts.")
    else:
        st.markdown("#### Define a Scenario")

        preset_scenarios = {
            "Custom (enter manually)": ("", ""),
            "Rural Community Adaptation": (
                "Rural Community Adaptation",
                "The policy must be adapted for implementation in rural or semi-urban communities "
                "where digital infrastructure is limited, literacy levels vary, and enforcement "
                "capacity is minimal. Priority is placed on grassroots participation, local "
                "governance structures, and low-cost delivery mechanisms.",
            ),
            "Youth and Education Focus": (
                "Youth and Education Focus",
                "The policy is being adapted for application within the national education sector, "
                "targeting school-age children and young adults. The emphasis shifts to awareness, "
                "skill-building, long-term behavioural change, and integration with school curricula.",
            ),
            "Climate and Environmental Constraints": (
                "Climate and Environmental Constraints",
                "The policy must be reoriented to account for the impacts of climate change, "
                "environmental degradation, and natural disaster risk. Provisions should "
                "prioritise sustainability, resilience, and environmental compliance.",
            ),
            "Post-Crisis Economic Recovery": (
                "Post-Crisis Economic Recovery",
                "The policy is being adapted for a period of post-economic crisis where fiscal "
                "constraints are severe, public trust in institutions is low, and rapid stabilisation "
                "is the primary objective. Emphasis is on phased rollout, cost-efficiency, and "
                "transparent accountability mechanisms.",
            ),
        }

        selected_preset = st.selectbox(
            "Select a preset scenario or define your own",
            list(preset_scenarios.keys()),
            key="preset_select",
        )

        default_name, default_desc = preset_scenarios[selected_preset]

        scenario_name = st.text_input(
            "Scenario name",
            value=default_name,
            placeholder="e.g., Rural Community Adaptation",
            key="scenario_name",
        )

        scenario_description = st.text_area(
            "Scenario description",
            value=default_desc,
            height=120,
            placeholder="Describe the context, target audience, constraints, or priorities...",
            key="scenario_description",
        )

        generate_btn = st.button(
            "Generate Adapted Policy Draft",
            type="primary",
            use_container_width=True,
        )

        if generate_btn:
            if not scenario_name.strip() or not scenario_description.strip():
                st.warning("Please provide both a scenario name and description.")
            else:
                with st.spinner(f"Generating adapted policy draft for: {scenario_name}..."):
                    draft = generate_scenario_draft(
                        st.session_state["summary"],
                        scenario_name,
                        scenario_description,
                    )
                    st.session_state["drafts"][scenario_name] = draft

        if st.session_state["drafts"]:
            st.markdown("---")
            st.markdown("#### :blue[Generated Policy Drafts]")
            st.caption(f"Total drafts generated: {len(st.session_state['drafts'])}")

            draft_keywords = ["Introduction", "Objectives", "Key Provisions", "Implementation Notes"]

            for name, draft in st.session_state["drafts"].items():
                clean_draft = draft.replace("*", "")
                clean_draft = re.sub(r"^#{1,6}\s*", "", clean_draft, flags=re.MULTILINE)
                clean_draft = re.sub(r"[ \t]+", " ", clean_draft).strip()
                st.session_state["drafts"][name] = clean_draft

                with st.expander(f"Scenario: {name}", expanded=(list(st.session_state["drafts"].keys())[-1] == name)):
                    for line in clean_draft.split("\n"):
                        line = line.strip()
                        if not line:
                            st.write("")
                        elif any(keyword in line for keyword in draft_keywords):
                            st.markdown(f"**:blue[{line}]**")
                        else:
                            st.write(line)

                    st.markdown("---")
                    text_to_download(
                        clean_draft,
                        filename=f"policy_draft_{name.lower().replace(' ', '_')}.txt",
                    )

            st.markdown("---")
            all_content = ""
            for name, draft in st.session_state["drafts"].items():
                all_content += f"=== Scenario: {name} ===\n\n{draft}\n\n\n"
            text_to_download(all_content, "all_policy_drafts.txt")

            if st.button("Clear all drafts", use_container_width=True):
                st.session_state["drafts"] = {}
                st.rerun()



# Footer

st.markdown("---")
st.caption(
    "System uses TF-based extractive NLP preprocessing followed by LLM-powered "
    "abstractive summarisation via Groq (LLaMA 3). "
    "All outputs are generated from the provided policy document and are not official government documents."
)
