# ChiralNet

**Agentic vision-language reasoning for identifying charge order in quantum materials.**

ChiralNet classifies scanning tunneling microscopy/spectroscopy (STM/STS) images as
**CDW**, **Chiral CDW**, or **Non-CDW** — *without any task-specific training*. Instead of
a single end-to-end classifier, each measurement is routed through a fixed pipeline of
specialist large-language-model (LLM) agents mirroring how an expert microscopist reasons:
moiré screening, real-space morphology, tool-computed reciprocal-space analysis, motif
handedness, and spectroscopic chirality. A final *judge* agent adjudicates their structured
evidence hierarchically and returns an auditable label, confidence, and rationale.

---

## Repository structure

| File | Purpose |
| --- | --- |
| `ui.py` | Streamlit app — primary interface. `streamlit run ui.py` |
| `core_pipeline.py` | LangGraph orchestration, agents, ensemble voting, judge. |
| `retrieval.py` | CLIP retriever: indexing, cosine similarity, metadata proximity. |
| `fft_core.py` | Windowed 2-D FFT, calibration, diagnostic figures. |
| `fft_peak_analysis.py` | Peak detection, CDW-evidence assessment, standalone CLI. |
| `curated_dataset.json` | 209-entry retrieval corpus (metadata + image references). |

---

## Installation

Python 3.10+. No `requirements.txt` yet; install:

```bash
pip install langgraph openai streamlit numpy scipy matplotlib Pillow
pip install torch open_clip_torch   # optional: enables CLIP retrieval
```

`torch` and `open_clip_torch` are optional — without them the retriever returns no
precedents and the judge is instructed not to adjust for missing retrieval; the rest of the
pipeline is unaffected. On first use, `open_clip` downloads `ViT-L-14` (`openai`) weights.
An **OpenAI API key** is required; the app prompts at runtime, nothing is hard-coded.

---

## Usage

**Streamlit app**

```bash
streamlit run ui.py
```

Paste your API key, upload a topograph (required) and optionally a dI/dV map, enter
acquisition metadata, and click **Run ChiralNet**. Supplying **both** topograph width and
height (nm) calibrates the FFT; otherwise frequencies are reported in cycles/pixel and
labeled `UNCALIBRATED`.

**Standalone FFT CLI**

```bash
python fft_peak_analysis.py topograph.png --Lx 10 --Ly 10 \
    --assess-cdw --save-figure fft.png --no-show
```

Run `--help` for all flags.

**Programmatic**

```python
import base64
from openai import OpenAI
import core_pipeline
from core_pipeline import agentic_graph, AgentState

core_pipeline.client = OpenAI(api_key="sk-...")
b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()

result = agentic_graph.invoke({
    "input_image_base64": b64("topograph.png"),
    "input_image_ext": ".png",
    "input_didv_base64": b64("didv_map.png"),   # or None
    "metadata": {"v_topo": -0.05, "scale_topo": 2.0, "fov_x": 10.0, "fov_y": 10.0},
})
print(result["final_label"], result["confidence"], result["explanation"])
```
