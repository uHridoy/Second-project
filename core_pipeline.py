import base64
import json
import os
import tempfile
from typing import TypedDict, Literal, Optional, List, Dict, Any, Callable

from langgraph.graph import StateGraph, END

try:
    import matplotlib
    matplotlib.use("Agg")

    matplotlib.rcParams.update({
        "axes.titlepad": 16.0,     
        "axes.titlesize": 11.0,
        "figure.titlesize": 13.0,
        "figure.subplot.hspace": 0.45,
        "figure.subplot.wspace": 0.35,
    })

    from fft_core import (
        analyze_image_fft,
        plot_peak_diagnostics,
    )
    from fft_peak_analysis import (
        detect_peaks,
        assess_cdw_evidence,
    )
    FFT_ANALYZER_AVAILABLE = True
    FFT_ANALYZER_IMPORT_ERROR = None
except Exception as _fft_exc:  
    FFT_ANALYZER_AVAILABLE = False
    FFT_ANALYZER_IMPORT_ERROR = str(_fft_exc)

from retrieval import retriever_agent

client = None

class AgentState(TypedDict, total=False):
    input_image_base64: str
    input_image_ext: Optional[str]
    input_didv_base64: Optional[str]
    metadata: Dict[str, Any]
    retrieval_results: List[Dict[str, Any]]
    moiré_analysis: Dict[str, Any]
    visual_analysis: Dict[str, Any]
    fft_numerical: Dict[str, Any]
    fft_peaks_figure_base64: Optional[str]
    fft_diagnostic_figure_base64: Optional[str]
    fourier_analysis: Dict[str, Any]
    Chiral_topo_analysis: Dict[str, Any]
    spectroscopy_analysis: Dict[str, Any]
    final_decision: Dict[str, Any]
    final_label: Literal["CDW", "Chiral CDW", "Non-CDW", "Inconclusive"]
    confidence: float
    explanation: str
    full_final_prompt: str

def run_ensemble(agent_func: Callable[[AgentState], AgentState], state: AgentState, agent_name: str, n_runs: int = 5) -> AgentState:
    results = []

    for i in range(n_runs):
        try:
            temp_state = state.copy()
            agent_func(temp_state)
            if agent_name in temp_state:
                results.append(temp_state[agent_name])
        except Exception:
            continue

    if not results:
        state[agent_name] = {"final_label": "None", "confidence": 0, "explanation": "All ensemble runs failed."}
        return state

    label_counts = {}
    total_conf = 0

    for res in results:
        label = res.get("final_label", "None")
        label_counts[label] = label_counts.get(label, 0) + 1
        total_conf += res.get("confidence", 50)

    final_label = max(label_counts, key=label_counts.get)
    avg_confidence = round(total_conf / len(results))

    best_exp = max(results, key=lambda x: x.get("confidence", 0)).get("explanation", "")

    state[agent_name] = {
        "final_label": final_label,
        "confidence": avg_confidence,
        "explanation": f"[Consensus across {len(results)} runs] {best_exp}",
    }
    return state

def b64_from_upload(file):
    if file is None: return None
    return base64.b64encode(file.getvalue()).decode("utf-8")

def _convert_responses_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]

    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}]

    converted: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            converted.append({"type": "input_text", "text": item})
            continue

        if not isinstance(item, dict):
            converted.append({"type": "input_text", "text": str(item)})
            continue

        item_type = item.get("type")
        if item_type == "text":
            converted.append({"type": "input_text", "text": item.get("text", "")})
        elif item_type == "image_url":
            image_url = item.get("image_url", {}) or {}
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if url:
                converted.append({"type": "input_image", "image_url": url})
        else:
            converted.append({"type": "input_text", "text": json.dumps(item, ensure_ascii=False, default=str)})

    return converted


def call_json_agent(system_prompt: str, user_content: Any, max_output_tokens: int = 700) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("Missing OpenAI API key")

    content = _convert_responses_content(user_content)
    content.insert(0, {
        "type": "input_text",
        "text": "Return JSON only. Output must be a valid JSON object."
    })

    response = client.responses.create(
        model="gpt-5.6",
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        text={"format": {"type": "json_object"}},
        max_output_tokens=max_output_tokens,
    )

    return json.loads(response.output_text)

def moiré_agent(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": """
You are an expert STM moiré pattern detection agent. Your job is to distinguish true moiré superlattices (structural interference patterns, Non-CDW) from Charge Density Wave (CDW) modulations and atomic lattices in STM topography images.

True moiré pattern (classify as "Moiré"):
1. Very large-scale periodicity, typically 5–30+ nm (much larger than atomic lattice).
2. Characteristic interference "beating" pattern: smooth, slowly varying contrast envelope over the atomic lattice.
3. Often shows hexagonal or triangular moiré lattice with AA/AB stacking contrast variation (bright spots in high-symmetry stacking regions).
4. The underlying atomic lattice is usually still clearly visible within the moiré cells.
5. Highly uniform and symmetric over large areas.
6. Common in twisted bilayer graphene, TMD heterostructures, or lattice-mismatched systems.

CDW patterns (classify as "Non-Moiré"):
1. Modulation wavelength is usually 2–5 times the atomic lattice constant (much smaller than typical moiré).
2. In triangular/kagome lattices: bright triangular or star-like clusters with strong local electronic contrast.
3. Stripe-like (1Q), checkerboard (2Q), or triangular (3Q) electronic modulations.
4. Often sharper local contrast, domain walls, or discommensurations.
5. Stronger electronic appearance rather than geometric interference.

Atomic lattice only or artifacts (classify as "Non-Moiré"):
1. Pure atomic resolution without larger-scale modulation.
2. Scan noise, drift, tip artifacts, or random contrast.

Decision rules:
1. Only classify as "Moiré" if the modulation scale is much larger than the atomic lattice and shows interference beating / stacking contrast.
2. Triangular patterns with bright spots on a ~√13 × √13 or similar CDW superlattice are CDW (Non-Moiré).
3. Stripe-like modulations are almost always CDW (Non-Moiré).
4. If the scale is ambiguous or the modulation looks electronic rather than geometric interference, default to "Non-Moiré".
5. Be conservative: only output "Moiré" when the moiré fingerprint is strong and unambiguous.
6. A uniform triangular array of broad bright clusters alone is not evidence of moiré; if no underlying atomic lattice is separately visible within the larger cells, classify it as “Non-Moiré.”
"""
        },
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]

    system = """
    Analyze this STM topograph carefully. Do not rely on metadata or other agents.
    Rules:
    1. Allowed labels: Moiré or Non-Moiré.
    2. Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """

    state["moiré_analysis"] = call_json_agent(system, prompt)
    return state

def visual_analyst(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": """
You are an expert STM image classifier. Your task is to classify the current STM topography image according to the visible morphology in the image itself. Do not override the visible morphology with abstract physical assumptions.

Critical rule:
The goal is to match the morphology that is visibly present in the current image, as a careful human annotator would label it from the topograph itself.

Instructions:
1. First, inspect only the current image and identify what is visibly present:
   i. atomic corrugation
   ii. stripe-like modulation
   iii. checkerboard/grid-like modulation
   iv. triangular/hexagonal superstructure
   v. domain pattern
   vi. defects
   vii. step edges / terraces
   viii. drift distortion
   ix. scan-line noise
   x. tip artifact
   xi. long-wavelength modulation beyond the atomic lattice
   xii. short-range repeated texture

2. Classify based on the dominant visible morphology, not on strict proof of microscopic origin.
   i. If a coherent repeated superstructure or modulation is visibly present, classify it according to the corresponding modulation/superlattice/CDW-like label.
   ii. If the image visibly contains a non-atomic repeating pattern that is spatially coherent, use the corresponding modulation/superlattice/CDW-like label.
   iii. Use negative or non-CDW labels when the image is dominated by atomic lattice only, artifacts, isolated defects, random contrast, drift, scan-line noise, or tip effects.

3. Distinguish between:
   i. real repeated modulation visible in the image
   ii. isolated defects
   iii. drift or line-by-line scan artifacts
   iv. tip distortions
   v. random contrast variation
   vi. pure atomic lattice without larger-scale structure

4. Decision policy:
   i. prioritize what a careful human annotator would label from the image appearance
   ii. label “present / likely present” when the target morphology is coherent, repeated, and visibly distinct from the atomic lattice or artifacts
   iii. label “absent / likely absent” when the image is dominated by lattice-only structure, artifacts, noise, or non-periodic contrast
   iv. do not force a CDW-like label from weak, random, or artifact-like texture

5. If uncertainty exists, do not immediately force the image into the opposite class.
   Instead, state whether the morphology is:
   i. clear/present
   ii. likely present
   iii. uncertain/weak possible modulation
   iv. likely absent
   v. absent
"""
        },
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]
    system = """
    You are an independent STM visual morphology agent. Analyze the image as the STM topograph. Do not rely on metadata or other agents.
    Rules:
    1. Allowed labels: CDW, Non-CDW.
    2. Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """

    state["visual_analysis"] = call_json_agent(system, prompt)
    return state

def fft_tool(state: AgentState) -> AgentState:
    if not FFT_ANALYZER_AVAILABLE:
        state["fft_numerical"] = {
            "status": "unavailable",
            "message": f"fft_analyzer could not be imported: "
                       f"{FFT_ANALYZER_IMPORT_ERROR}. Place fft_core.py "
                       "and fft_peak_analysis.py "
                       "next to this script and install numpy/scipy/"
                       "matplotlib/Pillow.",
        }
        return state

    try:
        raw_bytes = base64.b64decode(state["input_image_base64"])
        ext = (state.get("input_image_ext") or ".png").lower()
        if ext == ".tif":
            ext = ".tiff"
        if ext not in (".png", ".jpg", ".jpeg", ".tiff"):
            ext = ".png"
        meta = state.get("metadata") or {}
        Lx, Ly = meta.get("fov_x"), meta.get("fov_y")
        if Lx is None or Ly is None or not Lx or not Ly:
            Lx = Ly = None

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "topograph" + ext)
            with open(img_path, "wb") as f:
                f.write(raw_bytes)
            diag_path = os.path.join(tmpdir, "fft_diagnostic.png")
            peaks_path = os.path.join(tmpdir, "fft_peaks.png")

            results = analyze_image_fft(img_path, Lx=Lx, Ly=Ly,
                                        show=False, save_figure=diag_path)
            detection = detect_peaks(results)
            assessment = assess_cdw_evidence(results, detection)
            plot_peak_diagnostics(results, detection,
                                  show=False, save_figure=peaks_path)

            with open(diag_path, "rb") as f:
                state["fft_diagnostic_figure_base64"] = \
                    base64.b64encode(f.read()).decode("utf-8")
            with open(peaks_path, "rb") as f:
                state["fft_peaks_figure_base64"] = \
                    base64.b64encode(f.read()).decode("utf-8")

        accepted = detection["accepted"]
        rejected = detection["rejected"]
        reject_counts: Dict[str, int] = {}
        for p in rejected:
            r = str(p.get("reject_reason"))
            reject_counts[r] = reject_counts.get(r, 0) + 1

        top_peaks = sorted(accepted, key=lambda p: -p["snr"])[:12]
        peak_rows = [{
            "fx": round(p["fx"], 6), "fy": round(p["fy"], 6),
            "qx": round(p["qx"], 6) if p["qx"] is not None else None,
            "qy": round(p["qy"], 6) if p["qy"] is not None else None,
            "snr": round(float(p["snr"]), 1),
            "width_bins": round(float(p["width_bins"]), 2)
            if p["width_bins"] == p["width_bins"] else None,
            "pairing_score": round(float(p["pairing_score"]), 3),
        } for p in top_peaks]

        pair_rows = [{
            "f_radius": round(r["f_radius"], 6),
            "q_radius": round(r["q_radius"], 6)
            if r["q_radius"] is not None else None,
            "angle_deg": round(r["angle_deg"], 1),
            "min_snr": round(r["min_snr"], 1),
            "pairing_score": round(r["pairing_score"], 3),
            "status": r["status"],
            "flags": (r["failures"]
                      + (["axis_aligned"] if r["axis_aligned"] else [])
                      + (["jpeg_frequency"] if r["jpeg_suspect"] else [])),
        } for r in assessment["pairs"]]

        Ny, Nx = results["magnitude"].shape
        state["fft_numerical"] = {
            "status": "ok",
            "pipeline": "fft_analyzer (mean-subtracted, 2D-Hanning-"
                        "windowed FFT; conservative peak detection; "
                        "symmetric-pair CDW-evidence assessment)",
            "calibrated": results["calibrated"],
            "frequency_units": ("cycles/unit and rad/unit (calibrated)"
                                if results["calibrated"]
                                else "cycles/pixel (UNCALIBRATED — no "
                                     "physical scale supplied)"),
            "image_shape_px": {"Nx": Nx, "Ny": Ny},
            "field_of_view": {"Lx": Lx, "Ly": Ly},
            "n_accepted_peaks": len(accepted),
            "n_rejected_peaks": len(rejected),
            "rejection_reasons": reject_counts,
            "strongest_accepted_peaks": peak_rows,
            "n_symmetric_pairs": assessment["n_pairs_total"],
            "n_clean_pairs": assessment["n_pairs_clean"],
            "n_suspect_pairs": assessment["n_pairs_suspect"],
            "symmetric_pairs": pair_rows,
            "possible_q_organization": assessment["q_organization"],
            "artifact_warnings": assessment["warnings"],
            "cdw_evidence_level": assessment["evidence_level"],
            "note": "This is CDW-compatible FFT evidence, NOT a "
                    "definitive CDW classification: atomic-lattice, "
                    "structural, and moire periodicities produce "
                    "identical symmetric FFT peak pairs.",
        }
    except Exception as exc:
        state["fft_numerical"] = {"status": "failed", "message": str(exc)}
    return state
  

def fourier_agent(state: AgentState):
    fft_summary = state.get("fft_numerical") or {
        "status": "unavailable",
        "message": "Numerical FFT was not computed.",
    }

    prompt = [
        {
            "type": "text",
            "text": """
You are an expert STM Fourier-analysis classifier for Charge Density Waves (CDW).

Core principle: Distinguish atomic lattice from CDW superlattice.

Decision rules:

1. Strong FFT Evidence (CDW evidence level = Strong):
   i. This indicates a clear periodic modulation.
   ii. Now evaluate scale and morphology to decide if it's atomic or CDW.

2. Key Distinction:
   i. Atomic Lattice: Regular, high-frequency peaks + real-space shows uniform tight packing of atoms/dots with no larger repeating contrast modulation.
   ii. CDW: Lower-frequency peaks (superlattice) + real-space shows coherent electronic modulation (triangular clusters, stars, stripes, or honeycomb contrast) on top of or instead of pure atomic corrugation.

3. Uncalibrated Images:
   i. If real-space clearly shows a larger-scale superstructure (e.g. bright triangular domains, star patterns, or modulation wavelength visibly ~3-5x atomic spacing), classify as CDW.
   ii. Pure uniform atomic grid, classify as Non-CDW.

4. Chirality:
   i. Chiral CDW only if 3Q + visible local handedness (skewed/lopsided triangles, pinwheels).
   ii. Clean symmetric triangles, classify as CDW.

"""
        },
        {
            "type": "text",
            "text": "Numerical FFT results (computed by fft_analyzer):\n"
                    + json.dumps(fft_summary, indent=2, default=str),
        },
        {"type": "text", "text": "Image 1: STM topograph (real space)"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]

    if state.get("fft_peaks_figure_base64"):
        prompt.append({
            "type": "text",
            "text": "Image 2: computed FFT log-magnitude (visualization "
                    "only) with accepted peaks (green circles, cyan "
                    "pairing lines) and rejected peaks (red x with "
                    "reason).",
        })
        prompt.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['fft_peaks_figure_base64']}"},
        })

    system = """
    You are the STM Fourier agent. You are given computed
    numerical FFT results from a validated pipeline plus the images.
    Ground your label and explanation in those numbers. Do not rely on
    metadata or other agents.
    Rules:
    1. Allowed labels: CDW, Chiral CDW, Non-CDW.
    2. Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """

    state["fourier_analysis"] = call_json_agent(system, prompt)
    return state

def motif_chirality_agent(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": """
You are an STM topography classifier.

Classify as not Chiral CDW if the pattern is mainly:
1. clean repeated triangles
2. symmetric triangular / hexagonal / honeycomb / dot lattice
3. bright or dark triangular markers with no internal asymmetry
4. cellular honeycomb contrast with dark centers and bright rims
5. scan/grid/pixel artifacts over a regular lattice
6. triangles that are identical or equivalent by translation, rotation, or contrast change
7. simple single-peak elongated or oval spots with no resolvable internal structure
8. paired or lopsided features whose internal displacement changes randomly between sites
9. nonperiodic bright clusters or low-resolution contrast with no reproducible motif geometry

A preferred triangle orientation, elongation, unequal brightness, or local asymmetry alone is not chirality.

Classify as Chiral CDW only if multiple repeated CDW-scale motifs show the same clear handedness, such as:
1. consistently skewed triangular motifs with the same rotational sense
2. repeated lopsided arrowhead-like motifs that cannot be made equivalent to their mirror image by rotation
3. three-lobed features with a consistent clockwise or anticlockwise ordering of unequal lobes
4. the same corner-to-corner rotational intensity progression repeated across several sites
5. repeated pinwheel-like or twisted contrast with a common handed direction
6. compact two- or three-lobed motifs in which a weaker lobe, shoulder, or bright maximum is consistently displaced toward the same side across the lattice, producing a repeated mirror-asymmetric internal arrangement; an obvious pinwheel is not required
"""
        },
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]

    system = """ You are the Chiral CDW specialist agent. Do not rely on metadata or other agents.
    Rules:
    1. Allowed labels: Chiral CDW, Not Chiral
    2. Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """

    state["Chiral_topo_analysis"] = call_json_agent(system, prompt)
    return state

def Spectroscopy_chirality_agent(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": """
You are an STM spectroscopy analysis agent specialized in identifying Chiral CDW from dI/dV maps. The input image is a spatial dI/dV map of LDOS. Analyze the electronic modulation pattern, not the topographic height.

Important principle:
In dI/dV maps, Chiral CDW often appears as repeated LDOS motifs with triangular, three-lobed, pinwheel-like, arrowhead-like, or rotationally biased intensity. The chirality may be local, patchy, noisy, or domain-limited.

Classify as Chiral CDW if the LDOS map shows repeated non-random evidence of any of the following:
1. triangular or three-lobed LDOS maxima with a preferred handed orientation
2. pinwheel-like LDOS texture
3. clockwise or counterclockwise intensity arrangement around local maxima
4. repeated local mirror-symmetry breaking
5. unequal LDOS intensity along three CDW-related directions
6. rotationally biased clusters
7. diagonal/triangular ordering with consistent directional asymmetry
8. defect-pinned but spatially recurring handed motifs
9. sparse but repeated chiral LDOS features across multiple regions

Key rule:
If the dI/dV map contains many repeated triangular/three-lobed bright features with non-mirror-equivalent local arrangement, classify as Chiral CDW. Long-range perfect order is not required.

Do not require:
1. perfect periodicity
2. clean atomic resolution
3. globally uniform chirality
4. an obvious spiral
5. high contrast everywhere
6. absence of disorder
7. a full Fourier-space proof

Do not reject Chiral CDW just because:
1. the signal is weak
2. motifs are blurred
3. the image is noisy
4. the order is short-range
5. only local domains show chirality
6. defects or sparse bright spots are present

Classify as not Chiral only if:
1. LDOS modulation is mirror-symmetric or directionally neutral, or
2. features are purely random without repeated oriented motifs, or
3. apparent asymmetry is clearly due to scan artifacts, compression, edge effects, or isolated defects.

Evaluation rule:
Prioritize repeated local handed LDOS motifs over global perfection. If repeated triangular/pinwheel-like LDOS motifs are visible across the image, the correct label is Chiral CDW.
"""         },
            {"type": "text", "text": "Image 2: dI/dV map"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{state['input_didv_base64']}"},
            },
    ]
    system = """
        You are the Chiral CDW specialist agent. Analyze the image as the dI/dV map. Do not rely on metadata or other agents.
        Rules:
        1. Allowed labels: Chiral CDW, Not Chiral
        2. Output only JSON:
        {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """

    state["spectroscopy_analysis"] = call_json_agent(system, prompt)
    return state

def skip_spectroscopy(state: AgentState):
    state["spectroscopy_analysis"] = {
        "reasoning": "No Image 2 (dI/dV map) was provided, so spectroscopy analysis was skipped."
    }
    return state


def spectroscopy_router(state: AgentState) -> str:
    return "run_spectroscopy" if state.get("input_didv_base64") else "skip_spectroscopy"

def _format_retrieval_for_judge(state: AgentState) -> str:
    results = state.get("retrieval_results", []) or []
    if not results:
        return (
            "No retrieved precedents were available. "
            "Do not adjust classification based on missing retrieval."
        )
    lines = []
    for item in results:
        rank = item.get("rank", "?")
        label = item.get("label") or "Unknown"
        identifier = item.get("id", f"retrieved_{rank}")
        score = item.get("score")
        topo_sim = item.get("topograph_similarity")
        didv_sim = item.get("didv_similarity")
        didv_used = item.get("didv_used", False)
        snippet = item.get("snippet") or ""
        retrieval_basis = item.get("retrieval_basis", "Based on topograph only")

        lines.append(f"  Match {rank}")
        lines.append(f"  Dataset ID   : {identifier}")
        lines.append(f"  Label: {label}")
        lines.append(f"  Similarity score: {score}  (topo={topo_sim}" + (f", dI/dV={didv_sim}" if didv_used else "") + ")")
        lines.append(f"  Retrieval basis: {retrieval_basis}")
        if snippet:
            lines.append(f"  Dataset note : {snippet}")
        lines.append("")

    label_counts: Dict[str, int] = {}
    for item in results:
        lbl = item.get("label") or "Unknown"
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    majority_label = max(label_counts, key=label_counts.__getitem__)
    lines.append(f"Majority label of similar retrieved cases: {majority_label} ({label_counts})")

    return "\n".join(lines)

def ensemble_moiré(state: AgentState):
    return run_ensemble(moiré_agent, state, "moiré_analysis", n_runs=5)

def ensemble_visual(state: AgentState):
    return run_ensemble(visual_analyst, state, "visual_analysis", n_runs=5)

def ensemble_fourier(state: AgentState):
    return run_ensemble(fourier_agent, state, "fourier_analysis", n_runs=5)

def ensemble_motif_chirality(state: AgentState):
    return run_ensemble(motif_chirality_agent, state, "Chiral_topo_analysis", n_runs=5)

def ensemble_spectroscopy(state: AgentState):
    return run_ensemble(Spectroscopy_chirality_agent, state, "spectroscopy_analysis", n_runs=5)

def final_judge(state: AgentState):
    retrieval_context = _format_retrieval_for_judge(state)
    full_prompt = f"""
Retriever agent analysis:
{retrieval_context}

Moiré agent analysis:
{json.dumps(state.get("moiré_analysis", {}), indent=2, ensure_ascii=False, default=str)}

Visual agent analysis:
{json.dumps(state.get("visual_analysis", {}), indent=2)}

Fourier agent analysis:
{json.dumps(state.get("fourier_analysis", {}), indent=2)}

Motif chirality agent analysis:
{json.dumps(state.get("Chiral_topo_analysis", {}), indent=2)}

Spectroscopy chirality agent analysis:
{json.dumps(state.get("spectroscopy_analysis", {}), indent=2)}

"""
    state["full_final_prompt"] = full_prompt
    system = """
    You are the final judge. You are given ensemble results (5 runs each) from specialist agents.
    Rules:
    1. Allowed final labels: "CDW", "Chiral CDW", "Non-CDW", "Inconclusive".

    2. Moiré Override Rule (Highest Priority):
    i. If the Moiré agent returns "Moiré" with confidence ≥ 70, the final label must be Non-CDW.
    ii. Moiré patterns are structural interference effects and should never be classified as CDW.
    iii. Only override this strong rule if both Visual and Fourier agents have very high confidence (>85) in CDW and the Moiré confidence is low (<70).

    3. For distinguishing CDW vs Non-CDW (when Moiré agent says "Non-Moiré"):
    i. Use the Visual Agent and Fourier Agent.
    ii. If both agree, accept that label.
    iii. If they disagree, accept the label from the agent with the higher confidence.

    4. Chirality Decision (only when intermediate result is CDW):
    i. Use Motif Chirality Agent (30%) and Spectroscopy Chirality Agent (70%).
    ii. If spectroscopy is unavailable:
        - Use Fourier Agent for chirality only if it explicitly says "Chiral CDW".
        - Otherwise use Motif Chirality Agent.

    5. Retriever Agent Role:
    i. Only used for confidence adjustment.
    ii. If majority label matches your final decision, slightly increase confidence.
    iii. Otherwise ignore it.

    6. Numerical FFT evidence (deterministic fft_analyzer pipeline):
    i. It reports "CDW-compatible FFT evidence", not a definitive CDW classification.
    ii. cdw_evidence_level "insufficient" argues against CDW-family labels unless both Visual agents are highly confident (>85) in CDW.
    iii. cdw_evidence_level "strong" supports (but does not prove) CDW-family labels; lattice or moire periodicities can produce the same signature, so the Moiré Override Rule still applies.
    iv. If its status is "unavailable" or "failed", ignore it.

    Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """
    raw_decision = call_json_agent(system, full_prompt)
    state["final_decision"] = raw_decision
    state["final_label"] = state["final_decision"]["final_label"]
    state["confidence"] = float(state["final_decision"]["confidence"])
    state["explanation"] = state["final_decision"]["explanation"]
    return state
workflow = StateGraph(AgentState)

workflow.add_node("retriever_agent", retriever_agent)
workflow.add_node("ensemble_moiré", ensemble_moiré)
workflow.add_node("ensemble_visual", ensemble_visual)
workflow.add_node("fft_tool", fft_tool)
workflow.add_node("ensemble_fourier", ensemble_fourier)
workflow.add_node("ensemble_motif", ensemble_motif_chirality)
workflow.add_node("ensemble_spectroscopy", ensemble_spectroscopy)
workflow.add_node("skip_spectroscopy", skip_spectroscopy)
workflow.add_node("final_judge", final_judge)

workflow.set_entry_point("retriever_agent")
workflow.add_edge("retriever_agent", "ensemble_moiré")
workflow.add_edge("ensemble_moiré", "ensemble_visual")
workflow.add_edge("ensemble_visual", "fft_tool")   
workflow.add_edge("fft_tool", "ensemble_fourier")  
workflow.add_edge("ensemble_fourier", "ensemble_motif")
workflow.add_conditional_edges(
    "ensemble_motif",
    spectroscopy_router,
    {
        "run_spectroscopy": "ensemble_spectroscopy",
        "skip_spectroscopy": "skip_spectroscopy",
    },
)
workflow.add_edge("ensemble_spectroscopy", "final_judge")
workflow.add_edge("skip_spectroscopy", "final_judge")
workflow.add_edge("final_judge", END)

agentic_graph = workflow.compile()