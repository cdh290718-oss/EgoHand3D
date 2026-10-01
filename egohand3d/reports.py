"""Static reports, plots, and portable result archives; no web application."""
from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .storage import atomic_text, file_hash, load_json, save_json


def write_report(directory, title, paragraphs, rows, columns):
    markdown = [f"# {title}", "", *paragraphs, "", "|" + "|".join(columns) + "|",
                "|" + "|".join(["---"] * len(columns)) + "|"]
    markdown += ["|" + "|".join(str(v) for v in row) + "|" for row in rows]
    atomic_text(directory / "report.md", "\n".join(markdown) + "\n")
    table = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr>"
    table += "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
    images = "".join(f'<img src="{html.escape(p.name)}" alt="{html.escape(p.stem)}">'
                     for p in sorted(directory.glob("chart_*.png")))
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{max-width:1050px;margin:40px auto;padding:0 20px;font:16px/1.6 sans-serif;color:#223}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccd;padding:8px;text-align:left}}img{{max-width:100%}}</style>
<h1>{html.escape(title)}</h1>{''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)}<table>{table}</table>{images}</html>"""
    atomic_text(directory / "report.html", document)


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def build_run_report(directory: Path):
    summary = load_json(directory / "summary.json")
    flags = summary["quality_flags"]
    if flags:
        plt = pyplot()
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        ax.barh(list(flags), list(flags.values()), color="#437a9e")
        ax.set(xlabel="Hand observations flagged", title="Quality diagnostics (not accuracy)")
        fig.savefig(directory / "chart_quality.png", dpi=160)
        plt.close(fig)
    rows = [("Run state", summary["state"]), ("Processed frames", summary["frames_completed"]),
            ("Frames containing predictions", summary["frames_with_hands"]), ("Hands", summary["hands"]),
            ("MANO consistency failures", summary["roundtrip_failures"]),
            ("Failed tasks", len(summary["failures"])),
            ("Mean frame processing seconds", summary["mean_processing_seconds"])]
    write_report(directory, "EgoHand3D processing report", [
        "Generated from actual processing records. Detection coverage is not recall without ground truth.",
        "MANO consistency checks serialization/reconstruction, not pose estimation accuracy.",
        "run_config.json records asset/input identities, code hashes and options. index.csv maps artifacts to inputs.",
        "Images and meshes are stored in overlays/ and meshes/."], rows, ["Item", "Value"])


def build_sequence_report(directory: Path, traces: list[dict], summary: dict):
    if traces:
        plt = pyplot()
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
        chosen = (traces[0]["sequence_id"], traces[0]["track_id"])
        subset = [t for t in traces if (t["sequence_id"], t["track_id"]) == chosen]
        for variant in ["raw", "processed"]:
            wrist = np.array([t[f"{variant}_joints"][0] for t in subset])
            times = np.array([t["timestamp_s"] for t in subset])
            axes[0].plot(times, wrist[:, 0], label=variant)
            axes[1].plot(times, wrist[:, 2], label=variant)
        axes[0].set_ylabel("Wrist x (m)")
        axes[1].set_ylabel("Wrist z (m)")
        axes[1].set_xlabel("Time (s)")
        for ax in axes:
            ax.legend()
        axes[0].set_title(f"Sequence {chosen[0]}, track {chosen[1]} — observed frames")
        fig.savefig(directory / "chart_trajectory.png", dpi=160)
        plt.close(fig)
    write_report(directory, "EgoHand3D sequence report", [
        "Raw predictions are retained beside processed MANO and geometry.",
        "Interpolated hands are explicitly labelled and excluded from detection and accuracy evaluation.",
        "Lower motion speed does not imply better accuracy; evaluate against ground truth."],
        [(key, summary[key]) for key in ["frames", "sequences", "tracks", "detected_hands", "interpolated_hands",
                                       "smoothing_seconds", "mean_joint_speed_m_s"]], ["Item", "Value"])


def build_benchmark_report(directory: Path, result: dict):
    save_json(directory / "summary.json", result)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["sample_id", "gt_hand_index", "prediction_index", "mean_2d_px"])
    writer.writeheader()
    writer.writerows(result["details"])
    atomic_text(directory / "details.csv", buffer.getvalue())
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for denominator in ["matched_only", "all_gt_joints"]:
        pairs = [(float(k), result["pck_px"][k][denominator]) for k in sorted(result["pck_px"], key=float)
                 if result["pck_px"][k][denominator] is not None]
        if pairs:
            ax.plot(*zip(*pairs), marker="o", label=denominator)
    ax.set(xlabel="Threshold (px)", ylabel="PCK", ylim=(0, 1.02), title="2D PCK with explicit denominators")
    if ax.lines:
        ax.legend()
    fig.savefig(directory / "chart_pck.png", dpi=160)
    plt.close(fig)
    rows = [(key, result[key]) for key in ["frames", "gt_hands", "matched_hands", "missed_hands",
                                         "unmatched_predictions", "hand_recall", "hand_precision"]]
    rows += [(key, value["mean"]) for key, value in result["metrics"].items()]
    write_report(directory, "EgoHand3D benchmark report", [
        "Matching uses same-side bounding-box IoU, not the smallest ground-truth joint error.",
        "Joint errors use matched hands. All-GT PCK counts missing hands as failures.",
        f"Ground-truth manifest SHA-256: {result['ground_truth_sha256']}"], rows, ["Metric", "Value"])


def compare_benchmarks(paths: list[Path], out: Path):
    results = [load_json(p) for p in paths]
    if len(results) < 2:
        raise ValueError("At least two benchmark summaries are required")
    reference = results[0]
    for result in results:
        if result.get("schema") != "egohand3d.benchmark/1":
            raise ValueError("Comparison requires benchmark summaries")
        for key in ["ground_truth_sha256", "protocol", "evaluator_sha256"]:
            if result[key] != reference[key]:
                raise ValueError(f"Cannot compare different {key}")
        if list(result["pck_px"]) != list(reference["pck_px"]):
            raise ValueError("PCK thresholds differ")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Comparison output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    rows = [(str(path), result["variant"], result["hand_recall"], result["metrics"]["2d_px"]["mean"],
             result["metrics"]["root_mpjpe_mm"]["mean"], result["metrics"]["pa_mpjpe_mm"]["mean"])
            for path, result in zip(paths, results)]
    columns = ["Run", "Variant", "Recall", "2D px", "Root MPJPE mm", "PA-MPJPE mm"]
    write_report(out, "EgoHand3D benchmark comparison", ["Identical GT manifest, matching rules and evaluator verified."], rows, columns)
    save_json(out / "comparison.json", {"inputs": [str(p) for p in paths], "columns": columns, "rows": rows,
                                      "ground_truth_sha256": reference["ground_truth_sha256"]})
    return {"runs": len(rows), "report": str(out / "report.html")}


def package_results(directory: Path, destination: Path) -> dict:
    if not (directory / "summary.json").is_file():
        raise ValueError("Result directory needs summary.json")
    if destination.exists():
        raise FileExistsError(f"Archive already exists: {destination}")
    allowed = {".json", ".jsonl", ".csv", ".md", ".html", ".png", ".jpg", ".obj", ".mp4"}
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refusing to package symlink: {path}")
        if path.is_file() and path.suffix in allowed:
            files.append(path)
    manifest = {str(p.relative_to(directory)): {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in files}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, str(path.relative_to(directory)))
            archive.writestr("ARTIFACT_MANIFEST.json", json.dumps(manifest, indent=2))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"archive": str(destination), "files": len(files), "sha256": file_hash(destination)}
