"""Build the figure package for submission."""
import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HERE = Path(__file__).resolve().parent
FORMATS = ("tiff", "png", "jpg")

MAIN = {
    "Figure_1": "figure1_age_benchmark",
    "Figure_2": "figure2_removal",
    "Figure_3": "figure3_category",
    "Figure_4": "figure4_incremental",
    "Figure_5": "figure4_external",
    "Figure_6": "figure6_pipeline",
}
SUPP = {
    "Supplementary_Figure_S1": "supplementary_fig8_pitt_demographics",
    "Supplementary_Figure_S2": "supplementary_fig13_age_ranking_stability",
    "Supplementary_Figure_S3": "figure1_suppression",
    "Supplementary_Figure_S4": "supplementary_fig14_detector_robustness",
    "Supplementary_Figure_S5": "supplementary_fig1_removal_ladder",
    "Supplementary_Figure_S6": "supplementary_fig9_gender_control",
    "Supplementary_Figure_S7": "supplementary_fig7_spearman_heatmap",
    "Supplementary_Figure_S8": "supplementary_fig2_shap_spearman_overlap",
    "Supplementary_Figure_S9": "supplementary_fig3_geometry",
    "Supplementary_Figure_S10": "supplementary_fig5_calibration",
    "Supplementary_Figure_S11": "supplementary_fig11_recording_duration",
    "Supplementary_Figure_S12": "supplementary_fig12_speech_ratio",
}


def check(mapping):
    missing = []
    for name, stem in mapping.items():
        for ext in FORMATS:
            p = HERE / f"{stem}.{ext}"
            if not p.exists():
                missing.append(f"{name}: {p.name}")
    return missing


def manifest():
    out = ["figures.zip -- figure package for the npj Digital Medicine submission",
           "",
           "Packaged name <- source file in figures/",
           "",
           "MAIN FIGURES"]
    for name, stem in MAIN.items():
        out.append(f"  {name}.<tiff|png|jpg>  <-  {stem}.<tiff|png|jpg>")
    out += ["", "SUPPLEMENTARY FIGURES"]
    for name, stem in SUPP.items():
        out.append(f"  {name}.<tiff|png|jpg>  <-  {stem}.<tiff|png|jpg>")
    out += ["",
            "Figure 6 is a single schematic. The values that were previously a",
            "supplementary figure of the Kang recording conditions are reported",
            "as Supplementary Table 2.",
            "",
            "Source filenames predate the current supplementary numbering and do",
            "not match it; the mapping above is authoritative.",
            ""]
    return "\n".join(out)


def main(dests):
    missing = check(MAIN) + check(SUPP)
    if missing:
        sys.exit("missing renders:\n  " + "\n  ".join(missing))

    tmp = HERE.parent / "figures.zip"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("MANIFEST.txt", manifest())
        for folder, mapping in (("main", MAIN), ("supplementary", SUPP)):
            for name, stem in mapping.items():
                for ext in FORMATS:
                    z.write(HERE / f"{stem}.{ext}", f"{folder}/{name}.{ext}")

    data = tmp.read_bytes()
    n = len(zipfile.ZipFile(tmp).namelist())
    for d in dests:
        d = Path(d)
        if d.resolve() != tmp.resolve():
            d.write_bytes(data)
        print("  %-44s %d entries, %.1f MB" % (d, n, len(data) / 1e6))
    print("  md5 %s" % hashlib.md5(data).hexdigest())


if __name__ == "__main__":
    targets = sys.argv[1:] or [HERE.parent / "figures.zip"]
    main(targets)
