"""Removal rates by acoustic family, for Figure 3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from brainturtle import config as cfg, featuresets as fs, report

report.run_banner("e_category")

feats = fs.common_features()
dropped = set(fs.drop_set(cfg.PRIMARY_THRESHOLD, feats))
keep = fs.keep_set(cfg.PRIMARY_THRESHOLD, feats)
fs.assert_matches_models(keep)

global_rate = len(dropped) / len(feats)

tbl = pd.DataFrame({"feature": feats})
tbl["category"] = tbl["feature"].apply(fs.categorize)
tbl["dropped"] = tbl["feature"].isin(dropped)

cat = (tbl.groupby("category")
          .agg(total=("feature", "count"), dropped=("dropped", "sum"))
          .reset_index())
cat["rate"] = cat["dropped"] / cat["total"]
cat["enrichment"] = cat["rate"] / global_rate
cat["share_of_removed"] = cat["dropped"] / len(dropped)
cat = cat.sort_values("rate", ascending=False).reset_index(drop=True)

report.header("Figure 3 - removal rate by acoustic family",
              f"canonical rule = top {cfg.PRIMARY_THRESHOLD:.0%} SHAP "
              f"({len(dropped)} of {len(feats)}, measured rate "
              f"{global_rate:.3f})")
report.show(cat, ["category", "total", "dropped", "rate", "enrichment",
                  "share_of_removed"])

legacy = set(fs.legacy_union_drop_set(feats))
leg = pd.DataFrame({"feature": feats})
leg["category"] = leg["feature"].apply(fs.categorize)
leg["dropped"] = leg["feature"].isin(legacy)
legcat = (leg.groupby("category")
             .agg(total=("feature", "count"), dropped=("dropped", "sum"))
             .reset_index())
legcat["rate_legacy"] = legcat["dropped"] / legcat["total"]

cmp = cat.merge(legcat[["category", "rate_legacy"]], on="category")
cmp["crosses_reference_line"] = (
    (cmp["rate"] >= cfg.PRIMARY_THRESHOLD)
    != (cmp["rate_legacy"] >= cfg.PRIMARY_THRESHOLD))

print()
print(f"  superseded union rule removed {len(legacy)} features "
      f"(rate {len(legacy)/len(feats):.3f})")
report.show(cmp, ["category", "rate", "rate_legacy",
                  "crosses_reference_line"])
flipped = cmp[cmp["crosses_reference_line"]]["category"].tolist()
print(f"  families whose bar crosses the {cfg.PRIMARY_THRESHOLD:.0%} line "
      f"between rules: {flipped if flipped else 'none'}")

mf = tbl[tbl["category"] == "MFCC"].copy()
mf["subfamily"] = mf["feature"].apply(fs.mfcc_subfamily)
sub = (mf.groupby("subfamily")
         .agg(total=("feature", "count"), dropped=("dropped", "sum"))
         .reset_index())
sub["rate"] = sub["dropped"] / sub["total"]
print()
print("  MFCC breakdown (shape = skew/kurt):")
report.show(sub, ["subfamily", "total", "dropped", "rate"])

report.save(cat, "fig3_category_rates")
report.save(cmp, "fig3_rule_comparison")
report.save(sub, "fig3_mfcc_subfamily")
report.save(pd.DataFrame({"feature": sorted(dropped)}), "drop30_canonical")
report.save(pd.DataFrame({"feature": keep}), "keep30_canonical")
