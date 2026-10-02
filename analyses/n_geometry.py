"""Acoustic feature space geometry, for the supplementary panel and Discussion."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from brainturtle import config as cfg, data, featuresets as fs, report

warnings.filterwarnings("ignore", category=UserWarning)

report.run_banner("n_geometry")

K_RANGE = list(range(2, 7))
PCA_VARIANCE = 0.95
UMAP_SEED = 42

pitt = data.pitt()
feats = fs.common_features()
keep = fs.keep_set(cfg.PRIMARY_THRESHOLD, feats)
drop = fs.drop_set(cfg.PRIMARY_THRESHOLD, feats)

fs.assert_matches_models(keep, cfg.PRIMARY_THRESHOLD)

SPACES = {
    f"full ({len(feats)} features)": sorted(feats),
    f"deconfounded ({len(keep)} features)": sorted(keep),
}

labels = pitt["label"].values
sil_rows, sum_rows, emb_rows = [], [], []

for space_name, cols in SPACES.items():
    X = StandardScaler().fit_transform(pitt[cols].values)

    pca = PCA(n_components=PCA_VARIANCE, random_state=UMAP_SEED).fit(X)
    Z = pca.transform(X)
    print(f"\n  {space_name}: original dim {X.shape[1]} -> "
          f"PCA dim {Z.shape[1]} ({PCA_VARIANCE:.0%} variance)")

    sils = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=UMAP_SEED).fit(Z)
        s = float(silhouette_score(Z, km.labels_))
        sils[k] = s
        sil_rows.append({"space": space_name, "k": k, "silhouette": s})
        print(f"    K={k} silhouette={s:.3f}")

    mean_sil = float(np.mean(list(sils.values())))

    try:
        import hdbscan
        hdb = hdbscan.HDBSCAN(min_cluster_size=10).fit(Z)
        hlab = hdb.labels_
        n_clusters = int(len(set(hlab)) - (1 if -1 in hlab else 0))
        n_noise = int((hlab == -1).sum())
        sil_hdb = (float(silhouette_score(Z, hlab))
                   if n_clusters >= 2 else float("nan"))
    except ImportError:
        n_clusters, n_noise, sil_hdb = -1, -1, float("nan")
        print("    hdbscan not installed -- skipped")

    print(f"    HDBSCAN: {n_clusters} cluster(s), {n_noise}/{len(Z)} noise, "
          f"silhouette={sil_hdb:.3f}")

    try:
        import umap
        emb = umap.UMAP(n_components=2, random_state=UMAP_SEED,
                        n_neighbors=15, min_dist=0.1).fit_transform(Z)
    except ImportError:
        emb = np.full((len(Z), 2), np.nan)
        print("    umap not installed -- UMAP panel unavailable")

    for i, sid in enumerate(pitt["speaker_id"].values):
        emb_rows.append({
            "space": space_name, "speaker_id": sid, "label": int(labels[i]),
            "pc1": float(Z[i, 0]), "pc2": float(Z[i, 1]),
            "umap1": float(emb[i, 0]), "umap2": float(emb[i, 1]),
        })

    sum_rows.append({
        "space": space_name,
        "n_features": len(cols),
        "n_speakers": len(Z),
        "pca_dim": int(Z.shape[1]),
        "silhouette_mean_k2_k6": mean_sil,
        "silhouette_k2": sils[2],
        "hdbscan_clusters": n_clusters,
        "hdbscan_noise": n_noise,
        "hdbscan_silhouette": sil_hdb,
    })

sil = pd.DataFrame(sil_rows)
summary = pd.DataFrame(sum_rows)
embedding = pd.DataFrame(emb_rows)

report.header("Acoustic feature space geometry (Supplementary S3)",
              f"K-means K={K_RANGE[0]}..{K_RANGE[-1]} on PCA "
              f"({PCA_VARIANCE:.0%} variance), {len(pitt)} speaker rows")
report.show(summary, ["space", "n_features", "pca_dim",
                      "silhouette_mean_k2_k6", "silhouette_k2",
                      "hdbscan_clusters", "hdbscan_noise"])

print()
print("  silhouette by K:")
report.show(sil.pivot(index="k", columns="space",
                      values="silhouette").reset_index(), fmt="{:.3f}")

full = summary.iloc[0]
deco = summary.iloc[1]
print()
print(f"  The manuscript's 0.17 corresponds to the FULL space "
      f"({full['silhouette_mean_k2_k6']:.4f}).")
print(f"  The deconfounded space gives "
      f"{deco['silhouette_mean_k2_k6']:.4f} "
      f"(K=2: {deco['silhouette_k2']:.3f}).")
if deco["hdbscan_clusters"] <= 0:
    print("  HDBSCAN finds no valid clusters in the deconfounded space "
          "either -- every speaker is noise.")
print("  Either restate S3 as a property of the full space, or cite the "
      "deconfounded row above.")

report.save(sil, "geometry_silhouettes")
report.save(summary, "geometry_summary")
report.save(embedding, "geometry_embedding")
