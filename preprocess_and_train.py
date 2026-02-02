"""
Preprocess interpretation text into panel-specific comment/impression strings,
embed unique strings, join back to rows, and train a model to predict embeddings
from lab values (+ optional category/type features).

Requires:
  pip install pandas numpy scikit-learn sentence-transformers h5py openpyxl

Notes:
- parse Interpretation -> outputs -> unique texts -> embeddings -> map back
- Supports either:
    (A) embedding computed on the fly with SentenceTransformers, OR
    (B) loading precomputed embeddings from an H5 file that corresponds to the unique-text list order (i.e. from your embedding model of choice)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import re
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity

# Optional deps
try:
    import h5py
except Exception:
    h5py = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# -----------------------------
# Text parsing utilities
# -----------------------------

_BAD_ROW_DROP_1INDEXED = 278 
_DROP_INTERPRETATION_CONTAINS = "Although the normal dilute Russell"


def _safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)


def extract_comments_block(interpretation: str) -> str:
    s = _safe_str(interpretation).replace("\r\n", "")
    parts = s.split("    COMMENTS:")
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def split_into_paragraphs(comment_block: str) -> list[str]:
    s = _safe_str(comment_block)

    # Two newlines (common)
    p1 = [p for p in re.split(r"\n\s*\n", s) if p.strip()]
    # Two carriage returns (legacy)
    p2 = [p for p in re.split(r"\r\s*\r", s) if p.strip()]

    if len(p1) == 1 and len(p2) > 1:
        return [p.strip() for p in p2]
    if len(p2) == 1 and len(p1) > 1:
        return [p.strip() for p in p1]
    # If both are 1 or both >1, prefer p1
    return [p.strip() for p in p1]


def normalize_paragraph(p: str) -> str:
    p = _safe_str(p).strip()
    p = p.replace("\n", " ").replace("\r", " ")
    p = re.sub(r"\s+", " ", p).strip()
    return p


def drop_note_paragraphs(paragraphs: Iterable[str]) -> list[str]:
    out = []
    for p in paragraphs:
        if "NOTE:" in p:
            continue
        out.append(p)
    return out


# -----------------------------
# Panel specification
# -----------------------------

@dataclass(frozen=True)
class PanelSpec:
    name: str
    comment_types: tuple[str, ...]     # positive filters (any substring match)
    negative_filters: tuple[str, ...] = ()
    lab_tests: tuple[str, ...] = ()    # columns to pull as numeric features
    include_type_feature: bool = True  # whether to include panel name as a categorical feature


# Examples based on your MATLAB scripts.
DRVVT_SPEC = PanelSpec(
    name="DRVVT",
    comment_types=("DRVVT",),
    negative_filters=("prothrombin time (PT)",),
    lab_tests=("RVR1", "RVMR2", "RVCR3"),
)

APTT_SPEC = PanelSpec(
    name="APTT",
    comment_types=("APTT", "(RT)", "(TT)"),
    lab_tests=("APTSC", "APMSC", "PNPSA", "PNPPL", "STDEL", "TTSC", "RTSC"),
)

PT_SPEC = PanelSpec(
    name="PT",
    comment_types=("(PT)", "1:1 mixing", "factors II, V, VII and X"),
    lab_tests=("PTMSC", "PTSEC", "F_2", "FACTV", "F_7", "F_10"),
)

FIB_SPEC = PanelSpec(
    name="Fibrinogen",
    comment_types=("thrombin time", "reptilase time"),
    lab_tests=("TTSC", "RTSC", "CLFIB", "PTFIB", "DIMER", "SOLFM"),
)

AB_SPEC = PanelSpec(
    name="Antibody",
    comment_types=("glycoprotein", "antiphospholipid"),
    lab_tests=("GB2GP", "GCLIP", "MB2GP", "MCLIP"),
)

# Add more specs as needed, including “Impressions” parsing if you want a separate rule set.


# -----------------------------
# Core preprocessing
# -----------------------------

_number_pattern = re.compile(r"-?\d+(\.\d+)?")


def parse_numeric_cell(x) -> float:
    """
    extract first float-like number from a string cell.
    """
    s = _safe_str(x)
    m = _number_pattern.search(s)
    if not m:
        return np.nan
    return float(m.group(0))


def build_outputs_for_panel(
    df: pd.DataFrame,
    spec: PanelSpec,
    comment_fixes: Optional[dict[str, str]] = None,
    split_combined_comment: Optional[tuple[str, str]] = None,
    comments_to_split: Optional[set[str]] = None,
) -> pd.Series:
    """
    Produces one extracted text per row

    Parameters
    ----------
    comment_fixes:
        Exact-string replacements (e.g., the typo fix in drvvt script).
    split_combined_comment:
        Two strings to insert when a combined comment is detected.
    comments_to_split:
        Exact strings that indicate "this paragraph should be replaced by split_combined_comment".

    Returns
    -------
    pd.Series of extracted text.
    """
    comment_fixes = comment_fixes or {}
    comments_to_split = comments_to_split or set()

    outputs: list[str] = []

    for _, row in df.iterrows():
        interp = _safe_str(row.get("Interpretation", ""))
        block = extract_comments_block(interp)
        if not block:
            outputs.append("")
            continue

        paragraphs = split_into_paragraphs(block)
        paragraphs = drop_note_paragraphs(paragraphs)

        normalized: list[str] = []
        i = 0
        while i < len(paragraphs):
            p = normalize_paragraph(paragraphs[i])

            # exact-string fixups (MATLAB: if strcmp(output{j}, c(3)) then replace)
            if p in comment_fixes:
                p = comment_fixes[p]

            # split combined APTT+DRVVT comments into two
            if p in comments_to_split and split_combined_comment is not None:
                normalized.append(normalize_paragraph(split_combined_comment[0]))
                normalized.append(normalize_paragraph(split_combined_comment[1]))
            else:
                normalized.append(p)
            i += 1

        # positive filter + negative filter
        keep = []
        for p in normalized:
            if any(ct in p for ct in spec.comment_types):
                if any(nf in p for nf in spec.negative_filters):
                    continue
                keep.append(p)

        outputs.append(" ".join(keep).strip())

    return pd.Series(outputs, index=df.index, name="text")


def load_and_clean_export(excel_path: str, sheet: int | str = 1) -> pd.DataFrame:
    """
    shared cleanup
      - read Excel sheet
      - drop row 278 (1-indexed)
      - remove rows containing the 'Although the normal dilute Russell' phrase
    """
    df = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")

    # Drop row 278 in MATLAB (1-indexed) -> index 277 in Python (0-indexed) IF it exists.
    idx0 = _BAD_ROW_DROP_1INDEXED - 1
    if 0 <= idx0 < len(df):
        df = df.drop(df.index[idx0])

    # Remove the special cases
    if "Interpretation" in df.columns:
        mask = df["Interpretation"].astype(str).str.contains(_DROP_INTERPRETATION_CONTAINS, na=False)
        df = df.loc[~mask].copy()

    df = df.reset_index(drop=True)
    return df


def attach_lab_features(df: pd.DataFrame, spec: PanelSpec) -> pd.DataFrame:
    """
    Creates numeric feature columns for the spec.lab_tests.
    Uses first-number parsing to mimic regexp/str2num behavior.
    Missing numeric values remain NaN; you can impute later in the modeling pipeline.
    """
    out = df.copy()
    for col in spec.lab_tests:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = out[col].apply(parse_numeric_cell)
    return out


def embed_unique_texts(
    unique_texts: list[str],
    method: Literal["sbert", "h5"] = "sbert",
    *,
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    h5_path: Optional[str] = None,
    h5_dataset: str = "/embeddings",
) -> np.ndarray:
    """
    Produces an embedding matrix aligned to unique_texts.

    method="sbert": compute embeddings locally (recommended if you don't need exact text-gecko alignment)
    method="h5": load embeddings from an H5 file (must align with unique_texts order you used externally)
    """
    if method == "h5":
        if h5py is None:
            raise ImportError("h5py is required for method='h5'. pip install h5py")
        if not h5_path:
            raise ValueError("h5_path must be provided for method='h5'")
        with h5py.File(h5_path, "r") as f:
            X = np.array(f[h5_dataset])
        if X.shape[0] != len(unique_texts):
            raise ValueError(
                f"H5 embeddings rows ({X.shape[0]}) != number of unique_texts ({len(unique_texts)}). "
                "These must align 1:1 in the same order."
            )
        return X

    # SBERT embedding
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required for method='sbert'. "
            "pip install sentence-transformers"
        )
    model = SentenceTransformer(sbert_model_name)
    # Avoid embedding empty strings as meaningful targets
    texts = [t if t.strip() else "[EMPTY]" for t in unique_texts]
    X = model.encode(texts, normalize_embeddings=False, show_progress_bar=True)
    return np.asarray(X, dtype=np.float32)


def build_feature_table_for_panel(
    df: pd.DataFrame,
    spec: PanelSpec,
    *,
    id_col: str = "DE_IDNUMBER",
    embedding_method: Literal["sbert", "h5"] = "sbert",
    h5_path: Optional[str] = None,
    h5_dataset: str = "/embeddings",
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    comment_fixes: Optional[dict[str, str]] = None,
    split_combined_comment: Optional[tuple[str, str]] = None,
    comments_to_split: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    End-to-end panel table:
      acc/id, lab numeric features, extracted text, and embedding vector.
    """
    if id_col not in df.columns:
        raise ValueError(f"Missing id_col='{id_col}' in dataframe columns")

    # 1) Extract text outputs
    outputs = build_outputs_for_panel(
        df,
        spec,
        comment_fixes=comment_fixes,
        split_combined_comment=split_combined_comment,
        comments_to_split=comments_to_split,
    )

    # 2) Attach labs
    feat_df = attach_lab_features(df, spec)
    feat_df = feat_df.assign(panel=spec.name, text=outputs)

    # Optionally drop empty outputs like the MATLAB scripts often did later
    feat_df = feat_df.loc[feat_df["text"].astype(str).str.len() > 0].copy()

    # 3) Unique texts and embeddings
    unique_texts = feat_df["text"].drop_duplicates().tolist()
    emb_unique = embed_unique_texts(
        unique_texts,
        method=embedding_method,
        h5_path=h5_path,
        h5_dataset=h5_dataset,
        sbert_model_name=sbert_model_name,
    )

    # 4) Map embeddings back to rows by text
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    idxs = feat_df["text"].map(text_to_idx).to_numpy()
    emb_rows = emb_unique[idxs]

    # Store embeddings as a vector column (like MATLAB feats.Y_feats cell)
    feat_df["embedding"] = [emb_rows[i, :] for i in range(emb_rows.shape[0])]

    # Minimal final table
    cols = [id_col, "panel", "text", *spec.lab_tests, "embedding"]
    # Some tests may be missing in df; ensure existence
    cols = [c for c in cols if c in feat_df.columns]
    return feat_df[cols].reset_index(drop=True)


# -----------------------------
# Modeling: labs -> embedding
# -----------------------------

def train_embedding_regressor(
    table: pd.DataFrame,
    *,
    numeric_feature_cols: list[str],
    embedding_col: str = "embedding",
    categorical_cols: Optional[list[str]] = None,
    model_type: Literal["ridge", "rf"] = "ridge",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Trains a model to predict embedding vectors from lab values (+ optional categories).
    Returns fitted pipeline and a small evaluation dict.
    """
    categorical_cols = categorical_cols or []

    # X, Y
    X = table[numeric_feature_cols + categorical_cols].copy()
    Y = np.vstack(table[embedding_col].to_numpy())  # (n_samples, embedding_dim)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # Preprocess: scale numeric; one-hot categorical
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_feature_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )

    if model_type == "ridge":
        base = Ridge(alpha=1.0, random_state=random_state)
        reg = MultiOutputRegressor(base)
    else:
        base = RandomForestRegressor(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
            max_depth=None,
        )
        reg = MultiOutputRegressor(base)

    pipe = Pipeline([("pre", pre), ("reg", reg)])
    pipe.fit(X_train, Y_train)

    # Evaluate: cosine similarity between predicted and true embedding vectors
    Y_pred = pipe.predict(X_test)
    cos = np.diag(cosine_similarity(Y_pred, Y_test))
    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cosine_mean": float(np.mean(cos)),
        "cosine_median": float(np.median(cos)),
        "mse_mean": float(np.mean((Y_pred - Y_test) ** 2)),
    }
    return pipe, metrics


# -----------------------------
# Example usage
# -----------------------------

def main():
    excel_path = "sample_data.xlsx"  # <-- set this
    df = load_and_clean_export(excel_path, sheet=1)

    # Here we keep it simple and rely on positive-filtering only.
    drvvt_table = build_feature_table_for_panel(
        df,
        DRVVT_SPEC,
        embedding_method="sbert",  # or "h5"
        # h5_path="comments_and_embeddings/DRVVT.h5",
        # h5_dataset="/embeddings",
    )

    # Train model to predict embeddings from labs (+ include panel/category if desired)
    numeric_cols = [c for c in DRVVT_SPEC.lab_tests if c in drvvt_table.columns]
    categorical_cols = ["panel"]  # acts like feature (one-hot)

    ridge_model, ridge_metrics = train_embedding_regressor(
        drvvt_table,
        numeric_feature_cols=numeric_cols,
        categorical_cols=categorical_cols,
        model_type="ridge",
    )
    print("Ridge metrics:", ridge_metrics)

    rf_model, rf_metrics = train_embedding_regressor(
        drvvt_table,
        numeric_feature_cols=numeric_cols,
        categorical_cols=categorical_cols,
        model_type="rf",
    )
    print("RF metrics:", rf_metrics)

    # Save table if you want
    drvvt_table.to_parquet("DRVVT_feature_table.parquet", index=False)


if __name__ == "__main__":
    main()
