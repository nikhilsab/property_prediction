import os
import sys
import csv
import ast
import pandas as pd
from tqdm import tqdm

# If you still read huge structure strings, keep this:
csv.field_size_limit(sys.maxsize)

from mp_api.client import MPRester
from jarvis.db.figshare import data as jarvis_data


# -----------------------------
# Materials Project global state
# -----------------------------
def fetch_mp_state(material_ids, api_key=None):
    """
    Fetch dataset-conditional global state for MP IDs.
    Returns a DataFrame keyed by 'material_id' with mp_* columns.

    Notes:
      - Uses summary.origins -> task_id when possible.
      - Falls back to tasks search if origins isn't available.
      - Pulls run_type from thermo docs (functional family).
      - Pulls task_type + some INCAR/KPOINTS/POTCAR specs from tasks docs.
    """
    material_ids = sorted({mid for mid in material_ids if isinstance(mid, str) and mid.strip()})
    if not material_ids:
        return pd.DataFrame(columns=["material_id"])

    rows = []

    with MPRester(api_key) as mpr:
        # Summary docs (provenance)
        summ = mpr.materials.summary.search(
            material_ids=material_ids,
            fields=["material_id", "origins"]
        )

        # map material_id -> representative task_id (best effort)
        mat_to_task = {}
        for doc in summ:
            mid = str(doc.material_id)
            tid = None
            origins = getattr(doc, "origins", None) or []
            for o in origins:
                tid = getattr(o, "task_id", None) or (o.get("task_id") if isinstance(o, dict) else None)
                if tid:
                    break
            mat_to_task[mid] = str(tid) if tid else None

        # If some materials have no task_id from origins, try tasks search directly
        missing = [mid for mid, tid in mat_to_task.items() if not tid]
        if missing:
            # Grab *some* task for each missing material (best effort)
            # (tasks endpoint supports searching by material_ids in newer MP-API builds; if not, fallback below)
            try:
                tdocs = mpr.materials.tasks.search(
                    material_ids=missing,
                    fields=["material_id", "task_id"],
                    num_chunks=1,
                )
                for t in tdocs:
                    mid = str(getattr(t, "material_id", ""))
                    tid = str(getattr(t, "task_id", ""))
                    if mid and tid and not mat_to_task.get(mid):
                        mat_to_task[mid] = tid
            except Exception:
                # If tasks.search doesn't accept material_ids in your mp_api version, we keep None.
                pass

        # Collect task_ids we have
        task_ids = sorted({tid for tid in mat_to_task.values() if tid})

        # Thermo docs for run_type (functional family)
        task_to_run_type = {}
        if task_ids:
            try:
                thermo = mpr.materials.thermo.search(task_ids=task_ids, fields=["task_id", "run_type"])
                task_to_run_type = {str(t.task_id): str(t.run_type) for t in thermo}
            except Exception:
                task_to_run_type = {}

        # Task docs for input settings
        task_docs = {}
        if task_ids:
            try:
                tdocs = mpr.materials.tasks.search(
                    task_ids=task_ids,
                    fields=[
                        "task_id",
                        "task_type",
                        "input.incar",
                        "input.parameters",
                        "input.kpoints",
                        "input.potcar_spec",
                    ],
                )
                task_docs = {str(t.task_id): t for t in tdocs}
            except Exception:
                task_docs = {}

    # Build rows
    for mid in material_ids:
        tid = mat_to_task.get(mid)
        row = {"material_id": mid, "mp_task_id": tid}

        if tid and tid in task_docs:
            tdoc = task_docs[tid]
            row["mp_run_type"] = task_to_run_type.get(tid)
            row["mp_task_type"] = getattr(tdoc, "task_type", None)

            # Extract selected INCAR-like settings
            incar = {}
            try:
                incar = (tdoc.input.incar or {}) if getattr(tdoc, "input", None) else {}
            except Exception:
                incar = {}

            params = {}
            try:
                params = (tdoc.input.parameters or {}) if getattr(tdoc, "input", None) else {}
            except Exception:
                params = {}

            # Useful state (add/remove as you like)
            for k in ["ENCUT", "EDIFF", "ISMEAR", "SIGMA", "ISPIN", "LDAU", "LDAUU", "LDAUJ", "LMAXMIX"]:
                row[f"mp_incar_{k.lower()}"] = incar.get(k)

            for k in ["NELECT", "NBANDS"]:
                row[f"mp_param_{k.lower()}"] = params.get(k)

            # Compact kpoints/pseudo spec (avoid giant nested objects)
            try:
                kp = getattr(tdoc.input, "kpoints", None)
                row["mp_kpoints"] = str(kp)[:300] if kp is not None else None
            except Exception:
                row["mp_kpoints"] = None

            try:
                ps = getattr(tdoc.input, "potcar_spec", None)
                row["mp_potcar_spec"] = str(ps)[:300] if ps is not None else None
            except Exception:
                row["mp_potcar_spec"] = None

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# JARVIS-DFT global state
# -----------------------------
def fetch_jarvis_state(jarvis_ids, db_name="dft_3d"):
    """
    Fetch dataset-conditional global state for JARVIS IDs from the JARVIS database JSON.

    Returns DataFrame keyed by 'jarvis_id' with jarvis_* columns.

    Note:
      - Exact keys can vary across JARVIS releases.
      - We defensively pull a set of common fields if present.
    """
    jarvis_ids = sorted({jid for jid in jarvis_ids if isinstance(jid, str) and jid.strip()})
    if not jarvis_ids:
        return pd.DataFrame(columns=["jarvis_id"])

    recs = jarvis_data(db_name)  # downloads/caches if not present
    by_jid = {r.get("jid"): r for r in recs if isinstance(r, dict) and r.get("jid")}

    rows = []
    common_keys = [
        "jid",
        "functional",
        "method",
        "calculator",
        "encut",
        "spin_polarized",
        "soc",
        "kpoint_length_unit",
        "dft_type",
        "version",
        "reference",
        "source",
    ]

    for jid in jarvis_ids:
        r = by_jid.get(jid, {})
        row = {"jarvis_id": jid}

        for k in common_keys:
            if k in r:
                row[f"jarvis_{k}"] = r.get(k)

        # If there’s any nested metadata/info, keep a compact blob
        for k in ["metadata", "info"]:
            if k in r:
                row[f"jarvis_{k}"] = str(r.get(k))[:500]

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Unified join helper
# -----------------------------
def add_dataset_state(
    df,
    material_id_col="material_id",
    jarvis_id_col="jarvis_id",
    mp_api_key=None,
    jarvis_db="dft_3d",
):
    """
    Add dataset-conditional global state columns for rows that have either:
      - material_id (MP)
      - jarvis_id (JARVIS)

    Returns (df_with_state, state_only_df).
    """
    out = df.copy()

    mp_ids = []
    if material_id_col in out.columns:
        mp_ids = out[material_id_col].dropna().astype(str).tolist()

    jarvis_ids = []
    if jarvis_id_col in out.columns:
        jarvis_ids = out[jarvis_id_col].dropna().astype(str).tolist()

    # Fetch state tables
    mp_state = fetch_mp_state(mp_ids, api_key=mp_api_key) if mp_ids else pd.DataFrame(columns=[material_id_col])
    jarvis_state = fetch_jarvis_state(jarvis_ids, db_name=jarvis_db) if jarvis_ids else pd.DataFrame(columns=[jarvis_id_col])

    # Merge them in (left joins so no rows drop)
    if not mp_state.empty and material_id_col in out.columns:
        out = out.merge(mp_state, how="left", left_on=material_id_col, right_on="material_id")
        if "material_id_y" in out.columns:
            out = out.drop(columns=["material_id_y"])
        if "material_id_x" in out.columns:
            out = out.rename(columns={"material_id_x": material_id_col})

    if not jarvis_state.empty and jarvis_id_col in out.columns:
        out = out.merge(jarvis_state, how="left", on="jarvis_id")

    # State-only view
    original_cols = set(df.columns)
    state_cols = [c for c in out.columns if c not in original_cols]
    state_only = out[[c for c in ([material_id_col, jarvis_id_col] + state_cols) if c in out.columns]].copy()

    return out, state_only


# -----------------------------
# Example CLI-style usage
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--material-id-col", default="material_id")
    ap.add_argument("--jarvis-id-col", default="jarvis_id")
    ap.add_argument("--jarvis-db", default="dft_3d")
    ap.add_argument("--print-state-only", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input, engine="python", quoting=csv.QUOTE_MINIMAL, escapechar="\\")

    # MP API key: pass via env var MP_API_KEY or hardcode in mp_api_key
    mp_key = os.environ.get("MP_API_KEY", None)

    df2, state_df = add_dataset_state(
        df,
        material_id_col=args.material_id_col,
        jarvis_id_col=args.jarvis_id_col,
        mp_api_key=mp_key,
        jarvis_db=args.jarvis_db,
    )

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 220)

    print("\n=== State columns preview ===")
    print(state_df.head(10))

    if args.output:
        df2.to_csv(args.output, index=False)
        print(f"\nSaved with state to: {args.output}")

    if args.print_state_only:
        # Print just column names
        print("\n=== Added state columns ===")
        for c in state_df.columns:
            print(c)
