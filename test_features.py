# pip install mp_api pymatgen pandas tqdm

import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure

MP_API_KEY = "RmYSluuvUC3TJADjNCdvrN1AyifPkfog"

def enrich_dataframe(df):
    enriched_rows = []

    with MPRester(MP_API_KEY) as mpr:
        for _, row in df.iterrows():
            mp_id = row["material_id"]

            doc = mpr.materials.summary.search(
                material_ids=[mp_id],
                fields=["structure", "symmetry", "nsites", "volume", "density"]
            )[0]

            structure = doc.structure

            # Try oxidation states from MP → fallback to pymatgen guess
            try:
                ox_doc = mpr.materials.oxidation_states.search(
                    material_ids=[mp_id]
                )[0]
                ox_structure = ox_doc.structure
            except Exception:
                ox_structure = structure.copy()
                try:
                    ox_structure.add_oxidation_state_by_guess()
                except Exception:
                    pass

            ox_states = [
                getattr(site.specie, "oxi_state", None)
                for site in ox_structure.sites
            ]

            enriched_row = dict(row)

            # GLOBAL FEATURES
            enriched_row["spacegroup_number"] = doc.symmetry.number
            enriched_row["spacegroup_symbol"] = doc.symmetry.symbol
            enriched_row["crystal_system"] = doc.symmetry.crystal_system
            enriched_row["nsites_mp"] = doc.nsites
            enriched_row["volume_mp"] = doc.volume
            enriched_row["density_mp"] = doc.density
            enriched_row["volume_per_atom"] = doc.volume / doc.nsites

            # NODE FEATURES
            enriched_row["site_oxidation_states"] = ox_states

            enriched_rows.append(enriched_row)
    print(enriched_rows)    
    return pd.DataFrame(enriched_rows)


# from functools import lru_cache

# FIELDS = [
#     "material_id",
#     "structure",
#     "symmetry",          # space group info etc.
#     "nsites",
#     "volume",
#     "density",
#     "formula_pretty",
# ]

# @lru_cache(maxsize=None)
# def fetch_mp_doc(mp_id: str):
#     with MPRester(MP_API_KEY) as mpr:
#         doc = mpr.materials.summary.search(
#             material_ids=[mp_id],
#             fields=FIELDS
#         )[0]
#         # structure shortcut also exists in docs
#         # structure = mpr.get_structure_by_material_id(mp_id)
#         return doc

# def fetch_oxidation_states(mp_id: str):
#     """
#     Prefer MP oxidation states route if you use it in your environment.
#     If unavailable for some materials, fallback to pymatgen guess.
#     """
#     with MPRester(MP_API_KEY) as mpr:
#         # Depending on your mp-api version, oxidation states are accessible via a route.
#         # The API exposes "Materials Oxidation States" endpoints.  :contentReference[oaicite:8]{index=8}
#         ox_docs = mpr.materials.oxidation_states.search(material_ids=[mp_id])
#         if ox_docs and getattr(ox_docs[0], "structure", None) is not None:
#             return ox_docs[0].structure  # oxidation-decorated Structure (best case)
#     return None

# def enrich_row(row):
#     mp_id = row["material_id"]
#     doc = fetch_mp_doc(mp_id)

#     # Base structure
#     struct: Structure = doc.structure

#     # Oxidation-decorated structure (best) else guess
#     ox_struct = fetch_oxidation_states(mp_id)
#     if ox_struct is None:
#         # pymatgen fallback: will not always succeed
#         try:
#             ox_struct = struct.copy()
#             ox_struct.add_oxidation_state_by_guess()
#         except Exception:
#             ox_struct = struct  # last resort: keep undecorated

#     # Global features
#     sym = doc.symmetry
#     row["spacegroup_number"] = getattr(sym, "number", None)
#     row["spacegroup_symbol"] = getattr(sym, "symbol", None)

#     row["nsites_mp"] = doc.nsites
#     row["volume_mp"] = doc.volume
#     row["density_mp"] = doc.density
#     row["volume_per_atom"] = (doc.volume / doc.nsites) if doc.volume and doc.nsites else None

#     # Node-level oxidation states (store for later featurization)
#     # Each site.specie may now carry oxidation state; store as list aligned with sites
#     ox_list = []
#     for site in ox_struct.sites:
#         sp = site.specie
#         ox = getattr(sp, "oxi_state", None)
#         ox_list.append(float(ox) if ox is not None else None)
#     row["site_oxidation_states"] = ox_list

#     return row

# df = pd.read_csv("train.csv")  # your schema
# df_enriched = df.apply(enrich_row, axis=1)
# df_enriched.to_parquet("train.enriched.parquet", index=False)
