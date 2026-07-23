import pandas as pd
import numpy as np
import re
import os
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "deeptree_italy_raw") + os.sep
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results") + os.sep
os.makedirs(RESULTS_DIR, exist_ok=True)

MA_TARGET_COL_CANDIDATES = ["M&A Target", "Target M&A", "M&A target", "target M&A", "m&a target"]
ACQUISITION_YEAR_COL_CANDIDATES = [
    "Year of acquisation/merger", "Year of acquisition/merger",
    "year of acquisation/merger", "year of acquisition/merger",
    "Year of Acquisation/merger", "Year of Acquisition/merger",
]


def coalesce_columns(df, candidates, unified_name):
    """
    Fusionne plusieurs colonnes qui representent la MEME information mais
    avec des noms/casses differents selon le fichier source (ex: 'M&A
    Target', 'Target M&A', 'M&A target' sont 3 colonnes distinctes apres
    un pd.concat, alors qu'elles devraient etre une seule). Pour chaque
    ligne, prend la premiere valeur non-nulle trouvee parmi les colonnes
    candidates presentes dans le dataframe, dans l'ordre de la liste.
    Cree/remplace la colonne 'unified_name' avec le resultat fusionne.
    """
    present = [c for c in candidates if c in df.columns]
    if not present:
        return df

    unified = df[present[0]].copy()
    for col in present[1:]:
        unified = unified.where(unified.notna(), df[col])

    df[unified_name] = unified
    return df


def dedupe_keeping_positive(df, ma_col, year_col):
    """
    Deduplique par clean_name SANS jamais perdre un statut positif : si
    au moins une des lignes en double pour une entreprise donnee est
    marquee positive, la ligne gardee est UNE des versions positives,
    jamais une version negative. drop_duplicates(keep='first') seul
    choisirait arbitrairement selon l'ordre des fichiers, pouvant effacer
    silencieusement un vrai positif au profit d'un doublon negatif du
    meme nom.

    Implementation via tri + drop_duplicates plutot que groupby().apply()
    -- ce dernier peut faire disparaitre la colonne de regroupement selon
    la version de pandas, ce qui casse le reste du pipeline.
    """
    def row_is_positive(row):
        raw_flag = str(row.get(ma_col, "")).strip().lower()
        if raw_flag in TRUTHY_VALUES:
            return True
        year_val = row.get(year_col)
        try:
            return pd.notna(year_val) and float(year_val) >= 1980
        except (TypeError, ValueError):
            return False

    df = df.copy()
    df["_tmp_is_positive"] = df.apply(row_is_positive, axis=1)
    # Trie pour que, au sein de chaque clean_name, les lignes positives
    # (_tmp_is_positive=True) arrivent en premier -- puis garde la
    # premiere occurrence de chaque clean_name.
    df = df.sort_values("_tmp_is_positive", ascending=False)
    df = df.drop_duplicates(subset="clean_name", keep="first")
    df = df.drop(columns=["_tmp_is_positive"])
    return df

TRUTHY_VALUES = {"1", "1.0", "true", "yes", "y", "oui", "si", "s\u00ec"}

BASE_METRICS = [
    "REVENUES", "EBITDA", "EBITDA MARGIN", "NET PROFIT", "NFP", "NFP / EBITDA",
    "ROIC", "ROE", "NET INVESTED CAPITAL",
    "Revenues", "Other Revenues", "Operating Revenues (Turnover)", "COGS",
    "Gross Profit", "Operating Expenses", "Services", "Personnel Expenses",
    "Use of third-party assets", "Other Operating Expenses", "Provisions",
    "Depreciation and Amortisation", "EBIT", "Financial Income",
    "Financial Expenses", "Net Financial Income (Expenses)",
    "Other Financial Gains (Expenses)", "EBT", "Taxes", "Net Income",
    "Intangible Fixed Assets", "Tangible Fixed Assets", "Financial Fixed Assets",
    "Total Fixed Assets", "Inventory", "Receivables", "Payables",
    "Working Capital", "Other Operating Assets", "Other Operating Liabilities",
    "Other Operating Assets (liabilities)", "Other Financial Assets (liabilities)",
    "Uses", "Debt", "(Cash)", "Net Financial Position", "Equity", "Sources",
    "(\u2206WC)", "(Taxes)", "(\u2206 Other Operating Assets)",
    "Operating Cash Flow", "(CAPEX)", "(\u2206 Financial Fixed Assets)",
    "(\u2206 Intangible Fixed Assets)", "Investing Cash Flow", "\u2206 Debt",
    "(\u2206 Other Financial Assets)", "Financial profit (expenses)",
    "\u2206 Equity", "Financing Cash Flow", "Net Cash Flow",
]


def clean_name(name):
    name = str(name).upper()
    name = re.sub(r"[^A-Z0-9 ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def read_csv_robust(path):
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise UnicodeDecodeError(f"Impossible de lire {path}")


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_all_data():
    base_paths = sorted(
        p for p in glob.glob(os.path.join(DATA_DIR, "Data*.csv"))
        if re.match(r"^Data\d+\.csv$", os.path.basename(p))
    )
    if not base_paths:
        raise FileNotFoundError(f"Aucun fichier Data<N>.csv trouve dans {DATA_DIR}")

    base_dfs = [read_csv_robust(p) for p in base_paths]
    base = pd.concat(base_dfs, ignore_index=True, sort=False)
    # Fusion des variantes de nom de colonne (M&A Target / Target M&A /
    # M&A target...) qui deviennent des colonnes SEPAREES apres concat si
    # les fichiers sources n'utilisent pas exactement la meme casse.
    base = coalesce_columns(base, MA_TARGET_COL_CANDIDATES, "M&A Target")
    base = coalesce_columns(base, ACQUISITION_YEAR_COL_CANDIDATES, "Year of acquisation/merger")
    base["clean_name"] = base["NAME"].apply(clean_name)
    n_before = len(base)
    base = dedupe_keeping_positive(base, "M&A Target", "Year of acquisation/merger")
    print(f"Base (DataN.csv) : {len(base_paths)} fichier(s), {n_before} lignes brutes -> {len(base)} entreprises "
          f"(dedup en preservant tout statut positif trouve).")

    new_paths = sorted(
        p for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if re.match(r"^NewData\d*\.csv$", os.path.basename(p), re.IGNORECASE)
    )
    new_paths += sorted(
        p for p in glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
        if re.match(r"^NewData.*\.xlsx$", os.path.basename(p), re.IGNORECASE)
    )

    if not new_paths:
        print("Aucun fichier NewData trouve -- base utilisee telle quelle.")
        return base

    new_dfs = []
    for p in new_paths:
        df = pd.read_excel(p) if p.lower().endswith(".xlsx") else read_csv_robust(p)
        new_dfs.append(df)
    new_data = pd.concat(new_dfs, ignore_index=True, sort=False)
    new_data = coalesce_columns(new_data, MA_TARGET_COL_CANDIDATES, "M&A Target")
    new_data = coalesce_columns(new_data, ACQUISITION_YEAR_COL_CANDIDATES, "Year of acquisation/merger")
    new_data["clean_name"] = new_data["NAME"].apply(clean_name)
    new_data = dedupe_keeping_positive(new_data, "M&A Target", "Year of acquisation/merger")
    print(f"Corrections (NewData) : {len(new_paths)} fichier(s), {len(new_data)} entreprises.")

    # Priorite absolue a NewData : remplace entierement la ligne de base
    # correspondante, meme si NewData a moins de colonnes -- on privilegie
    # la donnee la plus fiable/riche, pas une fusion colonne par colonne.
    overridden_names = set(new_data["clean_name"])
    base_kept = base[~base["clean_name"].isin(overridden_names)]
    merged = pd.concat([base_kept, new_data], ignore_index=True, sort=False)
    # Fusion de securite finale, au cas ou le concat base+new_data
    # reintroduirait des colonnes dupliquees par nom different.
    merged = coalesce_columns(merged, MA_TARGET_COL_CANDIDATES, "M&A Target")
    merged = coalesce_columns(merged, ACQUISITION_YEAR_COL_CANDIDATES, "Year of acquisation/merger")

    print(f"{len(base) - len(base_kept)} entreprise(s) remplacee(s) par leur version NewData.")
    print(f"Dataset fusionne : {len(merged)} entreprises uniques.")

    return merged


def parse_number(val):
    """
    Convertit une valeur potentiellement formatee (ex: '\u20ac1 166 621,00')
    en float. Retourne None si la conversion echoue ou si la valeur est
    invalide/manquante.
    """
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if s == "" or s.lower() == "n.a.":
        return None

    # Retire le symbole euro et espaces (y compris insecables) utilises
    # comme separateur de milliers
    s = s.replace("\u20ac", "").replace("\xa0", "").replace(" ", "")
    # Gere les parentheses comptables pour les negatifs, ex: (1.234,56)
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]

    # Determine le separateur decimal : si une virgule est presente et
    # apparait apres le dernier point (ou qu'il n'y a pas de point), elle
    # est le separateur decimal -- le point (le cas echeant) est alors un
    # separateur de milliers a retirer.
    if "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")

    try:
        result = float(s)
        return -result if negative else result
    except (TypeError, ValueError):
        return None


def build_metric_column_map(columns):
    """
    Precalcule, UNE SEULE FOIS, la correspondance metric -> {annee: nom_colonne}
    a partir de la liste des colonnes du dataframe. Evite de refaire un
    regex-match sur toutes les colonnes pour chaque ligne et chaque
    metrique (ce qui etait la cause de la lenteur : ~85 millions
    d'operations sur 1822 lignes x 57 metriques x 829 colonnes).
    """
    metric_map = {metric: {} for metric in BASE_METRICS}
    for metric in BASE_METRICS:
        pattern = re.compile(rf"^{re.escape(metric)} (\d{{4}})$")
        for col in columns:
            m = pattern.match(col)
            if m:
                metric_map[metric][int(m.group(1))] = col
    return metric_map


def get_metric_years(row, metric, metric_map):
    """Lookup direct via le mapping precalcule -- pas de regex ici."""
    result = {}
    for year, col in metric_map[metric].items():
        parsed = parse_number(row[col])
        if parsed is not None:
            result[year] = parsed
    return result


def is_positive(row, ma_col, year_col):
    raw_flag = str(row.get(ma_col, "")).strip().lower() if ma_col else ""
    flag_true = raw_flag in TRUTHY_VALUES
    year_val = row.get(year_col) if year_col else None
    has_valid_year = False
    if pd.notna(year_val):
        try:
            has_valid_year = float(year_val) >= 1980
        except (TypeError, ValueError):
            has_valid_year = False
    return flag_true or has_valid_year


def build_features_for_row(row, cutoff_year, metric_map, strict_exclusion=True):
    """
    strict_exclusion=True (positifs) : la ligne est exclue si AUCUNE
    metrique n'a de valeur exploitable avant/pendant le cutoff.
    strict_exclusion=False (negatifs) : jamais exclue pour ce motif --
    les features restent a NaN si aucune donnee n'est disponible (XGBoost
    gere les NaN nativement), la ligne est quand meme gardee.
    """
    """
    cutoff_year : derniere annee autorisee (INCLUSE) pour eviter la fuite.
    Pour un positif, cutoff_year = annee de rachat elle-meme (choix
    assume, cf. docstring du module). Pour un negatif, cutoff_year=None.
    """
    features = {}
    any_value_found = False

    for metric in BASE_METRICS:
        years_values = get_metric_years(row, metric, metric_map)
        if cutoff_year is not None:
            years_values = {y: v for y, v in years_values.items() if y <= cutoff_year}

        safe_metric = re.sub(r"[^a-zA-Z0-9]", "_", metric).strip("_").lower()

        if not years_values:
            features[f"{safe_metric}_latest"] = np.nan
            features[f"{safe_metric}_prev"] = np.nan
            features[f"{safe_metric}_growth"] = np.nan
            features[f"{safe_metric}_n_years"] = 0
            continue

        sorted_years = sorted(years_values.keys(), reverse=True)
        latest_year = sorted_years[0]
        latest_val = years_values[latest_year]
        features[f"{safe_metric}_latest"] = latest_val
        any_value_found = True

        if len(sorted_years) > 1:
            prev_year = sorted_years[1]
            prev_val = years_values[prev_year]
            features[f"{safe_metric}_prev"] = prev_val
            if prev_val != 0:
                features[f"{safe_metric}_growth"] = (latest_val - prev_val) / abs(prev_val)
            else:
                features[f"{safe_metric}_growth"] = np.nan
        else:
            features[f"{safe_metric}_prev"] = np.nan
            features[f"{safe_metric}_growth"] = np.nan

        features[f"{safe_metric}_n_years"] = len(years_values)

    if strict_exclusion and not any_value_found:
        return None, "Aucune donnee financiere disponible avant/pendant le cutoff"

    return features, None


def main():
    full = load_all_data()

    ma_col = get_col(full, MA_TARGET_COL_CANDIDATES)
    year_col = get_col(full, ACQUISITION_YEAR_COL_CANDIDATES)
    print(f"Colonne label utilisee : '{ma_col}' / '{year_col}'")

    print("Precalcul de la correspondance metrique -> colonnes...")
    metric_map = build_metric_column_map(full.columns)
    print("Fait.")

    rows_out = []
    excluded_log = []
    total_rows = len(full)

    for i, (_, row) in enumerate(full.iterrows()):
        if i % 200 == 0:
            print(f"Traitement... {i}/{total_rows}")

        try:
            positive = is_positive(row, ma_col, year_col)

            if positive:
                year_x = row.get(year_col)
                try:
                    year_x_valid = pd.notna(year_x) and float(year_x) >= 1980
                except (TypeError, ValueError):
                    year_x_valid = False

                if not year_x_valid:
                    excluded_log.append((row["NAME"], "Positif mais annee de rachat manquante/invalide"))
                    continue
                cutoff_year = int(float(year_x))
            else:
                cutoff_year = None

            features, reason = build_features_for_row(row, cutoff_year, metric_map, strict_exclusion=positive)
            if reason:
                excluded_log.append((row["NAME"], reason))
                continue

            features["name"] = row["NAME"]
            features["is_ma_target"] = 1 if positive else 0
            rows_out.append(features)

        except Exception as e:
            excluded_log.append((row.get("NAME", f"ligne {i}"), f"ERREUR: {type(e).__name__}: {e}"))
            continue

    print(f"Traitement termine : {total_rows}/{total_rows}")

    final_df = pd.DataFrame(rows_out)

    n_pos = final_df["is_ma_target"].sum() if len(final_df) else 0
    n_neg = len(final_df) - n_pos
    print(f"\nTotal exclus : {len(excluded_log)}")
    print(f"Dataset final : {len(final_df)} lignes ({n_pos} positifs, {n_neg} negatifs)")

    output_data_path = os.path.join(SCRIPT_DIR, "..", "data", "training_set_italy.csv")
    final_df.to_csv(output_data_path, index=False)

    excluded_df = pd.DataFrame(excluded_log, columns=["name", "reason"])
    output_excluded_path = os.path.join(RESULTS_DIR, "excluded_players_log.csv")
    excluded_df.to_csv(output_excluded_path, index=False)
    print(f"Journal des exclusions sauvegarde ({len(excluded_df)} entreprises).")


if __name__ == "__main__":
    main()
