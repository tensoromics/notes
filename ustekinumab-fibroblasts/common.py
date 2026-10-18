"""Shared loaders: GEO series matrices, GPL13158 probe annotation, HGNC symbols."""
import gzip
import io
import pandas as pd

DATA = "data"


def read_series_matrix(path):
    """Return (expression: probes x samples, samples: one row per sample)."""
    with gzip.open(path, "rt") as fh:
        text = fh.read()
    head, rest = text.split("!series_matrix_table_begin\n")
    table = rest.split("!series_matrix_table_end")[0]
    expr = pd.read_csv(io.StringIO(table), sep="\t", index_col=0)

    fields = {}
    for line in head.split("\n"):
        if not line.startswith("!Sample_"):
            continue
        key, *values = line.split("\t")
        values = [v.strip('"') for v in values]
        if key == "!Sample_characteristics_ch1":
            # one line per characteristic, "name: value" in each cell
            name = values[0].split(":", 1)[0].strip()
            fields[name] = [v.split(":", 1)[1].strip() if ":" in v else v for v in values]
        else:
            fields.setdefault(key[len("!Sample_"):], values)
    samples = pd.DataFrame(fields)
    samples.index = samples["geo_accession"]
    return expr, samples


def hgnc_entrez_to_symbol():
    hgnc = pd.read_csv(f"{DATA}/hgnc_complete_set.txt", sep="\t", dtype=str, low_memory=False)
    hgnc = hgnc.dropna(subset=["entrez_id"])
    return dict(zip(hgnc["entrez_id"], hgnc["symbol"]))


def probe_to_symbol(platform="GPL13158"):
    """Map array probes to current HGNC symbols through Entrez IDs.

    Probes that hit more than one gene are dropped.
    """
    with gzip.open(f"{DATA}/{platform}.annot.gz", "rt", errors="replace") as fh:
        lines = fh.read().split("\n")
    start = lines.index("!platform_table_begin") + 1
    end = lines.index("!platform_table_end")
    annot = pd.DataFrame([l.split("\t") for l in lines[start + 1 : end]], columns=lines[start].split("\t"))
    annot = annot[(annot["Gene ID"] != "") & ~annot["Gene ID"].str.contains("///")]
    to_symbol = hgnc_entrez_to_symbol()
    symbols = annot["Gene ID"].map(to_symbol)
    return pd.Series(symbols.values, index=annot["ID"].values).dropna()


def gene_matrix(expr, probe_map):
    """Collapse probes to genes, keeping the probe with the highest mean per gene."""
    expr = expr.loc[expr.index.intersection(probe_map.index)]
    best = expr.mean(axis=1).groupby(probe_map.loc[expr.index]).idxmax()
    out = expr.loc[best.values]
    out.index = best.index
    return out
