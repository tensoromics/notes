"""Extract every variant TenK10K tested against the ETS2 enhancer peak (chr21:39094096-39094839) in CD14 monocytes.

Source: Hugging Face anglixue/TenK10K_multiome (CC BY 4.0),
caQTL_common_TensorQTL/CD14_Mono/TenK10K.cis_qtl_pairs.chr21.parquet (about 940 MB). The script reads the
phenotype_id column of each row group over HTTP, then reads full rows only from the row groups that hold this peak.
Output: a CSV of 7,356 rows (data/ets2_peak_cd14_cis_pairs.csv in this folder).

    python extract_ets2_peak.py data/ets2_peak_cd14_cis_pairs.csv
"""
import sys

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

URL = ("https://huggingface.co/datasets/anglixue/TenK10K_multiome/resolve/main/"
       "caQTL_common_TensorQTL/CD14_Mono/TenK10K.cis_qtl_pairs.chr21.parquet")
PEAK = "chr21:39094096-39094839"

out = sys.argv[1]
fs = fsspec.filesystem("http")
with fs.open(URL, "rb", block_size=8 * 2**20) as f:
    pf = pq.ParquetFile(f)
    groups = [i for i in range(pf.metadata.num_row_groups)
              if pc.any(pc.equal(pf.read_row_group(i, columns=["phenotype_id"])["phenotype_id"], PEAK)).as_py()]
    print("row groups holding the peak:", groups)
    tables = [pf.read_row_group(i) for i in groups]
table = pa.concat_tables(tables)
table = table.filter(pc.equal(table["phenotype_id"], PEAK))
table.to_pandas().to_csv(out, index=False)
print(table.num_rows, "rows ->", out)
