import pandas as pd, numpy as np, os, sys
base = r"student_resource\student_resource\dataset"
def asc(s):
    return s.encode("ascii","backslashreplace").decode()[:60]
def prof(split, src, n=400000):
    p = os.path.join(base, split, f"{split}_source{src}.tsv")
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, nrows=n)
    print(f"--- {split} source{src} (n={len(df)}) ---")
    print("country:", {k:round(v,4) for k,v in df.country.value_counts(normalize=True).items()})
    print("empty name:", round((df.business_name.str.strip()=="").mean(),4),
          "empty addr:", round((df.business_address.str.strip()=="").mean(),4))
    print("name len med:", df.business_name.str.len().median(), "addr len med:", df.business_address.str.len().median())
    print("name has digit:", round(df.business_name.str.contains(r"\d").mean(),3),
          "| addr has digit:", round(df.business_address.str.contains(r"\d").mean(),3))
    print("NAME :", [asc(x) for x in df.business_name.head(5)])
    print("ADDR :", [asc(x) for x in df.business_address.head(3)])
for s in ("train","test"):
    for i in (1,2,3): prof(s,i)
    print()
