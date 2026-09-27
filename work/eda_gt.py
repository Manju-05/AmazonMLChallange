import pandas as pd, numpy as np, collections
p = r"student_resource\student_resource\dataset\train\train_ground_truth.tsv"
gt = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
print("GT rows:", len(gt))
print("unique s1:", gt.source1_entity_id.nunique())
# distribution of #matches
cnt = gt.matched_entity_ids.apply(lambda s: 0 if s=="" else len(s.split(",")))
print("\n== #matches per S1 ==")
print(cnt.value_counts().sort_index().head(30))
print("\nmean", cnt.mean(), "median", cnt.median(), "max", cnt.max())
print("singletons:", (cnt==0).sum(), f"({(cnt==0).mean():.3%})")
# id prefix distribution
allids = [i for s in gt.matched_entity_ids for i in (s.split(",") if s else [])]
print("\ntotal matched ids:", len(allids))
pref = collections.Counter(i.split("-")[0] for i in allids)
print("prefix counts:", pref)
# KEY ASSUMPTION: does each S2/S3 map to at most 1 S1?
rev = collections.Counter(allids)
dupes = {k:v for k,v in rev.items() if v>1}
print("\nS2/S3 records matched by >1 S1 entity:", len(dupes), "total extra links:", sum(v-1 for v in dupes.values()))
print("max times a single S2/S3 is referenced:", max(rev.values()) if rev else 0)
