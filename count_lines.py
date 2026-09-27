import os
base = r"student_resource\student_resource\dataset"
for split in ("train","test"):
    for f in sorted(os.listdir(os.path.join(base,split))):
        p = os.path.join(base,split,f)
        n = 0
        with open(p,"rb") as fh:
            while True:
                b = fh.read(1<<24)
                if not b: break
                n += b.count(b"\n")
        print(f"{split}/{f}: {n:,} lines (incl header)")
