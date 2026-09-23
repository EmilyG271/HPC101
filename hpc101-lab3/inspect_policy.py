import tilelang.language as T
print("GemmWarpPolicy values:")
if hasattr(T, "GemmWarpPolicy"):
    p = T.GemmWarpPolicy
    for attr in dir(p):
        if not attr.startswith("_"):
            try:
                val = getattr(p, attr)
                print(f"  {attr} = {val}")
            except:
                pass
else:
    print("  NOT FOUND")
print("DONE")
