import tilelang
import tilelang.language as T

print("=== PassConfigKey ===")
for attr in dir(tilelang.PassConfigKey):
    if not attr.startswith("_"):
        print(f"  {attr}")

print("\n=== T.GemmWarpPolicy ===")
if hasattr(T, "GemmWarpPolicy"):
    for attr in dir(T.GemmWarpPolicy):
        if not attr.startswith("_"):
            print(f"  {attr}")
else:
    print("  NOT FOUND")

print("\n=== T.gemm signature ===")
import inspect
if hasattr(T, "gemm"):
    try:
        sig = inspect.signature(T.gemm)
        print(f"  {sig}")
    except:
        print("  Could not get signature")
else:
    print("  NOT FOUND")

print("\n=== T.wgmma_gemm ===")
if hasattr(T, "wgmma_gemm"):
    try:
        sig = inspect.signature(T.wgmma_gemm)
        print(f"  {sig}")
    except:
        print("  Could not get signature")
else:
    print("  NOT FOUND")

print("\n=== T.tcgen05_gemm ===")
if hasattr(T, "tcgen05_gemm"):
    try:
        sig = inspect.signature(T.tcgen05_gemm)
        print(f"  {sig}")
    except:
        print("  Could not get signature")
else:
    print("  NOT FOUND")

print("\n=== Other T attributes ===")
for attr in dir(T):
    if "gemm" in attr.lower() or "warp" in attr.lower() or "policy" in attr.lower():
        print(f"  {attr}")

print("\nDONE")
