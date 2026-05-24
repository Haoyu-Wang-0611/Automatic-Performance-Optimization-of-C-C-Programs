import subprocess
import sys
import os
import re
import math
import time
from pathlib import Path
from typing import List, Dict, Set

DEFAULT_WORK_DIR = Path(__file__).parent.resolve()
WORK_DIR = Path(os.getenv("OPTIMIZER_WORK_DIR", DEFAULT_WORK_DIR)).resolve()
BASELINE_SRC = WORK_DIR / "target_baseline.cpp"
TARGET_SRC = WORK_DIR / "target.cpp"
BASELINE_BIN = WORK_DIR / "verif_base"
TARGET_BIN = WORK_DIR / "verif_opt"
TOLERANCE = 1e-4

def compile_program(src: Path, bin_out: Path) -> bool:
    """Compiles the C++ program."""
    # -lm is for math library
    cmd = ["g++", "-O3", str(src), "-o", str(bin_out), "-lm"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=WORK_DIR)
    if res.returncode != 0:
        print(f"[COMPILE ERROR] {src.name}:\n{res.stderr}")
        return False
    return True

def get_file_snapshot(dir_path: Path) -> Dict[str, float]:
    """Returns a dict of filename -> mtime for all files in directory."""
    snapshot = {}
    for f in dir_path.glob("**/*"):
        if f.is_file():
            try:
                snapshot[str(f.relative_to(dir_path))] = f.stat().st_mtime
            except:
                pass
    return snapshot

def extract_numbers(text: str) -> List[float]:
    """Extracts all floating point numbers from a text."""
    # Matches scientific notation like -1.23e-4 or simple 1.23
    return [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', text)]

def is_time_line(line: str) -> bool:
    """Heuristic to detect if a line is just reporting execution time."""
    lower = line.lower()
    keywords = ["time", "second", "elapsed", "completed in", "clock", "xxx"]
    # If line has time keywords and at least one number, it's likely a timer
    if any(k in lower for k in keywords) and re.search(r'\d', line):
        return True
    return False

def compare_texts(text1: str, text2: str, source_label: str) -> bool:
    """Compares two texts numerically, ignoring lines that look like timers."""
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    # Filter out timing lines
    data1 = [l for l in lines1 if not is_time_line(l)]
    data2 = [l for l in lines2 if not is_time_line(l)]

    # Join back to treat as a stream of numbers
    clean1 = "\n".join(data1)
    clean2 = "\n".join(data2)

    nums1 = extract_numbers(clean1)
    nums2 = extract_numbers(clean2)

    if len(nums1) != len(nums2):
        print(f"[MISMATCH] {source_label}: Number count mismatch.")
        print(f"  Baseline has {len(nums1)} numbers, Optimized has {len(nums2)} numbers.")
        return False

    max_diff = 0.0
    for i, (n1, n2) in enumerate(zip(nums1, nums2)):
        diff = abs(n1 - n2)
        if diff > max_diff:
            max_diff = diff
        
        if diff > TOLERANCE:
            print(f"[MISMATCH] {source_label}: Logic difference detected.")
            print(f"  At number constant #{i+1}: Base={n1}, Opt={n2}, Diff={diff}")
            # Show context
            print(f"  Context (Base): ...{nums1[max(0,i-2):i+3]}...")
            return False

    print(f"  [OK] {source_label}: Parsed {len(nums1)} numbers. Max Diff: {max_diff:.6e}")
    return True

def run_and_track(bin_path: Path) -> tuple[str, Set[str]]:
    """Runs binary, captures stdout, and identifies output files."""
    
    # 1. Snapshot Before
    before = get_file_snapshot(WORK_DIR)
    
    # 2. Run
    try:
        res = subprocess.run(
            [str(bin_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=WORK_DIR,
        )
        stdout = res.stdout
        if res.returncode != 0:
            print(f"[RUNTIME ERROR] {bin_path.name} exited with code {res.returncode}")
            print(f"Stderr: {res.stderr}")
            return None, None
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {bin_path.name} took too long.")
        return None, None

    # 3. Snapshot After
    after = get_file_snapshot(WORK_DIR)
    
    # 4. Identify Generated/Modified Files
    changed_files = set()
    for f, mtime in after.items():
        if f not in before or before[f] != mtime:
            # Ignore binary verif files themselves and intermediate build artifacts
            if "verif_" in f or f.endswith(".o"):
                continue
            # Ignore standard perf files
            if f == "perf.data" or f == "perf.data.old":
                continue
            # Ignore python scripts
            if f.endswith(".py"):
                continue
                
            changed_files.add(f)
            
    return stdout, changed_files

def verify():
    print("=" * 70)
    print(" Universal Accuracy Verification (Auto-Detect)")
    print("=" * 70)

    # 1. Clean previous artifacts
    if BASELINE_BIN.exists(): BASELINE_BIN.unlink()
    if TARGET_BIN.exists(): TARGET_BIN.unlink()

    # 2. Compile
    if not compile_program(BASELINE_SRC, BASELINE_BIN): return
    if not compile_program(TARGET_SRC, TARGET_BIN): return

    print("Running Baseline...")
    base_out, base_files = run_and_track(BASELINE_BIN)
    if base_out is None: return

    print("Running Optimized...")
    opt_out, opt_files = run_and_track(TARGET_BIN)
    if opt_out is None: return

    # 3. Compare Stdout (ignoring timers)
    print("Comparing Standard Output...")
    if not compare_texts(base_out, opt_out, "STDOUT"):
        sys.exit(1)

    # 4. Compare Output Files
    print(f"Comparing Output Files ({len(base_files)} detected)...")
    
    # Check if optimized missed any files
    missing = base_files - opt_files
    if missing:
        print(f"[MISMATCH] Optimized version failed to generate files: {missing}")
        sys.exit(1)

    for fname in base_files:
        print(f"  Checking '{fname}'...")
        try:
            # Try reading as text
            f1 = (WORK_DIR / fname).read_text(errors='ignore')
            # Currently we assume we run optimized immediately after, so the file on disk IS the optimized version.
            # Wait, run_and_track runs sequentially. 
            # Baseline runs -> writes file. Optimized runs -> overwrites file.
            # Problem: We need to SAVE baseline output before running optimized.
            # My logic above in run_and_track was just detecting *what* changed.
            # But run_and_track(TARGET_BIN) overwrote the files!
            pass 
        except:
             pass

    # RE-RUN STRATEGY FOR FILES:
    # Since capturing files is tricky without temp dirs, we will simplify:
    # We already know WHICH files are generated (base_files).
    # We will rename Baseline files to .base_verif
    
    for fname in base_files:
        p = WORK_DIR / fname
        bak = WORK_DIR / (fname + ".base_verif")
        # Note: These are the files from OPTIMIZED run (latest run)
        # This logic is flawed. Let's fix.
        
    # Correct Logic:
    # 1. Run Base -> Move generated files to *.base.bak
    # 2. Run Opt -> Files are generated in place.
    # 3. Compare *.base.bak vs generated files.
    
    # Let's Restart the run sequence correctly
    
    # --- PROPER FILE COMPARISON RUN ---
    # Clean possible output files first
    for f in base_files:
        p = WORK_DIR / f
        if p.exists(): p.unlink()
        
    # Run Baseline Again
    # (We could have done this in step 1 if we planned ahead, but rerunning is safer to be sure)
    subprocess.run(
        [str(BASELINE_BIN)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=WORK_DIR,
    )
    
    # Rename outputs
    for fname in base_files:
        p = WORK_DIR / fname
        bak = WORK_DIR / (fname + ".verif_bak")
        if p.exists():
            p.replace(bak)
            
    # Run Optimized
    subprocess.run(
        [str(TARGET_BIN)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=WORK_DIR,
    )
    
    # Compare
    all_files_match = True
    for fname in base_files:
        base_bak = WORK_DIR / (fname + ".verif_bak")
        curr_opt = WORK_DIR / fname
        
        if not curr_opt.exists():
             print(f"[MISMATCH] Output file '{fname}' missing from optimized run.")
             all_files_match = False
             continue
             
        # Read content
        try:
             # Try text comparison with tolerance
             txt_base = base_bak.read_text()
             txt_opt = curr_opt.read_text()
             if not compare_texts(txt_base, txt_opt, f"File:{fname}"):
                 all_files_match = False
        except UnicodeDecodeError:
             # Binary comparison
             import hashlib
             d1 = base_bak.read_bytes()
             d2 = curr_opt.read_bytes()
             if hashlib.md5(d1).digest() != hashlib.md5(d2).digest():
                 print(f"[MISMATCH] Binary file '{fname}' differs.")
                 all_files_match = False
             else:
                 print(f"  [OK] File:{fname} (Binary match)")

        # Cleanup backup
        if base_bak.exists(): base_bak.unlink()

    if not all_files_match:
        sys.exit(1)

    print("-" * 60)
    print("[SUCCESS] All outputs match within tolerance.")
    sys.exit(0)

if __name__ == "__main__":
    verify()
