"""临时隔离测试脚本"""
import subprocess, pathlib, sys, re

targets = sys.argv[1:] if len(sys.argv) > 1 else ["P24X176", "P25X177"]

input_dir = pathlib.Path("data/input")
for f in sorted(input_dir.iterdir()):
    for t in targets:
        if t in f.name:
            print(f"=== Processing {t} ===")
            subprocess.run(
                [sys.executable, "main.py", "--image", str(f), "--force-all", "--format", "html"],
                check=False,
            )
            break

html_dir = pathlib.Path("data/output/html")
for h in sorted(html_dir.iterdir()):
    for t in targets:
        if t in h.name:
            content = h.read_text(encoding="utf-8", errors="replace")
            td_count = len(re.findall(r"<td", content))
            tr_count = len(re.findall(r"<tr", content))
            table_count = len(re.findall(r"<table", content))
            print(f"\n{t}: tables={table_count}, rows={tr_count}, cells={td_count}" +
                  (f", avg={td_count/tr_count:.1f}" if tr_count else ""))
            break
