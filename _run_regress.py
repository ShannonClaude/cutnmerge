"""全量回归：对所有问题图片跑 pipeline 并统计结构"""
import subprocess, pathlib, sys, re

targets = ["P24X176", "P25X177", "P32X266", "P33X267", "P33X234", "P34X235",
           "P27X225", "P28X229", "P29X231"]

input_dir = pathlib.Path("data/input")
images = []
for f in sorted(input_dir.iterdir()):
    for t in targets:
        if t in f.name:
            images.append(f)
            break

print(f"Found {len(images)} images to process")
for f in images:
    tag = [t for t in targets if t in f.name][0]
    print(f"\n=== {tag}: {f.name} ===")
    subprocess.run(
        [sys.executable, "main.py", "--image", str(f), "--force-all", "--format", "html"],
        check=False,
    )

print("\n\n=== RESULTS SUMMARY ===")
html_dir = pathlib.Path("data/output/html")
for h in sorted(html_dir.iterdir()):
    for t in targets:
        if t in h.name:
            content = h.read_text(encoding="utf-8", errors="replace")
            td_count = len(re.findall(r"<td", content))
            tr_count = len(re.findall(r"<tr", content))
            table_count = len(re.findall(r"<table", content))
            print(f"{t}: tables={table_count} rows={tr_count} cells={td_count}" +
                  (f" avg={td_count/tr_count:.1f}" if tr_count else " EMPTY"))
            break
