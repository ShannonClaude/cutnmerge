import sys
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup


def clean_cell(cell) -> str:
    """提取单元格文本，替换 <br> 为单一空格，转义 Markdown 竖线"""
    # 将 <br> 标签替换为单个空格
    for br in cell.find_all("br"):
        br.replace_with(" ")
    text = cell.get_text(separator=" ", strip=True)
    text = text.replace("|", "\\|")
    return " ".join(text.split())


def table_to_markdown(table) -> str:
    """利用 2D 矩阵模拟并解析复杂合并单元格的 <table>"""
    rows = table.find_all("tr")
    if not rows:
        return ""

    # 1. 探测表头所占的行深度（header_depth）
    header_depth = 1
    for cell in rows[0].find_all(["th", "td"]):
        try:
            rs = int(cell.get("rowspan", 1))
            if rs > header_depth:
                header_depth = rs
        except (ValueError, TypeError):
            pass

    thead = table.find("thead")
    if thead:
        thead_rows = thead.find_all("tr")
        if len(thead_rows) > header_depth:
            header_depth = len(thead_rows)

    # 2. 构建 2D 网格映射
    grid = {}
    max_cols = 0
    num_rows = len(rows)

    for r_idx, tr in enumerate(rows):
        c_idx = 0
        cells = tr.find_all(["th", "td"])
        for cell in cells:
            # 跳过此前 rowspan / colspan 已占用的网格
            while (r_idx, c_idx) in grid:
                c_idx += 1

            try:
                rowspan = int(cell.get("rowspan", 1))
            except (ValueError, TypeError):
                rowspan = 1
            try:
                colspan = int(cell.get("colspan", 1))
            except (ValueError, TypeError):
                colspan = 1

            text = clean_cell(cell)

            # 填充该单元格所覆盖的所有子坐标
            for dr in range(rowspan):
                for dc in range(colspan):
                    target_r = r_idx + dr
                    target_c = c_idx + dc
                    if dr == 0 and dc == 0:
                        grid[(target_r, target_c)] = text
                    else:
                        # 若在表头区域，保留父级文本用于多级表头合并
                        if target_r < header_depth:
                            grid[(target_r, target_c)] = text
                        else:
                            # 数据体区域的跨列，其余位置填空，保持列对齐
                            grid[(target_r, target_c)] = ""

            c_idx += colspan
            max_cols = max(max_cols, c_idx)

    if max_cols == 0 or num_rows == 0:
        return ""

    # 3. 智能合并表头（多层表头合并为：一级表头 - 二级表头）
    headers = []
    for c in range(max_cols):
        parts = []
        for r in range(header_depth):
            val = grid.get((r, c), "").strip()
            if val and val not in parts:
                parts.append(val)
        headers.append(" - ".join(parts) if parts else " ")

    # 4. 提取数据行
    data_rows = []
    for r in range(header_depth, num_rows):
        row_data = [grid.get((r, c), "").strip() for c in range(max_cols)]
        data_rows.append(row_data)

    # 5. 组装 Markdown 表格
    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in data_rows:
        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)


def html_to_markdown(content: str) -> str:
    """将 HTML 字符串转为 Markdown（保留标题/段落/表格顺序）。"""
    if not (content or "").strip():
        return ""

    soup = BeautifulSoup(content, "html.parser")

    # 提取顶层文档元素（保留标题、段落和表格的先后顺序）
    elements = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "table"])
    md_blocks = []

    for elem in elements:
        # 跳过嵌套在 table 内部的标签
        if elem.name != "table" and elem.find_parent("table"):
            continue
        if elem.name == "table" and elem.find_parent("table"):
            continue

        if elem.name.startswith("h"):
            level = elem.name[1]
            text = elem.get_text(strip=True)
            if text:
                md_blocks.append(f"{'#' * int(level)} {text}\n\n")
        elif elem.name == "p":
            text = elem.get_text(strip=True)
            if text:
                md_blocks.append(f"**{text}**\n\n")
        elif elem.name == "table":
            md_table = table_to_markdown(elem)
            if md_table:
                md_blocks.append(md_table + "\n\n")

    if md_blocks:
        return "".join(md_blocks).strip() + "\n"

    # 无结构化标签时：退回纯文本（兼容游离文本直出）
    text = soup.get_text(separator="\n", strip=True)
    return (text + "\n") if text else ""


def convert_html_to_md(html_path_str: str) -> Optional[Path]:
    """从 HTML 文件读取并写出同名 .md；成功返回输出路径。"""
    html_path = Path(html_path_str).resolve()
    if not html_path.is_file():
        print(f"[!] 找不到文件: {html_path}")
        return None

    content = None
    for enc in ["utf-8", "gbk", "gb18030", "utf-16"]:
        try:
            content = html_path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if content is None:
        print("[!] 错误: 无法解析文件编码。")
        return None

    md = html_to_markdown(content)
    if not md.strip():
        print("[!] 提示: 未在 HTML 中提取到有效内容。")
        return None

    output_path = html_path.with_suffix(".md")
    output_path.write_text(md if md.endswith("\n") else md + "\n", encoding="utf-8")
    print(f"[+] 转换成功 -> {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python html2md.py <HTML文件路径>")
        sys.exit(1)
    convert_html_to_md(sys.argv[1])