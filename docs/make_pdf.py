import markdown, weasyprint, pathlib, re
d = pathlib.Path(__file__).resolve().parent
src = re.sub(r'^([^\s\-\d|].*)\n(?=\s*(- |\d+\. ))', r'\1\n\n', (d/'PAPER.md').read_text(), flags=re.M)
src = re.sub(r'^((?:  )+)(?=- )', lambda m: m[1] * 2, src, flags=re.M)
body = markdown.markdown(src, extensions=['tables', 'fenced_code'])
css = """@page{size:A4;margin:18mm}body{font:10pt/1.4 'DejaVu Serif',serif}
h1{font-size:16pt}h2{font-size:13pt;margin-top:1.2em}h3{font-size:11pt}
table{border-collapse:collapse;font-size:8.5pt;margin:.6em 0}td,th{border:1px solid #999;padding:2px 5px}
code,pre{font:8pt 'DejaVu Sans Mono',monospace}pre{white-space:pre-wrap;background:#f4f4f4;padding:6px}
img{max-width:100%}"""
html = f"<html><head><meta charset=utf-8><style>{css}</style></head><body>{body}</body></html>"
weasyprint.HTML(string=html, base_url=str(d)).write_pdf(d/'PAPER.pdf')
