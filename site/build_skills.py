#!/usr/bin/env python3
"""
Build static skill subpages for AtomisticSkills docs.
Reads SKILL.md + example READMEs from .agents/skills/, writes HTML to site/skills/.
Run from the project root: python site/build_skills.py
"""

import os
import re
import json
import shutil
import base64
from pathlib import Path

# Fix: Define PROJECT_ROOT relative to this file's location (site/build_skills.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SKILLS_SRC_DIR = PROJECT_ROOT / ".agents" / "skills"
DOCS_SRC_DIR = PROJECT_ROOT / "docs"
SITE_DIR = PROJECT_ROOT / "site"
SKILLS_OUT_DIR = SITE_DIR / "skills"

CAT_COLORS = {
    "materials":        {"bg": "#eff6ff", "border": "#bfdbfe", "text": "#1d4ed8", "dot": "#3b82f6"},
    "chemistry":        {"bg": "#f0fdf4", "border": "#bbf7d0", "text": "#15803d", "dot": "#22c55e"},
    "machine-learning": {"bg": "#f3f0ff", "border": "#d4c8ff", "text": "#5b3de8", "dot": "#816cff"},
    "drug-discovery":   {"bg": "#fdf2f8", "border": "#f5d0fe", "text": "#a21caf", "dot": "#d946ef"},
    "general":          {"bg": "#fffbeb", "border": "#fde68a", "text": "#92400e", "dot": "#f59e0b"},
    "thermodynamics":   {"bg": "#ecfeff", "border": "#a5f3fc", "text": "#0e7490", "dot": "#06b6d4"},
}

def read_file(path):
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""

def parse_frontmatter(content):
    """Extract YAML frontmatter from markdown."""
    meta = {"name": "", "description": "", "category": []}
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            fm = content[3:end].strip()
            for line in fm.splitlines():
                if line.startswith("name:"):
                    meta["name"] = line.split(":", 1)[1].strip()
                elif line.startswith("description:"):
                    meta["description"] = line.split(":", 1)[1].strip()
                elif line.startswith("category:"):
                    raw = line.split(":", 1)[1].strip()
                    # handle [a, b] or single value
                    raw = raw.strip("[]")
                    meta["category"] = [c.strip() for c in raw.split(",")]
            content = content[end+3:].strip()
    return meta, content

def encode_image(img_path: Path) -> str | None:
    """Return base64 data URI for an image, or None if not found."""
    if not img_path.exists():
        return None
    ext = img_path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "svg": "image/svg+xml", "webp": "image/webp"}.get(ext[1:])
    if not mime:
        return None
    data = base64.b64encode(img_path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"

def process_markdown_images(md: str, base_dir: Path) -> str:
    """Replace file:/// image paths and relative paths with base64 data URIs."""
    def replace_img(m):
        alt = m.group(1)
        src = m.group(2)
        # resolve path
        if src.startswith("file:///"):
            img_path = Path(src[7:])
        elif src.startswith("http://") or src.startswith("https://"):
            return m.group(0)  # keep remote URLs
        else:
            img_path = base_dir / src
        uri = encode_image(img_path)
        if uri:
            return f"![{alt}]({uri})"
        return m.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_img, md)

def strip_carousel_syntax(md: str) -> str:
    """Convert carousel blocks to plain image list."""
    def replace_carousel(m):
        inner = m.group(1)
        # Remove <!-- slide --> separators
        slides = re.split(r"<!--\s*slide\s*-->", inner)
        return "\n\n".join(s.strip() for s in slides)

    return re.sub(r"````carousel\s*(.*?)````", replace_carousel, md, flags=re.DOTALL)

def make_skill_page(skill_id: str, meta: dict, skill_md: str, examples: list[dict], out_path: Path):
    cats = meta.get("category", [])
    primary_cat = cats[0] if cats else "materials"
    colors = CAT_COLORS.get(primary_cat, CAT_COLORS["materials"])
    cat_label = primary_cat.replace("-", " ").title()

    # Extract title server-side from first H1 (never show 'Loading...')
    title_match = re.search(r'^#\s+(.+)$', skill_md, re.MULTILINE)
    page_title = title_match.group(1).strip() if title_match else meta.get('name', skill_id)
    # Remove the H1 from body markdown since we display it in the hero
    body_md_no_h1 = re.sub(r'^#\s+.+\n?', '', skill_md, count=1, flags=re.MULTILINE).lstrip()

    # Build examples HTML
    examples_html = ""
    if examples:
        tabs_html = ""
        panels_html = ""
        for i, ex in enumerate(examples):
            active = "active" if i == 0 else ""
            ex_md_json = json.dumps(ex.get("md", ""))
            tabs_html += f'<button class="ex-tab {active}" onclick="switchTab(this, \'ex-{i}\')">{ex["title"]}</button>\n'
            panels_html += f'<div class="ex-panel {active}" id="ex-{i}"><div class="rendered-md" data-md=\'{ex_md_json}\'></div></div>\n'

        examples_html = f"""
<section class="skill-section">
  <h2 class="section-heading">Examples</h2>
  <div class="ex-tabs">{tabs_html}</div>
  <div class="ex-panels">{panels_html}</div>
</section>
"""

    # Use body_md_no_h1 for the JS content (title already in hero)
    skill_md_for_js = body_md_no_h1

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{meta['name']} — AtomisticSkills</title>
  <meta name="description" content="{meta['description']}"/>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"/>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" defer></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" defer></script>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    :root{{
      --bg:#f8f9fc; --bg2:#ffffff; --border:#e2e8f0;
      --text:#1b1464; --muted:#64748b; --accent:#816cff;
      --accent2:#5b3de8; --cat-bg:{colors['bg']}; --cat-border:{colors['border']};
      --cat-text:{colors['text']}; --cat-dot:{colors['dot']};
    }}
    html{{scroll-behavior:smooth}}
    body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);line-height:1.6;min-height:100vh}}

    /* NAV */
    nav{{position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;
      padding:0 2rem;height:56px;background:rgba(255,255,255,0.92);
      backdrop-filter:blur(12px);border-bottom:1px solid var(--border)}}
    .nav-logo{{display:flex;align-items:center;gap:10px;text-decoration:none}}
    .nav-logo img{{height:24px}}
    .nav-logo span{{font-weight:700;font-size:1rem;color:var(--text)}}
    .nav-back{{display:flex;align-items:center;gap:6px;color:var(--muted);text-decoration:none;font-size:0.88rem;font-weight:500;transition:color .2s}}
    .nav-back:hover{{color:var(--accent)}}
    .nav-right{{display:flex;align-items:center;gap:1rem}}
    .gh-link{{font-size:0.84rem;font-weight:600;color:var(--accent);text-decoration:none;
      border:1px solid var(--accent);border-radius:8px;padding:6px 14px;transition:all .2s}}
    .gh-link:hover{{background:var(--accent);color:#fff}}

    /* HERO */
    .skill-hero{{background:var(--bg2);border-bottom:1px solid var(--border);padding:3rem 2rem 2.5rem}}
    .skill-hero-inner{{max-width:900px;margin:0 auto}}
    .skill-cat-badge{{display:inline-flex;align-items:center;gap:6px;
      background:var(--cat-bg);border:1px solid var(--cat-border);
      color:var(--cat-text);border-radius:99px;padding:4px 14px;
      font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:1rem}}
    .cat-dot{{width:6px;height:6px;border-radius:50%;background:var(--cat-dot)}}
    .skill-id{{font-family:'JetBrains Mono',monospace;font-size:0.85rem;color:var(--muted);margin-bottom:0.75rem}}
    h1{{font-size:clamp(1.8rem,4vw,2.8rem);font-weight:800;line-height:1.15;color:var(--text);margin-bottom:0.75rem}}
    .skill-desc{{font-size:1.05rem;color:var(--muted);max-width:720px;line-height:1.6}}

    /* CONTENT */
    .skill-content{{max-width:900px;margin:0 auto;padding:2.5rem 2rem}}
    .skill-section{{margin-bottom:3rem}}
    .section-heading{{font-size:1.35rem;font-weight:700;color:var(--text);
      padding-bottom:0.75rem;border-bottom:2px solid var(--border);margin-bottom:1.5rem}}

    /* Rendered Markdown */
    .rendered-md h1,.rendered-md h2{{font-size:1.2rem;font-weight:700;margin:1.5rem 0 0.75rem;color:var(--text)}}
    .rendered-md h3{{font-size:1.05rem;font-weight:700;margin:1.25rem 0 0.5rem;color:var(--text)}}
    .rendered-md h4{{font-size:0.95rem;font-weight:700;margin:1rem 0 0.4rem;color:var(--muted)}}
    .rendered-md p{{margin:0.6rem 0;color:#374151;line-height:1.7}}
    .rendered-md ul,.rendered-md ol{{padding-left:1.5rem;margin:0.6rem 0}}
    .rendered-md li{{margin:0.3rem 0;color:#374151}}
    .rendered-md code{{font-family:'JetBrains Mono',monospace;font-size:0.83rem;
      background:#f1f5f9;border:1px solid #e2e8f0;border-radius:5px;padding:1px 6px;color:#be185d}}
    .rendered-md pre{{background:#1e293b;border-radius:10px;padding:1.25rem 1.5rem;overflow-x:auto;margin:1rem 0}}
    .rendered-md pre code{{background:none;border:none;color:#e2e8f0;padding:0;font-size:0.83rem;line-height:1.7}}
    .rendered-md blockquote{{border-left:3px solid var(--accent);background:#f8f9ff;
      padding:0.75rem 1rem;border-radius:0 8px 8px 0;margin:1rem 0}}
    .rendered-md blockquote p{{color:var(--muted)}}
    .rendered-md img{{max-width:100%;border-radius:10px;border:1px solid var(--border);margin:1rem auto;display:block;
      box-shadow:0 4px 24px rgba(0,0,0,0.08)}}
    .rendered-md table{{width:100%;border-collapse:collapse;margin:1rem 0;font-size:0.88rem}}
    .rendered-md th{{background:#f8fafc;padding:10px 14px;text-align:left;font-weight:600;border:1px solid var(--border)}}
    .rendered-md td{{padding:9px 14px;border:1px solid var(--border)}}
    .rendered-md tr:nth-child(even) td{{background:#fafbfc}}
    .rendered-md a{{color:var(--accent);text-decoration:none}}
    .rendered-md a:hover{{text-decoration:underline}}
    .rendered-md hr{{border:none;border-top:1px solid var(--border);margin:1.5rem 0}}

    /* Examples tabs */
    .ex-tabs{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1.25rem}}
    .ex-tab{{padding:8px 18px;border-radius:8px;border:1px solid var(--border);
      background:var(--bg2);font-size:0.85rem;font-weight:600;cursor:pointer;
      color:var(--muted);font-family:'JetBrains Mono',monospace;transition:all .2s}}
    .ex-tab:hover,.ex-tab.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
    .ex-panel{{display:none}}
    .ex-panel.active{{display:block}}

    /* Back button floating */
    .breadcrumb{{font-size:0.82rem;color:var(--muted);margin-bottom:1rem}}
    .breadcrumb a{{color:var(--accent);text-decoration:none}}
    .breadcrumb a:hover{{text-decoration:underline}}

    /* tip/note boxes */
    .rendered-md .admonition{{border-radius:8px;padding:0.75rem 1rem;margin:1rem 0;border-left:3px solid}}
    .rendered-md .admonition-tip{{background:#f0fdf4;border-color:#22c55e}}
    .rendered-md .admonition-note{{background:#eff6ff;border-color:#3b82f6}}
    .rendered-md .admonition-important{{background:#faf5ff;border-color:#a855f7}}
    .rendered-md .admonition-warning{{background:#fffbeb;border-color:#f59e0b}}
    .rendered-md .admonition-caution{{background:#fef2f2;border-color:#ef4444}}

    footer{{border-top:1px solid var(--border);padding:2rem;text-align:center;color:var(--muted);font-size:0.85rem}}
  </style>
</head>
<body>

<nav>
  <a class="nav-logo" href="../index.html">
    <img src="../../logo/atomisticskills_logo.svg" alt="AtomisticSkills"/>
  </a>
  <a class="nav-back" href="../index.html">← Back to Skills</a>
  <div class="nav-right">
    <a class="gh-link" href="https://github.com/bowen-bd/AtomisticSkills/tree/main/.agents/skills/{skill_id}/SKILL.md" target="_blank">View on GitHub</a>
  </div>
</nav>

<div class="skill-hero">
  <div class="skill-hero-inner">
    <div class="breadcrumb"><a href="../index.html">Skills</a> / {skill_id}</div>
    <div class="skill-cat-badge"><span class="cat-dot"></span>{cat_label}</div>
    <div class="skill-id">{skill_id}</div>
    <h1 id="skill-title">{page_title}</h1>
    <p class="skill-desc">{meta['description']}</p>
  </div>
</div>

<div class="skill-content">
  <section class="skill-section">
    <h2 class="section-heading">Skill Instructions</h2>
    <div class="rendered-md" id="skill-body"></div>
  </section>
  {examples_html}
</div>

<footer>
  <p>AtomisticSkills — Open-sourced AI research infrastructure &nbsp;·&nbsp;
    <a href="https://github.com/bowen-bd/AtomisticSkills" target="_blank">GitHub</a>
  </p>
</footer>

<script>
// Global tab switcher — must be defined immediately for inline onclicks
window.switchTab = function(btn, targetId) {{
  const tabs = btn.parentElement.querySelectorAll('.ex-tab');
  const panels = btn.parentElement.nextElementSibling.querySelectorAll('.ex-panel');
  tabs.forEach(t => t.classList.remove('active'));
  panels.forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  const panel = document.getElementById(targetId);
  if (panel) {{
    panel.classList.add('active');
    if (typeof renderMathInElement !== 'undefined') renderMathInElement(panel, KATEX_OPTS);
  }}
}};

document.addEventListener('DOMContentLoaded', function() {{
  const SKILL_MD = {json.dumps(skill_md_for_js)};
  marked.setOptions({{ breaks: true, gfm: true }});

  const KATEX_OPTS = {{
    delimiters: [
      {{left: '$$', right: '$$', display: true}},
      {{left: '$', right: '$', display: false}},
      {{left: '\\(', right: '\\)', display: false}},
      {{left: '\\[', right: '\\]', display: true}}
    ],
    throwOnError: false
  }};

  function renderMd(md, container) {{
    if (!md) return;
    const mathBlocks = [];
    const placeholder = (i) => `MATHPLACEHOLDER${{i}}X`;
    
    // Protect block math $$...$$
    let safe = md.replace(/\\$\\$([\\s\\S]+?)\\$\\$/g, (_, m) => {{
      mathBlocks.push(`$$${{m}}$$`);
      return placeholder(mathBlocks.length - 1);
    }});
    // Protect inline math $...$
    safe = safe.replace(/(?<!\\$)\\$([^\\$]+?)\\$(?!\\$)/g, (_, m) => {{
      mathBlocks.push(`$${{m}}$`);
      return placeholder(mathBlocks.length - 1);
    }});

    let html = marked.parse(safe);
    mathBlocks.forEach((tok, i) => {{
      html = html.replace(new RegExp(placeholder(i), 'g'), tok);
    }});

    html = html
      .replace(/<blockquote>\\s*<p>\\[!TIP\\]/gi,  '<blockquote class="admonition admonition-tip"><p>💡 <strong>Tip</strong>')
      .replace(/<blockquote>\\s*<p>\\[!NOTE\\]/gi, '<blockquote class="admonition admonition-note"><p>📝 <strong>Note</strong>')
      .replace(/<blockquote>\\s*<p>\\[!IMPORTANT\\]/gi, '<blockquote class="admonition admonition-important"><p>⚡ <strong>Important</strong>')
      .replace(/<blockquote>\\s*<p>\\[!WARNING\\]/gi, '<blockquote class="admonition admonition-warning"><p>⚠️ <strong>Warning</strong>')
      .replace(/<blockquote>\\s*<p>\\[!CAUTION\\]/gi, '<blockquote class="admonition admonition-caution"><p>🚨 <strong>Caution</strong>');

    container.innerHTML = html;
    if (typeof renderMathInElement !== 'undefined') renderMathInElement(container, KATEX_OPTS);
  }}

  // Render skill body
  renderMd(SKILL_MD, document.getElementById('skill-body'));

  // Render example panels
  document.querySelectorAll('.ex-panel .rendered-md').forEach(el => {{
    const raw = el.dataset.md;
    if (raw) {{
      try {{
        const mdData = JSON.parse(raw);
        renderMd(mdData, el);
      }} catch(e) {{
        renderMd(raw, el);
      }}
    }}
  }});
}});
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def make_generic_page(doc_id: str, meta: dict, body_md: str, out_path: Path):
    """Template for generic documentation pages (no category, no examples)."""
    # Extract title from H1
    title_match = re.search(r'^#\s+(.+)$', body_md, re.MULTILINE)
    page_title = title_match.group(1).strip() if title_match else meta.get('name', doc_id.replace('_', ' ').title())
    body_md_no_h1 = re.sub(r'^#\s+.+\n?', '', body_md, count=1, flags=re.MULTILINE).lstrip()

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{page_title} — AtomisticSkills</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"/>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" defer></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" defer></script>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    :root{{
      --bg:#f8f9fc; --bg2:#ffffff; --border:#e2e8f0;
      --text:#1b1464; --muted:#64748b; --accent:#816cff;
      --accent2:#5b3de8;
    }}
    html{{scroll-behavior:smooth}}
    body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);line-height:1.6;min-height:100vh}}

    /* NAV */
    nav{{position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;
      padding:0 2rem;height:56px;background:rgba(255,255,255,0.92);
      backdrop-filter:blur(12px);border-bottom:1px solid var(--border)}}
    .nav-logo{{display:flex;align-items:center;gap:10px;text-decoration:none}}
    .nav-logo img{{height:24px}}
    .nav-back{{display:flex;align-items:center;gap:6px;color:var(--muted);text-decoration:none;font-size:0.88rem;font-weight:500;transition:color .2s}}
    .nav-back:hover{{color:var(--accent)}}

    /* HERO */
    .doc-hero{{background:var(--bg2);border-bottom:1px solid var(--border);padding:4rem 2rem 3rem}}
    .doc-hero-inner{{max-width:850px;margin:0 auto}}
    .doc-breadcrumb{{font-size:0.82rem;color:var(--muted);margin-bottom:1rem}}
    .doc-breadcrumb a{{color:var(--accent);text-decoration:none}}
    h1{{font-size:clamp(2rem,5vw,3.2rem);font-weight:800;line-height:1.1;color:var(--text);margin:0}}

    /* CONTENT */
    .doc-content{{max-width:850px;margin:0 auto;padding:3rem 2rem}}
    .rendered-md h2{{font-size:1.6rem;font-weight:700;margin:2rem 0 1rem;color:var(--text);border-bottom:2px solid var(--border);padding-bottom:0.5rem}}
    .rendered-md h3{{font-size:1.25rem;font-weight:700;margin:1.5rem 0 0.75rem;color:var(--text)}}
    .rendered-md p{{margin:0.8rem 0;color:#374151;line-height:1.75;font-size:1.05rem}}
    .rendered-md ul,.rendered-md ol{{padding-left:1.5rem;margin:1rem 0}}
    .rendered-md li{{margin:0.5rem 0;color:#374151}}
    .rendered-md code{{font-family:'JetBrains Mono',monospace;font-size:0.85rem;
      background:#f1f5f9;border:1px solid #e2e8f0;border-radius:5px;padding:2px 6px;color:#be185d}}
    .rendered-md pre{{background:#1e293b;border-radius:10px;padding:1.5rem;overflow-x:auto;margin:1.5rem 0}}
    .rendered-md pre code{{background:none;border:none;color:#e2e8f0;padding:0;font-size:0.85rem;line-height:1.7}}
    .rendered-md blockquote{{border-left:4px solid var(--accent);background:#f8f9ff;
      padding:1rem 1.5rem;border-radius:0 10px 10px 0;margin:1.5rem 0}}
    .rendered-md img{{max-width:100%;border-radius:12px;border:1px solid var(--border);margin:2rem auto;display:block;box-shadow:0 8px 30px rgba(0,0,0,0.1)}}
    .rendered-md a{{color:var(--accent);text-decoration:none;font-weight:500}}
    .rendered-md a:hover{{text-decoration:underline}}

    .admonition{{border-radius:8px;padding:1rem;margin:1.5rem 0;border-left:4px solid}}
    .admonition-tip{{background:#f0fdf4;border-color:#22c55e}}
    .admonition-note{{background:#eff6ff;border-color:#3b82f6}}
    .admonition-important{{background:#faf5ff;border-color:#a855f7}}
    .admonition-warning{{background:#fffbeb;border-color:#f59e0b}}
    .admonition-caution{{background:#fef2f2;border-color:#ef4444}}

    footer{{border-top:1px solid var(--border);padding:3rem 2rem;text-align:center;color:var(--muted);font-size:0.9rem}}
  </style>
</head>
<body>
<nav>
  <a class="nav-logo" href="index.html"><img src="../logo/atomisticskills_logo.svg" alt="AtomisticSkills"/></a>
  <a class="nav-back" href="index.html">← Back Home</a>
</nav>
<div class="doc-hero">
  <div class="doc-hero-inner">
    <div class="doc-breadcrumb"><a href="index.html">Home</a> / Documentation</div>
    <h1>{page_title}</h1>
  </div>
</div>
<div class="doc-content">
  <div class="rendered-md" id="doc-body"></div>
</div>
<footer><p>AtomisticSkills — Open-sourced AI research infrastructure</p></footer>
<script>
document.addEventListener('DOMContentLoaded', function() {{
  const DOC_MD = {json.dumps(body_md_no_h1)};
  marked.setOptions({{ breaks: true, gfm: true }});
  const KATEX_OPTS = {{ delimiters: [
    {{left: '$$', right: '$$', display: true}},
    {{left: '$', right: '$', display: false}},
    {{left: '\\\\(', right: '\\\\)', display: false}},
    {{left: '\\\\[', right: '\\\\]', display: true}}
  ], throwOnError: false }};

    function renderMd(md, container) {{
    if (!md) return;
    const mathBlocks = [];
    const placeholder = (i) => `MATHPLACEHOLDER${{i}}X`;
    
    // Protect block math $$...$$
    let safe = md.replace(/\\$\\$([\\s\\S]+?)\\$\\$/g, (_, m) => {{
      mathBlocks.push(`$$${{m}}$$`);
      return placeholder(mathBlocks.length - 1);
    }});
    // Protect inline math $...$
    safe = safe.replace(/(?<!\\$)\\$([^\\$]+?)\\$(?!\\$)/g, (_, m) => {{
      mathBlocks.push(`$${{m}}$`);
      return placeholder(mathBlocks.length - 1);
    }});

    let html = marked.parse(safe);
    mathBlocks.forEach((tok, i) => {{
      html = html.replace(new RegExp(placeholder(i), 'g'), tok);
    }});

    html = html
      .replace(/<blockquote>\\s*<p>\\[!TIP\\]/gi,  '<blockquote class="admonition admonition-tip"><p>💡 <strong>Tip</strong>')
      .replace(/<blockquote>\\s*<p>\\[!NOTE\\]/gi, '<blockquote class="admonition admonition-note"><p>📝 <strong>Note</strong>')
      .replace(/<blockquote>\\s*<p>\\[!IMPORTANT\\]/gi, '<blockquote class="admonition admonition-important"><p>⚡ <strong>Important</strong>')
      .replace(/<blockquote>\\s*<p>\\[!WARNING\\]/gi, '<blockquote class="admonition admonition-warning"><p>⚠️ <strong>Warning</strong>')
      .replace(/<blockquote>\\s*<p>\\[!CAUTION\\]/gi, '<blockquote class="admonition admonition-caution"><p>🚨 <strong>Caution</strong>');

    container.innerHTML = html;
    if (typeof renderMathInElement !== 'undefined') renderMathInElement(container, KATEX_OPTS);
  }}
  renderMd(DOC_MD, document.getElementById('doc-body'));
}});
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")



def render_example_readme(readme_path: Path) -> str:
    """Render example README as HTML fragment using marked.js (returns md string for injection)."""
    content = read_file(readme_path)
    base_dir = readme_path.parent
    # Process images inline
    content = strip_carousel_syntax(content)
    content = process_markdown_images(content, base_dir)
    return content


def build_generic_docs():
    """Build generic documentation pages from root docs folder."""
    print("\nBuilding generic documentation pages...")
    for md_file in DOCS_SRC_DIR.glob("*.md"):
        doc_id = md_file.stem
        # Skip technical system files or non-site docs
        if doc_id in ["README", "CODE_OF_CONDUCT", "CONTRIBUTING"]:
            continue
        
        raw = read_file(md_file)
        # generic docs might not have fm, but let's try
        meta, body_md = parse_frontmatter(raw)
        
        # Process images (relative to docs/ folder)
        body_md = process_markdown_images(body_md, DOCS_SRC_DIR)
        
        out_path = SITE_DIR / f"{doc_id}.html"
        make_generic_page(doc_id, meta, body_md, out_path)
        print(f"  Built: {doc_id}.html")


def build_skills():
    """Build skill-specific pages."""
    print("\nBuilding skill pages...")
    skill_index = []
    for skill_dir in sorted(SKILLS_SRC_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_id = skill_dir.name
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            continue

        raw = read_file(skill_md_path)
        meta, body_md = parse_frontmatter(raw)
        if not meta["name"]:
            meta["name"] = skill_id

        # Process images in skill body
        body_md = strip_carousel_syntax(body_md)
        body_md = process_markdown_images(body_md, skill_dir)

        # Collect examples
        examples = []
        ex_dir = skill_dir / "examples"
        if ex_dir.is_dir():
            # Check for root-level README
            root_readme = ex_dir / "README.md"
            if root_readme.exists():
                md = render_example_readme(root_readme)
                # Use data-md attribute so the main script can render it uniformly
                examples.append({"title": "Example", "md": md})

            for sub in sorted(ex_dir.iterdir()):
                if sub.is_dir():
                    readme = sub / "README.md"
                    if readme.exists():
                        md = render_example_readme(readme)
                        examples.append({
                            "title": sub.name.replace("_", " ").replace("-", " "),
                            "md": md
                        })

        out_path = SKILLS_OUT_DIR / f"{skill_id}.html"
        make_skill_page(skill_id, meta, body_md, examples, out_path)
        print(f"  Built: skills/{skill_id}.html  ({len(examples)} examples)")

        cats = meta.get("category", ["materials"])
        skill_index.append({
            "id": skill_id,
            "name": meta["name"],
            "description": meta["description"],
            "category": cats,
            "has_examples": len(examples) > 0,
            "num_examples": len(examples),
        })

    # Write updated skills index
    index_out = SITE_DIR / "skills_index.js"
    json_data = json.dumps(skill_index, indent=2, ensure_ascii=False)
    index_out.write_text(f"window.SKILLS_DATA = {json_data};")
    print(f"\n✅ Index written to site/skills_index.js")


def build_all():
    SKILLS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 0. Asset copying removed to avoid duplication. HTML will reference ../logo directly.

    # 1. Build skills
    build_skills()
    
    # 2. Build generic docs
    build_generic_docs()
    
    print("\n✨ Build complete!")


if __name__ == "__main__":
    build_all()
