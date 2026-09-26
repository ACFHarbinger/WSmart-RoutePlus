"""
Shared helpers for the report-generation scripts (dataset / simulation analysis).

Centralises:
  - loading of the JSON configurations under logic/gen/json/
  - loading of matplotlib style sheets under logic/gen/style/
  - loading of JS snippets under logic/gen/js/
  - Jinja2 rendering of the markdown templates under logic/gen/jinja/
  - markdown post-processing (full-width <figure> images, Figure/Table numbering)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
JSON_DIR = SCRIPT_DIR / "json"
STYLE_DIR = SCRIPT_DIR / "style"
JS_DIR = SCRIPT_DIR / "js"
JINJA_DIR = SCRIPT_DIR / "jinja"

PLACEHOLDER = "<!-- [ANALYSIS: Insert your observations here] -->"


def load_json(name: str) -> dict:
    """Load a JSON configuration from logic/gen/json/."""
    return json.loads((JSON_DIR / name).read_text(encoding="utf-8"))


def load_theme(name: str) -> dict:
    """
    Return the theme dict from themes.json, with the resolved .mplstyle path
    added under the "mplstyle_path" key.
    """
    themes = load_json("themes.json")
    if name not in themes:
        raise SystemExit(f"Unknown theme '{name}' (available: {', '.join(themes)})")
    theme = dict(themes[name])
    theme["name"] = name
    theme["mplstyle_path"] = str(STYLE_DIR / theme["mplstyle"])
    return theme


def apply_theme(theme: dict) -> None:
    """Apply the theme's matplotlib style sheet to the global rcParams."""
    plt.style.use(theme["mplstyle_path"])


def load_js(name: str, **subs: str) -> str:
    """Load a JS snippet from logic/gen/js/, substituting __KEY__ markers."""
    js = (JS_DIR / name).read_text(encoding="utf-8")
    for key, val in subs.items():
        js = js.replace(f"__{key.upper()}__", val)
    return js


def render_template(name: str, **context) -> str:
    """
    Render a Jinja2 template from logic/gen/jinja/.

    Templates use non-default delimiters (<% %>, << >>, <# #>) instead of the
    Jinja defaults, since the markdown they emit embeds LaTeX table snippets
    whose own {} / {{ }} syntax would otherwise be misparsed as Jinja.
    """
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(JINJA_DIR)),
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
    )
    return env.get_template(name).render(**context)


def savefig(fig: plt.Figure, path: Path) -> None:
    """Save a figure honouring the active style's facecolor and close it."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=plt.rcParams["figure.facecolor"])
    plt.close(fig)
    print(f"  Saved: {path.name}")


def figureize_images(md: str) -> str:
    """Convert all ![alt](path) markdown images to full-width HTML <figure> blocks."""
    return re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda m: (
            f'<figure style="display:block;width:100%;margin:0.8em 0;padding:0;">'
            f'<img src="{m.group(2)}" alt="{m.group(1)}" width="100%"'
            f' style="width:100% !important;max-width:100% !important;'
            f'height:auto !important;display:block !important;margin:0;" />'
            f"</figure>"
        ),
        md,
    )


def apply_figure_table_numbers(md: str) -> str:
    """Add sequential **Figure N** / **Table N** labels to generated markdown."""
    fig_n = [0]
    tab_n = [0]

    def _fig_num(m):
        fig_n[0] += 1
        return f"{m.group(1)}\n\n**Figure {fig_n[0]}:** {m.group(2)}\n"

    md = re.sub(
        r"(<figure\b[^>]*>.*?</figure>)\n+(\*[^*\n][^\n]*\*)\n",
        _fig_num,
        md,
        flags=re.DOTALL,
    )

    def _tab_num(m):
        tab_n[0] += 1
        return f"**Table {tab_n[0]}:** *{m.group(1).strip()}*"

    md = re.sub(r"_TABCAP_: ([^\n]+)", _tab_num, md)
    return md


def finalize_markdown(md: str) -> str:
    """Apply the full markdown post-processing pipeline."""
    return apply_figure_table_numbers(figureize_images(md))
