"""
Generate the grant holder's final report (Relatório Final do Bolseiro e Orientador) as an editable .docx.

The layout follows the university's "Modelo do Relatório Final" form: the institution logo in the
page header, a centred title and subtitle, the grant holder's full name and grant ID, and the
research report (sections with a heading and paragraphs). The logo is the IST mark (``logic/gen/svg/Tecnico_ID_Principal_RBG-COR.svg``),
rasterised with Inkscape (ImageMagick as a fallback).

The document is written as plain Office Open XML with the standard library, so no extra Python
dependency is needed. Word and LibreOffice open it for editing.

Non-Python content lives in sibling directories, as with the other generators:
  json/grant_report.json   labels, field values, the report text, fonts and the logo path

Idempotent: will not overwrite an existing report unless --force is passed (the report is meant to
be edited after generation).

Usage
-----
    uv run python logic/gen/gen_grant_report.py
    uv run python logic/gen/gen_grant_report.py --config my_report.json \\
        --out docs/private/reports/relatorio_final_bolseiro.docx --force
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "json" / "grant_report.json"
DEFAULT_OUT = ROOT / "docs" / "private" / "reports" / "relatorio_final_bolseiro.docx"

EMU_PER_CM = 360000
LOGO_PX_HEIGHT = 400  # raster height; the document scales it to logo_height_cm

W_NS = (
    'xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
)


# --------------------------------------------------------------------------- #
# Logo                                                                         #
# --------------------------------------------------------------------------- #


def rasterise_svg(svg: Path, png: Path) -> None:
    """Render ``svg`` to ``png`` at LOGO_PX_HEIGHT pixels high with Inkscape or ImageMagick."""
    if shutil.which("inkscape"):
        cmd = [
            "inkscape",
            str(svg),
            "--export-type=png",
            f"--export-filename={png}",
            f"--export-height={LOGO_PX_HEIGHT}",
        ]
    elif shutil.which("magick") or shutil.which("convert"):
        tool = shutil.which("magick") or shutil.which("convert")
        cmd = [str(tool), "-background", "none", "-density", "300", str(svg), "-resize", f"x{LOGO_PX_HEIGHT}", str(png)]
    else:
        raise SystemExit("Rasterising the logo needs Inkscape or ImageMagick on PATH")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not png.is_file():
        raise SystemExit(f"Could not rasterise {svg}:\n{result.stderr}")


def png_size(png: Path) -> tuple:
    """Return (width, height) in pixels from the PNG header."""
    data = png.read_bytes()[:24]
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{png} is not a PNG")
    return struct.unpack(">II", data[16:24])


# --------------------------------------------------------------------------- #
# WordprocessingML                                                             #
# --------------------------------------------------------------------------- #


def run(text: str, font: str, size_pt: float, bold: bool = False) -> str:
    """One text run."""
    props = f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>'
    if bold:
        props += "<w:b/><w:bCs/>"
    half = int(round(size_pt * 2))
    props += f'<w:sz w:val="{half}"/><w:szCs w:val="{half}"/>'
    return f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def paragraph(runs: List[str], align: str = "both", after_pt: float = 12, line: int = 360) -> str:
    """A paragraph; ``align`` is left, center or both (justified); ``line`` is in 240ths of a line."""
    ppr = f'<w:pPr><w:spacing w:after="{int(after_pt * 20)}" w:line="{line}" w:lineRule="auto"/><w:jc w:val="{align}"/></w:pPr>'
    return f"<w:p>{ppr}{''.join(runs)}</w:p>"


def logo_drawing(width_emu: int, height_emu: int) -> str:
    """Inline picture referencing relationship rIdLogo."""
    return (
        '<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{width_emu}" cy="{height_emu}"/><wp:docPr id="1" name="Logo IST"/>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="logo.png"/><pic:cNvPicPr/></pic:nvPicPr>'
        '<pic:blipFill><a:blip r:embed="rIdLogo"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        "</wp:inline></w:drawing></w:r>"
    )


def build_document(cfg: Dict[str, Any]) -> str:
    """The body: title, the labelled fields, and the research report with its sections."""
    font, label_font, size = cfg["font"], cfg["label_font"], float(cfg["font_size_pt"])
    body: List[str] = [
        paragraph([run(cfg["title"], label_font, size + 3)], align="center", after_pt=4),
        paragraph([run(cfg["subtitle"], label_font, size - 2, bold=True)], align="center", after_pt=24),
    ]
    for field in cfg["fields"]:
        body.append(
            paragraph([run(field["label"], label_font, size, bold=True), run(f": {field['value']}", font, size)])
        )
    body.append(paragraph([run(f"{cfg['report_label']}:", label_font, size, bold=True)], after_pt=6))
    for section in cfg["report"]:
        if section.get("heading"):
            body.append(paragraph([run(section["heading"], label_font, size, bold=True)], align="left", after_pt=6))
        for text in section["paragraphs"]:
            body.append(paragraph([run(text, font, size)]))
    sect = (
        '<w:sectPr><w:headerReference w:type="default" r:id="rIdHeader"/>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="2268" w:right="1701" w:bottom="1701" w:left="1701" w:header="709" w:footer="709" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {W_NS}><w:body>{"".join(body)}{sect}</w:body></w:document>'


def build_header(width_emu: int, height_emu: int) -> str:
    """Page header with the centred logo."""
    p = f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>{logo_drawing(width_emu, height_emu)}</w:p>'
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:hdr {W_NS}>{p}</w:hdr>'


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>"""

DOCUMENT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
<Relationship Id="rIdHeader" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>
</Relationships>"""

HEADER_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rIdLogo" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/logo.png"/>
</Relationships>"""


def build_styles(cfg: Dict[str, Any]) -> str:
    """Default paragraph font and language (pt-PT) so new text typed in the editor matches."""
    font, half = cfg["font"], int(round(float(cfg["font_size_pt"]) * 2))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>'
        f'<w:sz w:val="{half}"/><w:szCs w:val="{half}"/><w:lang w:val="pt-PT"/></w:rPr></w:rPrDefault>'
        '<w:pPrDefault><w:pPr><w:spacing w:after="240" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr></w:pPrDefault>'
        "</w:docDefaults>"
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
        "</w:styles>"
    )


def build_core(cfg: Dict[str, Any]) -> str:
    """Document properties (title, creation time)."""
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{escape(cfg['title'])}</dc:title><dc:language>pt-PT</dc:language>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        "</cp:coreProperties>"
    )


def write_report(cfg: Dict[str, Any], out: Path) -> None:
    """Rasterise the logo and write the .docx package to ``out``."""
    logo_svg = (ROOT / cfg["logo"]).resolve()
    if not logo_svg.is_file():
        raise SystemExit(f"Logo not found: {logo_svg}")
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "logo.png"
        rasterise_svg(logo_svg, png)
        w_px, h_px = png_size(png)
        height_emu = int(float(cfg["logo_height_cm"]) * EMU_PER_CM)
        width_emu = int(height_emu * w_px / h_px)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", CONTENT_TYPES)
            z.writestr("_rels/.rels", ROOT_RELS)
            z.writestr("docProps/core.xml", build_core(cfg))
            z.writestr("word/document.xml", build_document(cfg))
            z.writestr("word/_rels/document.xml.rels", DOCUMENT_RELS)
            z.writestr("word/header1.xml", build_header(width_emu, height_emu))
            z.writestr("word/_rels/header1.xml.rels", HEADER_RELS)
            z.writestr("word/styles.xml", build_styles(cfg))
            z.write(png, "word/media/logo.png")


def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="report content (JSON)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output .docx path")
    ap.add_argument("--force", action="store_true", help="overwrite an existing report")
    args = ap.parse_args(argv)
    if args.out.exists() and not args.force:
        print(f"{args.out} exists (it may hold your edits); pass --force to overwrite", file=sys.stderr)
        return 1
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    write_report(cfg, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
