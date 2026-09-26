/**
 * Plain-text equation fallback (§H.4) — ports `_plain_fallback` from
 * `logic/gen/gen_presentation.py`; the symbol table is kept as data.
 */

const FALLBACK_SYMBOLS: [string, string][] = [
  ["\\Rightarrow", " => "],
  ["\\rightarrow", " -> "],
  ["\\geq", " >= "],
  ["\\leq", " <= "],
  ["\\in", " in "],
  ["\\notin", " not in "],
  ["\\times", " x "],
  ["\\cdot", " * "],
  ["\\quad", "   "],
  ["\\qquad", "     "],
];

const FALLBACK_DROP = ["mathbf", "mathrm", "textbf", "text", "mathcal", "left", "right", "dfrac", "frac"];

export function plainFallback(latex: string): string {
  let text = latex;
  for (const [sym, plain] of FALLBACK_SYMBOLS) text = text.split(sym).join(plain);
  text = text.replace(new RegExp(`\\\\(${FALLBACK_DROP.join("|")})\\b`, "g"), "");
  text = text.replace(/\\([a-zA-Z]+)/g, "$1"); // keep the word, drop the backslash
  text = text.replace(/[\\{}$]/g, "").replace(/,/g, " ");
  return text.replace(/\s+/g, " ").trim();
}
