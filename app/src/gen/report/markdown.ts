/**
 * Markdown post-processing pipeline (§H.5) — native port of
 * `logic/gen/report_utils.py`: full-width <figure> wrapping and sequential
 * Figure/Table numbering.
 */

export const PLACEHOLDER = "<!-- [ANALYSIS: Insert your observations here] -->";

/** Convert all ![alt](path) markdown images to full-width HTML <figure> blocks. */
export function figureizeImages(md: string): string {
  return md.replace(
    /!\[([^\]]*)\]\(([^)]+)\)/g,
    (_m, alt: string, path: string) =>
      `<figure style="display:block;width:100%;margin:0.8em 0;padding:0;">` +
      `<img src="${path}" alt="${alt}" width="100%"` +
      ` style="width:100% !important;max-width:100% !important;` +
      `height:auto !important;display:block !important;margin:0;" />` +
      `</figure>`
  );
}

/** Add sequential **Figure N** / **Table N** labels to generated markdown. */
export function applyFigureTableNumbers(md: string): string {
  let figN = 0;
  let tabN = 0;
  md = md.replace(
    /(<figure\b[^>]*>[\s\S]*?<\/figure>)\n+(\*[^*\n][^\n]*\*)\n/g,
    (_m, figure: string, caption: string) => {
      figN += 1;
      return `${figure}\n\n**Figure ${figN}:** ${caption}\n`;
    }
  );
  md = md.replace(/_TABCAP_: ([^\n]+)/g, (_m, caption: string) => {
    tabN += 1;
    return `**Table ${tabN}:** *${caption.trim()}*`;
  });
  return md;
}

export function finalizeMarkdown(md: string): string {
  return applyFigureTableNumbers(figureizeImages(md));
}

/** Strip the leading "public/" segment for report-relative asset paths (ports _to_rel). */
export function toRel(path: string): string {
  return path.startsWith("public/") ? path.replace("public/", "") : path;
}
