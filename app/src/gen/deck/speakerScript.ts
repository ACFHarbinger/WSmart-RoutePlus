/**
 * Speaker script DOCX (§H.6) — native port of `gen_speaker_script` from
 * `logic/gen/gen_presentation.py`, replacing docxtpl + the .docx template
 * with a programmatic `docx` document of the same structure.
 */
import {
  AlignmentType,
  Document,
  HeadingLevel,
  Packer,
  Paragraph,
  TextRun,
} from "docx";

export interface SlideScript {
  number: number;
  title: string;
  script: string;
}

export async function buildSpeakerScript(
  deckTitle: string,
  author: string,
  slides: SlideScript[]
): Promise<Blob> {
  const children: Paragraph[] = [
    new Paragraph({
      heading: HeadingLevel.TITLE,
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: deckTitle, bold: true })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: `Speaker script — ${author}`, italics: true, color: "5A6A7A" })],
      spacing: { after: 400 },
    }),
  ];
  for (const slide of slides) {
    children.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun({ text: `Slide ${slide.number} — ${slide.title}`, bold: true })],
        spacing: { before: 300, after: 120 },
      })
    );
    for (const para of slide.script.split("\n\n").filter(Boolean)) {
      children.push(
        new Paragraph({
          children: [new TextRun({ text: para })],
          spacing: { after: 120 },
        })
      );
    }
  }
  const doc = new Document({
    creator: author,
    title: `${deckTitle} — speaker script`,
    sections: [{ children }],
  });
  return await Packer.toBlob(doc);
}
