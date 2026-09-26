/**
 * Bundled brand/illustration assets for the native report/deck generator (§H).
 *
 * Ports `logic/gen/images/` + `logic/gen/svg/` into the Studio bundle.
 * All exports are Vite asset URLs; use `assetToDataUrl` to obtain base64
 * payloads for PPTX embedding or file export.
 */

// ── Institutional logos (PNG) ────────────────────────────────────────────────
import fctAHorizBranco from "./images/2022_FCT_Logo_A_horizontal_branco.png";
import fctAHorizPreto from "./images/2022_FCT_Logo_A_horizontal_preto.png";
import fctBHorizBranco from "./images/2022_FCT_Logo_B_horizontal_branco.png";
import fctBHorizPreto from "./images/2022_FCT_Logo_B_horizontal_preto.png";
import fctCVertBranco from "./images/2022_FCT_Logo_C_vertical_branco.png";
import fctCVertPreto from "./images/2022_FCT_Logo_C_vertical_preto.png";
import inescid01A from "./images/INESC-ID-new-logo_version-01A.png";
import inescid01B from "./images/INESC-ID-new-logo_version-01B.png";
import inescid01C from "./images/INESC-ID-new-logo_version-01C.png";
import inescid02A from "./images/INESC-ID-new-logo_version-02A.png";
import inescid02B from "./images/INESC-ID-new-logo_version-02B.png";
import inescid02C from "./images/INESC-ID-new-logo_version-02C.png";
import inescid04A from "./images/INESC-ID-new-logo_version-04A.png";
import inescid04B from "./images/INESC-ID-new-logo_version-04B.png";
import inescid05A from "./images/INESC-ID-new-logo_version-05A.png";
import inescid05B from "./images/INESC-ID-new-logo_version-05B.png";
import cegistNegative from "./images/logo_CEGIST_negative.png";
import cegistOriginal from "./images/logo_CEGIST_original.png";
import gaipsOriginal from "./images/logo_GAIPS_original.webp";
import gaipsWhite from "./images/logo_GAIPS_white.webp";
import gaipsYellow from "./images/logo_GAIPS_yellow.webp";
import conf2026 from "./images/logo_OPTIMIZATION_CONF2026_transp.png";

// ── Illustrations & icons ────────────────────────────────────────────────────
import vrppIllustrationSource from "./images/vrpp_illustration_source.png";
import wasteBinIcon from "./images/waste_bin_icon.png";
import wasteTruckIcon from "./images/waste_truck_icon.png";

// ── Técnico identity (SVG) ───────────────────────────────────────────────────
import tecnicoPrincipalBranco from "./svg/Tecnico_ID_Principal_RBG-BRANCO.svg";
import tecnicoPrincipalCor from "./svg/Tecnico_ID_Principal_RBG-COR.svg";
import tecnicoPrincipalPreto from "./svg/Tecnico_ID_Principal_RBG-PRETO.svg";
import tecnicoVerticalCor from "./svg/Tecnico_ID_Vertical_RBG-COR.svg";
import tecnicoVerticalBranco from "./svg/Tecnico_ID_Vertical_RBG_BRANCO.svg";
import tecnicoVerticalPreto from "./svg/Tecnico_ID_Vertical_RBG_PRETO.svg";

export const GEN_IMAGES = {
  fct_a_horizontal_branco: fctAHorizBranco,
  fct_a_horizontal_preto: fctAHorizPreto,
  fct_b_horizontal_branco: fctBHorizBranco,
  fct_b_horizontal_preto: fctBHorizPreto,
  fct_c_vertical_branco: fctCVertBranco,
  fct_c_vertical_preto: fctCVertPreto,
  inescid_01a: inescid01A,
  inescid_01b: inescid01B,
  inescid_01c: inescid01C,
  inescid_02a: inescid02A,
  inescid_02b: inescid02B,
  inescid_02c: inescid02C,
  inescid_04a: inescid04A,
  inescid_04b: inescid04B,
  inescid_05a: inescid05A,
  inescid_05b: inescid05B,
  cegist_negative: cegistNegative,
  cegist_original: cegistOriginal,
  gaips_original: gaipsOriginal,
  gaips_white: gaipsWhite,
  gaips_yellow: gaipsYellow,
  conference_2026: conf2026,
  vrpp_illustration_source: vrppIllustrationSource,
  waste_bin_icon: wasteBinIcon,
  waste_truck_icon: wasteTruckIcon,
} as const;

export const TECNICO_SVGS = {
  principal_branco: tecnicoPrincipalBranco,
  principal_cor: tecnicoPrincipalCor,
  principal_preto: tecnicoPrincipalPreto,
  vertical_cor: tecnicoVerticalCor,
  vertical_branco: tecnicoVerticalBranco,
  vertical_preto: tecnicoVerticalPreto,
} as const;

/** Fetch a bundled asset URL and return it as a base64 data URL. */
export async function assetToDataUrl(url: string): Promise<string> {
  const blob = await (await fetch(url)).blob();
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

/** Natural pixel size of an image URL/data URL (for aspect-fit placement). */
export async function imageSize(url: string): Promise<{ w: number; h: number }> {
  return await new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => reject(new Error(`failed to load image: ${url.slice(0, 60)}`));
    img.src = url;
  });
}
