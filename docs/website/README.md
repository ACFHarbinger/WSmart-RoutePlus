# WSmart Route+ — Public Website

React documentation and product portal for **WSmart Route+**: combinatorial
optimization for waste collection vehicle routing (neural + classical OR).

This site is **separate** from the Tauri desktop Studio in `app/`. The public
site explains architecture, research, and evidence; the Studio operates.

## Stack

- React 19 + TypeScript
- Vite 6
- React Router 7
- Lucide icons
- Custom CSS (operations-research observatory theme)

## Develop

```bash
cd docs/website
npm install
npm run dev
```

## Build

```bash
npm run build
npm run preview
```

Optional base path for GitHub Pages or a subpath deploy:

```bash
SITE_BASE=/WSmart-Route/ npm run build
```

## Routes

| Path           | Purpose                                      |
| -------------- | -------------------------------------------- |
| `/`            | Landing — hero, workbench, solvers, atlas    |
| `/platform`    | Architecture: logic / Studio / evidence      |
| `/research`    | Problem families and research map            |
| `/benchmarks`  | Comparison philosophy and caveats            |
| `/studio`      | Desktop Studio capabilities                  |
| `/docs`        | Links into repository Markdown               |
| `/roadmap`     | Living research / product direction          |

## Content model

Copy and structured lists live in `src/data/site.ts`. Prefer updating that
module over hardcoding repeated claims in components.

## Design notes

- Dark graphite foundation, eco neon (`#3dffa8`) and route cyan (`#4cc9f0`)
- Instrument Serif for editorial emphasis; DM Sans / DM Mono for UI
- Hero uses an animated canvas route graph (`RouteGraph`); atlas is interactive SVG
- Keep heavy Studio deps (Deck.gl, ECharts, etc.) out of this package

## Coordination

Multi-agent notes: `.agent/cache/website_coordination.md` and
`.agent/cache/AGENT_BUS.md` in the repository root.
