# Miguel Vázquez’s academic website

The site builds to static HTML and assets for GitHub Pages at [mvazquez.ai](https://mvazquez.ai). Astro and TypeScript provide the templates and content validation. Page interactions use browser-native controls and small scripts.

## Work locally

Use Node.js 24 and npm. The lockfile fixes the dependency graph.

```sh
npm ci
npm run dev
```

Open the local address printed by Astro. To inspect the exact publication artifact and run the checks:

```sh
npm run check
npm run build
npm run test:dist
npm run preview
```

For browser tests, run `npx playwright install chromium` once, then `npm test` after building. Playwright starts the preview when needed. Set `PLAYWRIGHT_CHROME_PATH` to test with an installed Chrome binary instead. Set `ASTRO_TELEMETRY_DISABLED=1` to disable Astro CLI telemetry locally, as CI does.

## Maintain the content

| Update | Source |
| --- | --- |
| Bio, affiliation, CV URL and profiles | `src/data/profile.ts` and the longer biography in `src/pages/about/index.astro` |
| Publication | Add one entry to `src/data/publications.ts`. Set the status, authors, venue and verified resource URLs. The page and BibTeX use the same entry. Set `featured` to surface it on the homepage. Existing IDs are public fragments. |
| Research direction | Edit `src/data/research.ts`. The map and dedicated page use the same three entries. Update its SVG in `src/components/ResearchMap.astro` if the relationships change. |
| Course or supervision | Add a record to `src/data/teaching.ts`. Use `category` and add only confirmed public materials to `links`. |
| Project | Add a record to `src/data/projects.ts`. Include your role and any collaborators. Optional images need intrinsic dimensions and descriptive alternative text. |
| Academic activity | Add a dated entry to `src/data/news.ts`. The homepage sorts newest first. |
| Photographs | Update the image imports in `src/data/profile.ts`. The homepage treatment is in `src/pages/index.astro` and is applied with CSS. Replace `public/images/miguel-vazquez.webp` when updating the social preview portrait. |
| CV | Replace `public/cv/Miguel_Vazquez_CV.pdf` with the current PDF. Keep the public filename for bookmarks and update the internal reference source as needed. |

### Publish an article

Create a Markdown file in `src/content/writing/`. Its filename, without `.md`, becomes the public slug. Provide these fields and use level-two headings for the article body:

```yaml
---
title: "Your article title"
date: "2026-11-25"
summary: "A short description of the article."
tags: ["Research notes"]
draft: true
---
```

Use fenced code with a lowercase language, `$…$` for inline math and `$$…$$` for display math. Put public images and resources in `public/blog/media/<slug>/` and link to their root-relative paths. Use meaningful image descriptions. Change `draft` to `false` when ready. Only published articles generate pages, index entries, sitemap entries and RSS items. Keep draft-only files outside `public/`.

### Maintain existing URLs

All original articles are served as files at `/blog/output/<slug>/content.html`. Astro’s `preserve` build format keeps these file routes and the main sections’ directory URLs. If a slug changes, add an explicit compatibility page in `src/pages/blog/output/` with a canonical link, a redirect and a clickable fallback. The old blog landing page is an example. The search-index compatibility endpoint and the CV retain their old URLs. Update `tests/dist.mjs` when changing compatibility routes.

The project demonstrations are standalone, with the Plotly renderer served through a static Astro endpoint. The Dijkstra and A* page contains a snapshot of the author’s visualizer. Update it from the repository only after checking its assets and running the browser test.

## Deploy after approval

Validation runs on pushes and pull requests. The deployment workflow only runs after a manual dispatch from `main` with the approval checkbox selected.

When the redesign is approved, coordinate the merge with changing **Settings → Pages → Build and deployment → Source** to **GitHub Actions**, so a branch-based publishing source cannot serve the Astro source tree during the switch. Keep the current custom domain and DNS settings. From **Actions**, run **Deploy approved site** on `main` and check the deployed site and legacy URLs. The artifact is `dist/`, which includes the domain file `public/CNAME`. [GitHub Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

### Analytics

The previous site loaded Google Analytics. This build leaves it off unless `PUBLIC_GA_MEASUREMENT_ID` is deliberately set for a production build. The deploy workflow reads this from the matching GitHub Actions repository variable. Analytics sends usage measurements to Google and can collect device and approximate location information depending on settings. Review your privacy notice, configuration and how you obtain visitor consent before enabling it. [Google Analytics data and privacy controls](https://support.google.com/analytics/answer/12017362)

## Source layout

- `src/pages/` and `src/layouts/` contain routes and the shared page template.
- `src/components/`, `src/styles/` and `src/assets/` contain the design system and editable inputs.
- `src/data/` and `src/content/writing/` contain structured records and Markdown.
- `public/` contains directly served assets. Everything inside it ships in the build.
- `docs/` contains design and implementation notes. It stays outside the deployment artifact.
- `tests/` checks the static artifact and browser behavior.
