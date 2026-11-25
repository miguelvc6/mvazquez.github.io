# Implementation report

## Site and content

The redesigned site has an editorial homepage and dedicated Research, Publications, Teaching, Projects, Writing, and About pages. Ivory, ink, terracotta, serif display type, and the existing black-and-white portrait set the visual direction. A separate dark palette carries the same hierarchy. The research map uses SVG and native radio controls, with explanatory text for Repair, Verify, and Guide. Its modes describe three research questions.

Astro renders the site to static files. TypeScript and validation schemas structure the profile, publication, course, project, and activity records. Small scripts handle theme preference, copy controls, and the writing search. The search loads the article index when a visitor first enters a query. The production build includes canonical URLs, Open Graph metadata, JSON-LD, a sitemap, robots.txt, and RSS.

The supplied PDF and LaTeX CV determine the academic record and the statuses of the three initial publications. The ESWC publisher record and associated repository confirm the DOI, author list, and available paper and code. The Wikidata Workshop PDF, ESSAI course site, and WU profile corroborate the published workshop item, teaching link, and institutional identity. [ESWC publisher](https://doi.org/10.1007/978-3-032-25156-5_15) · [ESWC resources](https://github.com/Tschoui/kg_repairs) · [Wikidata Workshop paper](https://wikidataworkshop.github.io/2026/papers/paper17.pdf) · [ESSAI course](https://wikidata-essai.ai.wu.ac.at/) · [WU research profile](https://research.wu.ac.at/en/persons/miguel-vazquez/)

## Migration and privacy

- All five articles retain their dates, canonical HTML addresses, and legacy heading fragments. The old list URL redirects to Writing, which includes full-text search, and the old search index remains available. The Markdown preserves the body, figures, citations, and sample resources. The migration adds image descriptions and intrinsic dimensions, adjusts one malformed TeX expression, and normalizes code fence labels. The three Torch Tracer figures serve locally, including the credited Wikimedia diagram.
- The existing CV URL now contains the supplied two-page PDF. The existing photographs and project screenshots remain available at their former asset URLs.
- The Torch Tracer project links to the ray-tracing article. The Hero Embeddings demo keeps its public URL and data. Its pinned Plotly renderer is served locally, and the page adds keyboard-operable rotation and textual search results. The Dijkstra and A* visualizer is a local snapshot of the author’s educational demo, with links to the upstream materials. [Dijkstra and A* repository](https://github.com/miguelvc6/dijkstra_Astar)
- Analytics is off by default. An optional production build variable controls Google Analytics. The README documents the configuration and privacy trade-offs. Prototypes, LaTeX, and development documentation stay out of the published artifact.
- GitHub Actions validates contributions. Publishing is a separate manual workflow on `main` and requires the approval flag. The custom domain remains `mvazquez.ai`.

## Verification

- Chrome layout checks covered 1440, 1280, 768, 390, and 320 pixels in light and dark themes. The interaction suite covers keyboard navigation, theme persistence, research selection with JavaScript disabled, BibTeX and code copy, CV download, writing search, legacy routes, equations, both project demos, and topic navigation.
- Axe scans covered the seven principal pages and all five published articles in both themes against automated WCAG 2.2 AA rules. The interactive plots have separate keyboard and functional checks. Automated checks cover only a subset of accessibility requirements.
- The static test resolves same-site HTML links, fragments, and resource references, and checks the sitemap, feed, compatibility files, and exclusions. A temporary draft fixture was absent from generated HTML, RSS, sitemap, and the search index.
- Screenshots cover the desktop homepage, dark mobile homepage, desktop research and dark publications, tablet teaching, and a mobile article. Visual review led to a spacing adjustment in the hero and a clearer route for the diagram’s feedback connector.
- Lighthouse 13.6 scored the locally served homepage 99 for performance and 100 for accessibility, best practices, and SEO. The simulated mobile run reported 2.1 seconds LCP, 0.005 CLS, and 203 KiB transferred. These are local lab measurements.

## Open items

- The CV lists the TGDK manuscript as submitted. The site uses that status and omits a manuscript download pending a verified public URL. Publication slides, posters, a standalone BilAI-KG resource, and internal course materials also await public URLs.
- LinkedIn returned an automated-request block. The indexed public profile corroborates the name and WU affiliation. One historical article link to an In Silico Chemistry tutorial returned HTTP 503 and remains in the archive. [LinkedIn public profile](https://at.linkedin.com/in/miguel-vazquez123)
- The standalone demos retain their separate interfaces. The Hero Embeddings visualization depends on WebGL, and the Dijkstra replay carries a large embedded trace. Both load only when opened.
- Production verification and the GitHub Pages publishing-source switch belong to the approved release. If the publication or course catalogs grow substantially, revisit navigation within those pages and the size of the writing search index.
