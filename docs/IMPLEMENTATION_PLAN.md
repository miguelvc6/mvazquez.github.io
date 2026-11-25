# Implementation plan

## Baseline

The current custom-domain site is a standalone HTML portfolio published from GitHub Pages. Custom CSS and JavaScript control the navigation, theme and project carousel. A Python Markdown and Jinja pipeline writes the blog index, individual articles and search index under `blog/output/`. The hero embeddings demo is a standalone Plotly page.

## Design and structure

The editorial prototype supplies the ivory paper, ink, terracotta, serif typography, portrait composition and publication rows. The Research Lab map becomes a keyboard-operable diagram with three research questions and a textual equivalent. The Studio prototype informs course and project layouts. The homepage introduces Miguel’s current work, with dedicated Research, Publications, Teaching, Projects, Writing and About pages and direct access to the CV.

## Build and migration

Use static Astro, TypeScript, validated content records, custom CSS and progressive enhancement. Keep Markdown as the article source and render legacy article URLs directly so bookmarks continue to resolve. Retain the old blog index and search artifact URLs, public media paths, downloadable article resources, demo and CV path. Port homepage section anchors where practical. Keep prototypes, LaTeX and development files outside the deploy artifact.

Use the academic CV as the source for professional details. Confirm additional links against public primary sources and omit unconfirmed assets. Document Google Analytics as an opt-in decision. Validate every branch with CI and gate deployment on a manual run from main.

## Acceptance

Verify the complete production build, page and resource routing, old bookmarks, citations, themes, responsive layouts, research interactions and writing. Inspect screenshots across representative pages. Keep production untouched pending explicit approval.
