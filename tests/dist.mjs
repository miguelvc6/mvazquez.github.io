import assert from 'node:assert/strict';
import { readdir, readFile } from 'node:fs/promises';
import { resolve, join, relative } from 'node:path';
import { load } from 'cheerio';

const dist = resolve('dist');
const site = 'https://mvazquez.ai';
const legacySlugs = [
  '2024-10-28-welcome-post',
  '2024-11-12-torch-tracing-01',
  '2024-11-17-llm-agents',
  '2024-11-20-agents-ecosystem',
  '2024-11-25-reasoning-and-action-agents',
];
const articlePaths = legacySlugs.map((slug) => `/blog/output/${slug}/content.html`);
const mainPaths = ['/', '/about/', '/projects/', '/publications/', '/research/', '/teaching/', '/writing/'];
const paths = [
  ...mainPaths, ...articlePaths, '/blog/output/blog.html', '/blog/output/search-index.json',
  '/cv/Miguel_Vazquez_CV.pdf', '/projects/3d_hero_embeddings_visualization_js.html',
  '/projects/dijkstra-astar.html', '/robots.txt', '/sitemap-index.xml', '/rss.xml', '/CNAME', '/.nojekyll',
];
const demoPath = '/projects/3d_hero_embeddings_visualization_js.html';

async function entries(dir) {
  const children = await readdir(dir, { withFileTypes: true });
  const files = await Promise.all(children.map((child) => child.isDirectory()
    ? entries(join(dir, child.name))
    : [join(dir, child.name)]));
  return files.flat();
}

const deployedFiles = await entries(dist);
const deployedPaths = new Set(deployedFiles);

function findFile(pathname) {
  const path = resolve(dist, `.${decodeURIComponent(pathname)}`);
  assert.ok(path === dist || path.startsWith(`${dist}/`), pathname);
  // GitHub Pages paths are case-sensitive even when the build runs on macOS.
  return [path, join(path, 'index.html')].find((candidate) => deployedPaths.has(candidate));
}

const failures = [];
const pages = new Map();
async function markup(file) {
  if (!pages.has(file)) pages.set(file, load(await readFile(file, 'utf8')));
  return pages.get(file);
}

for (const path of paths) assert.ok(findFile(path), `Missing build artifact ${path}`);
assert.equal((await readFile(join(dist, 'CNAME'), 'utf8')).trim(), 'mvazquez.ai');
assert.ok((await readFile(join(dist, 'cv/Miguel_Vazquez_CV.pdf'))).subarray(0, 5).equals(Buffer.from('%PDF-')));

assert.ok(!deployedFiles.some((file) => /(?:\.tex|\.md|\.DS_Store)$|\/docs\//.test(relative(dist, file))), 'Development files leaked');
const htmlFiles = deployedFiles.filter((file) => file.endsWith('.html'));
let checked = 0;
for (const file of htmlFiles) {
  const path = `/${relative(dist, file).replace(/(^|\/)index\.html$/, '$1')}`;
  const $ = await markup(file);
  if (path !== demoPath) {
    assert.equal($('h1').length, 1, `Expected one h1 in ${path}`);
    assert.ok($('meta[name="description"]').attr('content'), `Missing description in ${path}`);
    const ids = $('[id]').map((_, el) => $(el).attr('id')).get();
    assert.equal(new Set(ids).size, ids.length, `Duplicate IDs in ${path}`);
  }
  for (const element of $('a[href], img[src], link[href], script[src]')) {
    const address = $(element).attr('href') ?? $(element).attr('src');
    const url = new URL(address, site + path);
    if (url.origin !== site) continue;
    checked++;
    const target = findFile(url.pathname);
    if (!target) {
      failures.push(`${path} → ${address} (missing file)`);
      continue;
    }
    if (url.hash && target.endsWith('.html')) {
      const destination = await markup(target);
      if (!destination('[id]').toArray().some((node) => destination(node).attr('id') === decodeURIComponent(url.hash.slice(1)))) {
        failures.push(`${path} → ${address} (missing ID)`);
      }
    }
  }
}

for (const path of articlePaths) {
  const $ = await markup(findFile(path));
  assert.equal($('link[rel="canonical"]').attr('href'), site + path, path);
  assert.ok($('.article-body').text().trim().length > 100, path);
}
const torch = await markup(findFile(articlePaths[1]));
assert.ok(torch('math').length >= 5 && !torch('.katex-error').length, 'Torch Tracer equations');
assert.ok(torch('pre.astro-code span[style]').length > 5, 'Torch Tracer highlighted code');
const rss = load(await readFile(join(dist, 'rss.xml'), 'utf8'), { xml: true });
const sitemap = load(await readFile(join(dist, 'sitemap-0.xml'), 'utf8'), { xml: true });
for (const path of [...mainPaths, ...articlePaths]) {
  assert.ok(sitemap('loc').toArray().some((node) => sitemap(node).text() === site + path), `Missing sitemap entry ${path}`);
}
for (const path of articlePaths) {
  assert.ok(rss('item link').toArray().some((node) => rss(node).text() === site + path), `Missing RSS entry ${path}`);
}
assert.deepEqual(failures, []);
console.log(`Verified ${htmlFiles.length} pages, ${checked} same-site links and references, five legacy articles, RSS, sitemap, CV and static assets.`);
