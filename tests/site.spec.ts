import { expect, test } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const pages = ['/', '/research/', '/publications/', '/teaching/', '/projects/', '/writing/', '/about/'];
const article = '/blog/output/2024-11-12-torch-tracing-01/content.html';
const articles = [
  '/blog/output/2024-10-28-welcome-post/content.html',
  article,
  '/blog/output/2024-11-17-llm-agents/content.html',
  '/blog/output/2024-11-20-agents-ecosystem/content.html',
  '/blog/output/2024-11-25-reasoning-and-action-agents/content.html',
];
const widths = [1440, 1280, 768, 390, 320];

for (const width of widths) {
  for (const theme of ['light', 'dark'] as const) {
    test(`${width}px ${theme}: complete layout`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.emulateMedia({ colorScheme: theme, reducedMotion: 'reduce' });
      for (const path of [...pages, article]) {
        expect((await page.goto(path))?.status(), path).toBe(200);
        await expect(page.locator('h1')).toBeVisible();
        expect(await page.evaluate(() => document.documentElement.scrollWidth), path).toBeLessThanOrEqual(width);
        expect(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme), path).toBe(theme);
      }
    });
  }
}

test('navigation, keyboard access, and theme persistence', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/#main$/);
  const toggle = page.getByRole('button', { name: 'Dark theme' });
  await toggle.click();
  await expect(toggle).toHaveAttribute('aria-pressed', 'true');
  const menu = page.locator('.mobile-nav summary');
  await menu.focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('.mobile-nav')).toHaveAttribute('open', '');
  await page.keyboard.press('Escape');
  await expect(menu).toBeFocused();
  await page.keyboard.press('Enter');
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Publications' }).click();
  await expect(page).toHaveURL('/publications/');
  await expect(toggle).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
});

test('research map works with the keyboard and without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('/research/');
  await expect(page.getByRole('img', { name: 'Knowledge graphs, neural models, and symbolic verification' })).toBeVisible();
  await expect(page.getByText('Which edit improves the graph?')).toBeVisible();
  await page.getByRole('radio', { name: 'Repair' }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('radio', { name: 'Verify' })).toBeChecked();
  await expect(page.getByText('Can a rule flag a generated claim?')).toBeVisible();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('radio', { name: 'Guide' })).toBeChecked();
  await expect(page.getByText('What feedback helps revise an edit?')).toBeVisible();
  await expect(page.getByText('Which edit improves the graph?')).toBeHidden();
  await context.close();
});

test('publications, clipboard, and CV download', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/publications/');
  const paper = page.locator('#structure-is-the-signal');
  await expect(paper.getByRole('link', { name: 'DOI' })).toHaveAttribute('href', /^https:\/\/doi\.org\//);
  await expect(page.getByRole('heading', { name: 'Submitted work' })).toBeVisible();
  await paper.locator('summary').click();
  await paper.getByRole('button', { name: 'Copy BibTeX' }).click();
  await expect(paper.getByRole('status')).toHaveText('Copied.');
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain('10.1007/978-3-032-25156-5_15');
  await page.goto('/');
  const [download] = await Promise.all([
    page.waitForEvent('download'), page.getByRole('link', { name: 'CV (PDF)' }).click(),
  ]);
  expect(download.suggestedFilename()).toBe('Miguel_Vazquez_CV.pdf');
});

test('old links, equations, figures, table of contents, and code copying', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/blog/output/blog.html');
  await expect(page).toHaveURL('/writing/');
  await page.goto('/projects/');
  await page.getByRole('link', { name: 'Ray-tracing article' }).click();
  await expect(page).toHaveURL(article);
  await expect(page.locator('math').first()).toBeVisible();
  await expect(page.locator('.article-body img').first()).toBeVisible();
  await page.getByRole('navigation', { name: 'Article contents' }).getByRole('link', { name: 'Materials', exact: true }).click();
  await expect(page).toHaveURL(/#materials$/);
  const code = page.locator('.code-frame').first();
  await code.getByRole('button', { name: 'Copy code' }).click();
  await expect(code.getByRole('status')).toHaveText('Copied.');
  expect((await page.evaluate(() => navigator.clipboard.readText())).length).toBeGreaterThan(100);
});

test('writing search finds a term in the article body and clears', async ({ page }) => {
  await page.goto('/writing/');
  const articles = page.locator('[data-writing-result]');
  const search = page.getByRole('searchbox', { name: 'Search articles' });
  await expect(articles.first()).toBeVisible();
  await search.fill('Lambertian');
  await expect(page.getByRole('status')).toHaveText('1 matching article.');
  await expect(articles.filter({ hasText: 'Torch Tracer' })).toBeVisible();
  await search.fill('no-article-contains-this-phrase');
  await expect(page.getByRole('status')).toHaveText('No matching articles.');
  await search.clear();
  await expect(articles.first()).toBeVisible();
});

test('Hero Embeddings loads its WebGL plot and searches points', async ({ page }) => {
  await page.goto('/projects/3d_hero_embeddings_visualization_js.html');
  await expect(page.locator('#myDiv canvas').first()).toBeVisible({ timeout: 30000 });
  const input = page.getByRole('searchbox', { name: /search/i });
  await input.fill('batman');
  await expect(page.getByRole('status')).toContainText(/[1-9]\d* matches?/);
  await page.getByRole('button', { name: 'Rotate right' }).click();
});

test('Dijkstra and A* visualizer replays source-code events', async ({ page }) => {
  await page.goto('/projects/dijkstra-astar.html');
  await page.getByRole('button', { name: 'A*', exact: true }).click();
  const message = page.locator('#message');
  const before = await message.textContent();
  await page.locator('#next').click();
  await expect(message).not.toHaveText(before ?? '');
  await expect(page.getByRole('img', { name: 'Directed weighted graph and current search state' })).toBeVisible();
});

for (const theme of ['light', 'dark'] as const) {
  test(`WCAG 2.2 AA automated scan in ${theme}`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: theme, reducedMotion: 'reduce' });
    for (const path of [...pages, ...articles]) {
      await page.goto(path);
      const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'])
        .analyze();
      expect(results.violations.map(({ id, nodes }) => ({ id, targets: nodes.map(({ target }) => target) })), path).toEqual([]);
    }
  });
}
