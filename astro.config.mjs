import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import { legacyHeadings, renderMath, writingResources } from './src/utils/markdown.mjs';

export default defineConfig({
  site: 'https://mvazquez.ai',
  output: 'static',
  build: { format: 'preserve' },
  compressHTML: true,
  devToolbar: { enabled: false },
  integrations: [sitemap({
    filter: (url) => !/\/(?:blog(?:\.html)?|404(?:\.html)?)\/?$/.test(url),
    serialize: (item) => ({
      ...item,
      url: /\/content(?:\.html)?\/?$/.test(item.url)
        ? item.url.replace(/\/content(?:\.html)?\/?$/, '/content.html')
        : item.url.replace(/\/?$/, '/'),
    }),
  })],
  markdown: {
    syntaxHighlight: { type: 'shiki', excludeLangs: ['mermaid'] },
    shikiConfig: { themes: { light: 'github-light-high-contrast', dark: 'github-dark-high-contrast' }, defaultColor: false },
    processor: unified({
      smartypants: false,
      remarkPlugins: [remarkMath],
      rehypePlugins: [legacyHeadings, writingResources, renderMath],
    }),
  },
});
