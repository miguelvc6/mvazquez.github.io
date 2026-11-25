import rss from '@astrojs/rss';
import type { APIRoute } from 'astro';
import { profile } from '../data/profile';
import { articleUrl } from '../utils/urls';
import { publishedWriting } from '../utils/writing';

export const GET: APIRoute = async ({ site }) => rss({
  title: `${profile.name} — Writing`,
  description: 'Technical articles and research notes by Miguel Vázquez.',
  site: site!,
  items: (await publishedWriting()).map(({ id, data }) => ({
    title: data.title,
    description: data.summary,
    pubDate: new Date(data.date),
    categories: data.tags,
    link: articleUrl(id),
  })),
});
