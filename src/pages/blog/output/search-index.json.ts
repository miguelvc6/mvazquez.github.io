import type { APIRoute } from 'astro';
import { publishedWriting } from '../../../utils/writing';
import { profile } from '../../../data/profile';

export const GET: APIRoute = async () => {
  const posts = await publishedWriting();
  return Response.json(posts.map(({ id, data, body }) => ({
    title: data.title,
    date: data.date,
    url: `${id}/content.html`,
    author: profile.name,
    content: body ?? data.summary,
  })));
};
