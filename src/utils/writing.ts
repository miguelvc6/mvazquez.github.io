import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'writing'>;

export async function publishedWriting() {
  return (await getCollection('writing', ({ data }) => !data.draft))
    .sort((a, b) => b.data.date.localeCompare(a.data.date) || a.id.localeCompare(b.id));
}

export function readingTime(body: string) {
  const prose = body
    .replace(/^```[^\n]*\n[\s\S]*?^```[ \t]*$/gm, '')
    .replace(/<[^>]*>/g, '')
    .replace(/https?:\/\/\S+/g, '')
    .replace(/^\[\^[^\]]+\]:.*$/gm, '');
  return Math.max(1, Math.ceil((prose.match(/[\p{L}\p{N}]+/gu)?.length ?? 0) / 220));
}

export function displayDate(date: string, options: Intl.DateTimeFormatOptions = { dateStyle: 'long' }) {
  return new Intl.DateTimeFormat('en', { ...options, timeZone: 'UTC' }).format(new Date(date));
}
