import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'zod';
import { internalPath } from './data/schema';

const writing = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/writing' }),
  schema: z.object({
    title: z.string().min(1),
    date: z.iso.date(),
    summary: z.string().min(1),
    tags: z.array(z.string().min(1)).min(1),
    draft: z.boolean().default(false),
    image: internalPath.optional(),
  }),
});
export const collections = { writing };
