import { z } from 'zod';

export const slug = z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/);
export const internalPath = z.string().regex(/^\/(?!\/)/);
export const externalUrl = z.url({ protocol: /^https$/ });
export const resource = z.object({
  label: z.string().min(1),
  url: z.union([internalPath, externalUrl]),
});
export type Resource = z.infer<typeof resource>;

export function records<T extends z.ZodType<{ id: string }>>(schema: T, entries: z.input<T>[]) {
  return z.array(schema).refine(
    (items) => new Set(items.map(({ id }) => id)).size === items.length,
    'IDs must be unique',
  ).parse(entries);
}
