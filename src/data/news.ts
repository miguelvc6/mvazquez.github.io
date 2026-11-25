import { z } from 'zod';
import { records, resource, slug } from './schema';

const activity = z.object({
  id: slug,
  date: z.string().regex(/^\d{4}-(0[1-9]|1[0-2])$/),
  title: z.string().min(1),
  description: z.string().min(1),
  link: resource.optional(),
});

export const news = records(activity, [
  {
    id: 'bilai-retreat', date: '2026-11', title: 'Bilateral AI Research Retreat',
    description: 'Organized and presented a workshop on the Bilateral AI Knowledge Graph in Graz.',
    link: { label: 'Bilateral AI Knowledge Graph', url: '/projects/#bilai-kg' },
  },
  {
    id: 'iswc', date: '2026-10', title: 'ISWC and the Wikidata Workshop',
    description: 'Presented our workshop paper, served on the workshop program committee, and participated in the ISWC Doctoral Consortium in Bari.',
    link: { label: 'Workshop paper', url: '/publications/#wikidata-claim-verification' },
  },
  {
    id: 'essai', date: '2026-07', title: 'Teaching Wikidata at ESSAI',
    description: 'Tutored the Wikidata course at the European Summer School on Artificial Intelligence in Vienna.',
    link: { label: 'Summer school course', url: '/teaching/#essai-wikidata' },
  },
  {
    id: 'eswc', date: '2026-05', title: 'Graph repair at ESWC',
    description: 'Our full research paper appeared in the ESWC proceedings in Dubrovnik.',
    link: { label: 'Conference paper', url: '/publications/#structure-is-the-signal' },
  },
]).sort((a, b) => b.date.localeCompare(a.date));
