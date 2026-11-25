import { z } from 'zod';
import { externalUrl, records, resource, slug } from './schema';
import { person, profile } from './profile';

const publication = z.object({
  id: slug,
  title: z.string().min(1),
  authors: z.array(z.string().min(1)).min(1),
  year: z.number().int().min(1900),
  venue: z.string().min(1),
  shortVenue: z.string().min(1),
  status: z.enum(['Peer-reviewed conference paper', 'Peer-reviewed workshop paper', 'Submitted manuscript']),
  description: z.string().min(1),
  doi: z.string().regex(/^10\.\d{4,9}\/.+/).optional(),
  date: z.iso.date().optional(),
  paper: externalUrl.optional(),
  details: z.string().optional(),
  bib: z.record(z.string(), z.string()).optional(),
  links: z.array(resource).default([]),
  featured: z.boolean().default(false),
});
export type Publication = z.infer<typeof publication>;

export const publications = records(publication, [
  {
    id: 'structure-is-the-signal',
    title: 'Structure is the Signal: Graph Encodings and GNNs for Constraint Repair in Collaborative KGs',
    authors: [profile.name, 'Kevin Innerebner', 'Alexander Prock', 'Günter Klambauer', 'Elisabeth Lex', 'Johannes Schimunek', 'Axel Polleres'],
    year: 2026,
    date: '2026-05-07',
    venue: 'The Semantic Web – ESWC 2026',
    shortVenue: 'ESWC',
    status: 'Peer-reviewed conference paper',
    description: 'We compared graph encodings and graph neural networks for repairing Wikidata constraint violations from edit history. The experiments showed that how the graph encodes relationships and roles matters for recovering historical repairs.',
    doi: '10.1007/978-3-032-25156-5_15',
    paper: 'https://github.com/Tschoui/kg_repairs/blob/main/kg_repairs.pdf',
    details: 'Lecture Notes in Computer Science, vol. 16549, pp. 278–296. Springer.',
    bib: { series: 'Lecture Notes in Computer Science', volume: '16549', pages: '278--296', publisher: 'Springer' },
    links: [{ label: 'Code and data instructions', url: 'https://github.com/Tschoui/kg_repairs' }],
    featured: true,
  },
  {
    id: 'wikidata-claim-verification',
    title: 'Detecting Large Language Model Hallucinations with Knowledge-Graph Integrity Constraints: An Evaluation Framework Using Wikidata',
    authors: ['Simon D. Herbst', profile.name],
    year: 2026,
    venue: '6th Wikidata Workshop at ISWC 2026',
    shortVenue: 'ISWC Workshop',
    status: 'Peer-reviewed workshop paper',
    description: 'We evaluated a post-generation audit that grounds model answers in Wikidata and applies property constraints to flag specific types of factual inconsistency.',
    paper: 'https://wikidataworkshop.github.io/2026/papers/paper17.pdf',
    details: 'Bari, Italy.',
    bib: { address: 'Bari, Italy' },
    featured: true,
  },
  {
    id: 'neural-constraint-factors',
    title: 'Learning Repair Consequences with Neural Constraint Factors and Symbolic Verification',
    authors: [profile.name, 'Axel Polleres'],
    year: 2026,
    venue: 'Transactions on Graph Data and Knowledge (TGDK)',
    shortVenue: 'TGDK',
    status: 'Submitted manuscript',
    description: 'This manuscript studies how learned models can anticipate the consequences of candidate knowledge graph edits and how symbolic verification can evaluate those edits.',
    details: 'Special issue “Rules and Reasoning for Graph-Based Data & Knowledge.”',
    featured: true,
  },
]);

export function publicationUrl(item: Publication) {
  return item.doi ? `https://doi.org/${item.doi}` : item.paper ?? `/publications/#${item.id}`;
}

export function bibtex(item: Publication) {
  const submitted = item.status === 'Submitted manuscript';
  const fields = {
    title: `{${item.title}}`,
    author: item.authors.join(' and '),
    year: String(item.year),
    ...(submitted ? { note: `Submitted to ${item.venue}. ${item.details ?? ''}`.trim() } : { booktitle: item.venue }),
    ...item.bib,
    ...(item.doi ? { doi: item.doi } : {}),
    ...(item.paper ? { url: item.paper } : {}),
  };
  const content = Object.entries(fields).map(([key, value]) => `  ${key} = {${value}}`).join(',\n');
  return `@${submitted ? 'unpublished' : 'inproceedings'}{${item.id}-${item.year},\n${content}\n}`;
}

export function scholarlyArticle(item: Publication) {
  return {
    '@type': 'ScholarlyArticle',
    '@id': `https://mvazquez.ai/publications/#${item.id}`,
    headline: item.title,
    description: item.description,
    author: item.authors.map((name) => name === profile.name ? { '@id': person['@id'] } : { '@type': 'Person', name }),
    creativeWorkStatus: item.status,
    ...(item.status !== 'Submitted manuscript' ? { isPartOf: { '@type': 'CreativeWork', name: item.venue } } : {}),
    ...(item.date ? { datePublished: item.date } : {}),
    ...(item.doi ? { identifier: item.doi, sameAs: publicationUrl(item) } : {}),
    ...(item.paper ? { url: item.paper } : {}),
  };
}
