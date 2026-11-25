import { z } from 'zod';
import { articleUrl } from '../utils/urls';
import { profile } from './profile';
import { internalPath, records, resource, slug } from './schema';

const project = z.object({
  id: slug,
  title: z.string().min(1),
  kind: z.enum(['Research collaboration', 'Research software', 'Educational software', 'Software experiment']),
  description: z.string().min(1),
  contribution: z.string().min(1),
  technologies: z.array(z.string().min(1)).min(1),
  featured: z.boolean().default(false),
  image: z.object({ path: internalPath, alt: z.string().min(1), width: z.number().int().positive(), height: z.number().int().positive() }).optional(),
  links: z.array(resource).min(1),
});
export type Project = z.infer<typeof project>;

export const projects = records(project, [
  {
    id: 'bilai-kg', title: 'Bilateral AI Knowledge Graph', kind: 'Research collaboration',
    description: 'A shared knowledge graph for representing information across the Bilateral AI research network.',
    contribution: 'I contribute to the graph with colleagues across the cluster and organized and presented a workshop on it at the 2026 research retreat in Graz.',
    technologies: ['Knowledge graphs', 'Semantic Web'], featured: true,
    links: [{ label: 'Bilateral AI Cluster of Excellence', url: profile.affiliationUrl }],
  },
  {
    id: 'kg-repairs', title: 'Knowledge Graph Repairs', kind: 'Research software',
    description: 'The research code for comparing how graph encodings and neural models recover repairs from Wikidata edit history.',
    contribution: 'I led the accompanying ESWC paper. The implementation and evaluation were a collaboration across the Bilateral AI network.',
    technologies: ['Python', 'Wikidata', 'Graph neural networks'],
    links: [{ label: 'Source and reproduction instructions', url: 'https://github.com/Tschoui/kg_repairs' }, { label: 'Publication', url: '/publications/#structure-is-the-signal' }],
  },
  {
    id: 'pathfinding', title: 'Dijkstra and A*', kind: 'Educational software',
    description: 'An exploration of two shortest-path algorithms and how a heuristic can change which parts of a graph the search explores.',
    contribution: 'I implemented the examples to make the search process easier to follow.',
    technologies: ['Python', 'Graph search'],
    links: [{ label: 'Open the pathfinding visualizer', url: '/projects/dijkstra-astar.html' }, { label: 'Source code and experiments', url: 'https://github.com/miguelvc6/dijkstra_Astar' }],
  },
  {
    id: 'torch-tracer', title: 'Torch Tracer', kind: 'Software experiment',
    description: 'A GPU-accelerated ray tracer that batches the work of tracing many rays through the same scene.',
    contribution: 'I wrote the Python implementation and documented how it handles intersections, materials, and parallel computation.',
    technologies: ['Python', 'PyTorch', 'Ray tracing'], featured: true,
    image: { path: '/assets/img/project_raytracing.png', alt: 'Ray-traced scene with many spheres under different lighting and material effects', width: 1200, height: 675 },
    links: [{ label: 'Source code', url: 'https://github.com/miguelvc6/torch-tracer' }, { label: 'Ray-tracing article', url: articleUrl('2024-11-12-torch-tracing-01') }],
  },
  {
    id: 'react-agent', title: 'ReAct Agent', kind: 'Software experiment',
    description: 'A Python example of how a language model can alternate between reasoning steps and calls to tools.',
    contribution: 'I built the implementation and wrote a walkthrough that traces the loop through an example task.',
    technologies: ['Python', 'ReAct', 'Tool use'],
    image: { path: '/assets/img/project_react.png', alt: 'An example reasoning trace showing the sequence of agent steps and tool calls', width: 579, height: 378 },
    links: [{ label: 'Source code', url: 'https://github.com/miguelvc6/react-agent' }, { label: 'ReAct walkthrough', url: articleUrl('2024-11-25-reasoning-and-action-agents') }],
  },
  {
    id: 'hero-embeddings', title: 'Hero Embeddings', kind: 'Software experiment',
    description: 'A three-dimensional view of similarities between fictional characters, based on embeddings derived from character descriptions.',
    contribution: 'I built the embedding experiment and its interactive visualization.',
    technologies: ['BERT', 'Embeddings', 'Plotly'],
    image: { path: '/assets/img/project_hero_embeddings.png', alt: 'Three-dimensional scatter plot showing clusters of fictional character embeddings', width: 780, height: 592 },
    links: [{ label: 'Open interactive visualization', url: '/projects/3d_hero_embeddings_visualization_js.html' }, { label: 'Source code', url: 'https://github.com/miguelvc6/heroes' }],
  },
]);

export const bilaiProject = projects.find(({ id }) => id === 'bilai-kg')!;
