import { z } from 'zod';
import { records, resource, slug } from './schema';
import { profile } from './profile';

const course = z.object({
  id: slug,
  title: z.string().min(1),
  institution: z.string().min(1),
  level: z.string().min(1),
  date: z.string().min(1),
  role: z.string().min(1),
  description: z.string().min(1),
  category: z.enum(['course', 'supervision']),
  links: z.array(resource).default([]),
});
export type Course = z.infer<typeof course>;

export const teaching = records(course, [
  {
    id: 'foundations-ai', title: 'Foundations of Artificial Intelligence', institution: profile.institution,
    level: 'MSc in Digital Economy', date: 'Winter semester 2026', role: 'Lecturer and course organizer',
    description: 'I organize and teach the course and develop its lectures and practical exercises. The materials cover symbolic AI, machine learning, and neural networks.',
    category: 'course',
  },
  {
    id: 'foundations-is', title: 'Foundations of Information Systems', institution: profile.institution,
    level: 'BSc', date: 'Summer and winter semesters 2026', role: 'Lecturer',
    description: 'I teach this undergraduate course on the foundations of information systems.',
    category: 'course',
  },
  {
    id: 'essai-wikidata', title: 'Wikidata: A Backbone for Hybrid/Bilateral AI', institution: 'European Summer School on Artificial Intelligence, Vienna',
    level: 'Summer school', date: 'July 2026', role: 'Tutor',
    description: 'I helped teach hands-on sessions on querying and working with Wikidata in hybrid AI applications.',
    category: 'course', links: [{ label: 'Course site and materials', url: 'https://wikidata-essai.ai.wu.ac.at/' }],
  },
  {
    id: 'thesis-supervision', title: 'Bachelor’s thesis supervision', institution: profile.institution,
    level: 'BSc', date: '2026', role: 'Supervisor',
    description: 'I supervise bachelor’s research on Wikidata integrity constraints and factual error detection in language model outputs.',
    category: 'supervision',
  },
]);
