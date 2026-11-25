import portrait from '../assets/photo_03_lightweight.webp';
import aboutPhoto from '../assets/photo_01_lightweight.webp';

export const profile = {
  name: 'Miguel Vázquez',
  role: 'PhD researcher in artificial intelligence',
  institution: 'WU Vienna',
  institute: 'Institute for Data, Process and Knowledge Management',
  affiliation: 'Bilateral AI Cluster of Excellence',
  affiliationUrl: 'https://www.bilateral-ai.net/',
  location: 'Vienna, Austria',
  email: 'miguel@mvazquez.ai',
  cv: '/cv/Miguel_Vazquez_CV.pdf',
  portrait,
  aboutPhoto,
  introduction: 'I study how knowledge graphs and symbolic constraints can help AI systems detect factual errors, repair structured knowledge, and use feedback to revise their answers.',
  profiles: [
    { label: 'Google Scholar', url: 'https://scholar.google.com/citations?user=hrX5DZQAAAAJ' },
    { label: 'ORCID', url: 'https://orcid.org/0009-0002-9067-959X' },
    { label: 'GitHub', url: 'https://github.com/miguelvc6' },
    { label: 'LinkedIn', url: 'https://www.linkedin.com/in/miguel-vazquez123/' },
    { label: 'WU profile', url: 'https://research.wu.ac.at/en/persons/miguel-vazquez/' },
  ],
} as const;

export const person = {
  '@type': 'Person',
  '@id': 'https://mvazquez.ai/#miguel-vazquez',
  name: profile.name,
  jobTitle: profile.role,
  email: profile.email,
  worksFor: { '@type': 'CollegeOrUniversity', name: profile.institution, url: 'https://www.wu.ac.at/' },
  affiliation: { '@type': 'ResearchOrganization', name: profile.affiliation, url: profile.affiliationUrl },
  sameAs: profile.profiles.map(({ url }) => url),
};
