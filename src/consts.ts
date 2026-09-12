// Global site data. Import from anywhere with the `import` keyword.

export const SITE_TITLE = "Marcus's ML Notes";
export const SITE_DESCRIPTION =
	'LLM fundamentals and AI systems performance engineering — notes, deep dives, and back-of-envelope math.';

export const GITHUB_URL = 'https://github.com/ChiaXinLiang';
export const LINKEDIN_URL = 'https://www.linkedin.com/';

// One entry per content series. Post frontmatter references `series` by id.
export const SERIES = [
	{
		id: 'llm-basics',
		name: 'Fundamental of LLM',
		tagline: 'From perceptrons to Transformers — how modern AI actually works, no ML background assumed.',
		level: 'Beginner friendly',
	},
	{
		id: 'ai-performance',
		name: 'AI Performance Engineering',
		tagline: 'GPU memory math, CUDA kernels, KV caches, and serving LLMs at scale — for people who run models on real hardware.',
		level: 'Advanced',
	},
	{
		id: 'efficient-ai',
		name: 'Efficient AI & Co-Design',
		tagline: 'Why chips and algorithms evolve together — accelerators, number formats, and the economics of every token.',
		level: 'Intermediate',
	},
] as const;

export type SeriesId = (typeof SERIES)[number]['id'];
