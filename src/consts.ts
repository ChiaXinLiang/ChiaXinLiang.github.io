import { readPublishedCatalog } from './lib/content-database';

// Global site data. Import from anywhere with the `import` keyword.

export const SITE_TITLE = "Xin Liang Chia's Notes";
export const SITE_DESCRIPTION =
	'LLM fundamentals and AI systems performance engineering: notes, deep dives, and back-of-envelope math.';

export const GITHUB_URL = 'https://github.com/ChiaXinLiang';
export const LINKEDIN_URL = 'https://www.linkedin.com/in/xin-liang-chia-7126261a6/';

export const SERIES = readPublishedCatalog().series;
export type SeriesId = (typeof SERIES)[number]['id'];
