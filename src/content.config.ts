import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const blog = defineCollection({
	// Long-form articles, organized by series/topic.
	loader: glob({ base: './src/content/blog', pattern: '**/*.{md,mdx}' }),
	schema: ({ image }) =>
		z.object({
			title: z.string(),
			description: z.string(),
			pubDate: z.coerce.date(),
			updatedDate: z.coerce.date().optional(),
			// LinkedIn promotion date (from the series SCHEDULE) — separate from
			// pubDate, which is the date the article was uploaded to the blog.
			linkedinDate: z.coerce.date().optional(),
			heroImage: z.optional(image()),
			series: z.enum(['llm-basics', 'ai-performance', 'efficient-ai', 'comp-arch', 'llm-architectures']).optional(),
			// article code in content.db (e.g. 'nn-1') — lets `tools/content sync` auto-track status
			code: z.string().optional(),
			// reading order within the series (drives series-page sorting)
			order: z.number().optional(),
			topic: z.string().optional(),
			tags: z.array(z.string()).optional(),
		}),
});

const posts = defineCollection({
	// Short-form notes: quick takes, TILs, links, life updates.
	loader: glob({ base: './src/content/posts', pattern: '**/*.{md,mdx}' }),
	schema: z.object({
		title: z.string(),
		description: z.string().optional(),
		pubDate: z.coerce.date(),
		tags: z.array(z.string()).optional(),
	}),
});

export const collections = { blog, posts };
