import type { CollectionEntry } from 'astro:content';

// Match the visible topic groups. Frontmatter order may include unpublished
// curriculum entries, so it is a sorting hint rather than a display number.
export function publishedSeriesOrder(posts: CollectionEntry<'blog'>[]) {
  const sorted = [...posts].sort((a, b) =>
    (a.data.order ?? 999) - (b.data.order ?? 999) ||
    a.data.pubDate.valueOf() - b.data.pubDate.valueOf());
  const groups = new Map<string, typeof sorted>();
  for (const post of sorted) {
    const topic = post.data.topic ?? 'Other';
    if (!groups.has(topic)) groups.set(topic, []);
    groups.get(topic)!.push(post);
  }
  return [...groups.values()].flat();
}
