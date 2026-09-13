import { DatabaseSync } from 'node:sqlite';
import { resolve } from 'node:path';

export interface PublishedSeries { id: string; name: string; tagline: string; tag: string; level: string; topicTags: string[]; }
interface ArticleRow {
  id: string; series: string | null; code: string; title: string; description: string;
  pub_date: string; updated_date: string | null; linkedin_date: string | null;
  topic: string | null; level: string; reading_order: number;
  hero_image: string | null; tags_json: string; content_path: string; public_url: string;
}
const calendarDate = (value: string) => /^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00` : value;
export const publishedDatabasePath = resolve(process.cwd(), 'data/content.db');

// Server/build only. This database is never copied into the site's public files.
export function readPublishedCatalog() {
  const db = new DatabaseSync(publishedDatabasePath, { readOnly: true });
  try {
    const series = db.prepare('SELECT id,name,tagline,tag,level,topic_tags_json FROM series ORDER BY reading_order').all().map(row => {
      const { topic_tags_json, ...metadata } = row;
      return { ...metadata, topicTags: JSON.parse(topic_tags_json as string) };
    }) as unknown as PublishedSeries[];
    const rows = db.prepare('SELECT * FROM articles ORDER BY id').all() as unknown as ArticleRow[];
    const articles = rows.map(row => ({
      id: row.id, contentPath: row.content_path, url: row.public_url,
      metadata: {
        title: row.title, description: row.description, pubDate: calendarDate(row.pub_date),
        code: row.code, order: row.reading_order, level: row.level, tags: JSON.parse(row.tags_json) as string[],
        ...(row.series ? { series: row.series } : {}),
        ...(row.topic ? { topic: row.topic } : {}),
        ...(row.updated_date ? { updatedDate: calendarDate(row.updated_date) } : {}),
        ...(row.linkedin_date ? { linkedinDate: calendarDate(row.linkedin_date) } : {}),
        ...(row.hero_image ? { heroImage: row.hero_image } : {}),
      },
    }));
    return { series, articles };
  } finally { db.close(); }
}
