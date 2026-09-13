import { glob } from 'astro/loaders';
import type { Loader } from 'astro/loaders';
import { readPublishedCatalog } from './content-database';

export function databaseMarkdown(): Loader {
  return {
    name: 'published-database-markdown',
    async load(context) {
      const catalog = readPublishedCatalog();
      const byId = new Map(catalog.articles.map(article => [article.id, article]));
      const byFile = new Map(catalog.articles.map(article => [article.contentPath.replace(/^src\/content\/blog\//, ''), article]));
      const metadataDigest = JSON.stringify(catalog);
      const loader = glob({
        base: new URL('../content/blog/', import.meta.url),
        pattern: [...byFile.keys()],
        generateId: ({ entry }) => {
          const article = byFile.get(entry);
          if (!article) throw new Error(`Content file has no published database record: ${entry}`);
          return article.id;
        },
      });
      await loader.load({
        ...context,
        // Metadata edits must invalidate the content cache even if Markdown is unchanged.
        generateDigest: data => context.generateDigest(JSON.stringify(data) + metadataDigest),
        parseData: ({ id, filePath }) => {
          const article = byId.get(id);
          if (!article) throw new Error(`Missing published metadata: ${id}`);
          return context.parseData({ id, filePath, data: article.metadata });
        },
      });
    },
  };
}
