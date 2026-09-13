# Content database workflow

The master database is `../content.db`, outside the blog repository. It owns article metadata and series definitions. Markdown/MDX supplies the article body. Metadata in article frontmatter is a legacy copy and is not used to render database-managed articles.

## 1. Record links

The master `articles` table includes:

- `content_path`: Markdown/MDX path relative to the blog, such as `src/content/blog/profiling-basics-where-time-goes/index.md`.
- `canonical_path`: optional original source path relative to the parent content workspace.
- `slug`: stable public article ID.
- Title, description, topic, tags, reading order, image reference, and dates.

The `article_links` view exposes the public hyperlink for a published record. `tools/content show SERIES CODE`, run from the parent workspace, displays the metadata, content paths, and public URL.

The master `series` table owns names, descriptions, subject tags, and whole-series Beginner/Intermediate/Advanced labels. These labels do not split the inside of a series into levels.

## 2. Edit metadata

Run these commands from the blog directory. They modify the master database, not Markdown frontmatter:

```sh
python3 tools/edit-content.py article ai-performance serve-1
python3 tools/edit-content.py article ai-performance serve-1 --title "Profiling Basics: Finding Where the Time Actually Goes"
python3 tools/edit-content.py article ai-performance serve-1 --description "Find the actual bottleneck before choosing an optimization."
python3 tools/edit-content.py series ai-networking --tag "AI Infrastructure" --level Intermediate
```

With no edit flags, the command displays the record. Article flags include `--title`, `--description`, `--topic`, `--content-path`, `--hero-image`, `--order`, and `--tags` (comma-separated). Series flags include `--name`, `--tagline`, `--tag`, and `--level`. An image path is relative to the linked Markdown file. A content path must identify a file under `src/content/blog`.

Edit the linked Markdown file for content changes. Keep the slug stable to preserve existing URLs. Article status and publication dates remain explicit editorial fields in the master database; only `published` and `promoted` records are exported.

## 3. Build and deploy

```sh
npm run build
```

The build first runs `tools/export-content.py`. It creates `data/content.db`, a generated published-only SQLite database containing public article metadata and active series. Drafts, plans, editorial notes, and canonical filesystem paths are excluded. This file belongs in Git so GitHub Pages can build from it when the master database is unavailable.

Astro's database loader selects content files by `content_path` and uses database metadata for those files. Database metadata participates in the content digest, so a title or tag edit invalidates the cached entry even when its Markdown body is unchanged. The [Astro Content Loader API](https://docs.astro.build/en/reference/content-loader-reference/) preserves Markdown rendering and image handling.

Commit the regenerated `data/content.db` and any edited Markdown/code, then deploy through the existing GitHub Pages workflow. CI validates the committed database snapshot and queries it during the static build. The database itself is not copied into `dist` or served to browser visitors. Search and filters use public data embedded in the generated HTML; there is no runtime database server.

`npm run metadata` refreshes the published database without building. `npm run dev` refreshes it before starting the preview; restart the preview after editing master metadata. The parent `tools/content sync` now validates published DB-to-content links instead of reimporting metadata from Markdown.

## 4. Verification

An isolated test database was used to change an article title and description, a whole-series subject/level tag, and a publication status without editing Markdown. The resulting build reflected the metadata changes and excluded the draft. The real master database was unchanged, then the normal published snapshot was restored. All 94 long-form articles still have consecutive visible numbering and aligned Previous/Next links. Published database snapshots validate content paths, IDs, tags, URLs, and the 30-article series limit before replacing the previous snapshot.
