# Technical blog style research (2026-09-12)

Analyzed 8 exemplar personal technical blogs via firecrawl:
Lilian Weng (lilianweng.github.io), Eugene Yan, Julia Evans (jvns.ca),
Simon Willison, Chip Huyen, Dan Luu, Phil Schmid, Jay Alammar.

## What the best ones share

1. **Extreme minimalism, content-first.** No hero banners, no cards, no
   stock images. Dan Luu is literally a list of links. Fast load, high
   text density. Decoration budget goes to in-article figures instead.
2. **Dense scannable archives.** `date · title` one-liners, all posts on
   one page (jvns: "every post I've ever written, organized by category";
   Phil Schmid groups by month/year).
3. **One-line identity at the top.** "Hi, this is Lilian. I'm documenting
   my learning notes in this blog since 2017." / Jay Alammar:
   "Visualizing machine learning one concept at a time." Personal, plain,
   memorable.
4. **Article metadata**: date + estimated reading time on every post
   (Lilian: "Date | Estimated Reading Time: 40 min | Author").
5. **Table of Contents** with anchor links on long articles (Lilian's
   posts have 3-level ToCs; every heading gets a # permalink).
6. **Tags + categories** as the discovery mechanism (Simon Willison shows
   per-tag counts; Eugene tags every post).
7. **Citation block** (Lilian ends every post with BibTeX "Citation" +
   "References" sections → academic credibility, gets cited in papers).
8. **Subscribe surfaces everywhere**: RSS/Atom always visible; newsletter
   CTA on homepage (Eugene: 11,800+ readers; jvns: weekly digest).
9. **A curated entry point**: Eugene's "Start Here", jvns' "Favorites",
   Simon's "Guides" — don't make new readers dig through the archive.
10. **Search** on content-heavy sites (Lilian, Eugene, Phil ⌘K).
11. **Voice**: personal narrative openings ("I got nerdsniped when…"),
    "Notes on X" humility framing (jvns), concrete numbers early.

## Anti-patterns to avoid (seen in marketing blogs, absent here)

- Card grids with hero images for every post (wastes scan density)
- Multiple competing CTAs / popups
- Category mega-menus; deep pagination instead of a single archive

## Roadmap for this blog

Phase 1 (structure, now): dense date·title lists ✅(done in /blog),
reading-time metadata, visible RSS link, tighter identity line.
Phase 2 (when articles exist): auto-ToC on articles, per-tag pages,
BibTeX citation block for deep dives, "Start Here" page.
Phase 3 (growth): Pagefind search, newsletter (LinkedIn hook posts serve
as the digest at first).
