# Article visual policy

Each technical article uses Overview → Deep dive → Conclusion. Include one overview and 3–6 deep-dive figures, chosen according to distinct mechanisms rather than a fixed count. Split complex mechanisms into focused components. Place each figure before its explanatory prose. The conclusion has no figure.

Use AI thesis/CVPR research figures: white backgrounds, restrained colors, legible labels, architecture blocks, tensor/matrix views and clean arrows. Real images are appropriate for vision inputs. Avoid exaggerated scenes, treasure chests, valves, robots, conveyors, anthropomorphic machines and decorative props. Numbers must be supported by the article or clearly labeled illustrative.

Generate bitmaps with the built-in image_gen tool and visually review style and technical accuracy. Synchronize approved assets with canonical source copies. Update the editorial catalog hero_image to ./section-overview.png and export data/content.db; frontmatter alone does not update listing covers.

The shared viewer includes all inline figures in reading order, with arrows, keyboard navigation and swipe. Clicking a figure opens the image viewer with copy and download controls.

The GitHub Pages workflow runs npm run build. Its prebuild validates phase order, figure placement, catalog cover consistency, asset existence and article-specific targets in article-visual-coverage.json. Add targets for new technical articles. Validation failures block deployment.

The completed rollout covers 171 technical articles: 171 overview and 629 deep-dive figures, 800 total. All 660 generated assets and 140 retained research-style assets were visually reviewed. Counts vary: 100 articles have 3 deep dives, 36 have 4, 25 have 5 and 10 have 6.
