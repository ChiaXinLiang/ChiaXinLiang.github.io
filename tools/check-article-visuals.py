"""Validate the article structure and visual coverage before deployment."""
from pathlib import Path
import re,json,collections,sqlite3
blog=Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name=='tools' else Path('/Volumes/X10ProMacus1/Linkedin/blog')
images=re.compile(r'^!\[[^\]]*\]\(([^\n)]+)\)\s*$',re.M)
targets={row['content_path']:row['deep_dive_images'] for row in json.loads((blog/'docs/article-visual-coverage.json').read_text())}
with sqlite3.connect('file:'+str(blog/'data/content.db')+'?mode=ro',uri=True) as db:
 catalog=dict(db.execute('select content_path,hero_image from articles'))
counts=collections.Counter();total=0
for article in sorted((blog/'src/content/blog').glob('*/index.md')):
 text=article.read_text()
 if not re.search(r'^series:',text,re.M):continue
 body=text.split('---',2)[2].strip()
 headings=re.findall(r'^## (.+)$',body,re.M)
 assert headings==['Overview','Deep dive','Conclusion'],(article,headings)
 assert catalog.get(str(article.relative_to(blog)))=='./section-overview.png',(article,'Catalog cover disagrees with overview')
 overview,deep,conclusion=body.split('## Overview\n',1)[1].split('## Deep dive\n',1)[0],body.split('## Deep dive\n',1)[1].split('## Conclusion\n',1)[0],body.split('## Conclusion\n',1)[1]
 assert overview.lstrip().startswith('!['),article
 overview_refs=images.findall(overview);deep_refs=images.findall(deep)
 assert overview_refs==['./section-overview.png'],(article,overview_refs)
 assert 3<=len(deep_refs)<=6,(article,len(deep_refs))
 assert len(deep_refs)==targets.get(str(article.relative_to(blog))),(article,'Article-specific coverage target',len(deep_refs))
 assert not images.search(conclusion),article
 assert re.search(r"^heroImage: ['\"]\./section-overview\.png['\"]$",text,re.M),article
 for section in re.split(r'^### .+$',deep,flags=re.M):
  if images.search(section):assert section.lstrip().startswith('!['),(article,'Image does not precede section explanation')
 for part in (overview,deep):
  ms=list(images.finditer(part))
  for i,m in enumerate(ms):
   after=part[m.end():ms[i+1].start() if i+1<len(ms) else len(part)].strip()
   assert after and not after.startswith('#'),(article,'Image has no following explanation',m[1])
 for ref in overview_refs+deep_refs:
  assert ref.startswith('./') and (article.parent/ref).is_file(),(article,ref)
 counts[len(deep_refs)]+=1;total+=1
assert total>=171,total
print(json.dumps({'articles':total,'overview_per_article':1,'deep_dive_distribution':dict(counts),'structure':'Overview -> Deep dive -> Conclusion','image_placement':'Before explanation'},indent=2))
