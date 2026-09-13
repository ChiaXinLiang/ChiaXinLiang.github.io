"""Edit master database metadata; npm run build exports it for deployment."""
from pathlib import Path
import argparse,sqlite3,json
blog=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('--db',type=Path,default=blog.parent/'content.db'); commands=parser.add_subparsers(dest='kind',required=True)
article=commands.add_parser('article'); article.add_argument('series'); article.add_argument('code')
for field in ['title','description','topic','content-path','hero-image']:article.add_argument('--'+field)
article.add_argument('--order',type=int); article.add_argument('--tags',help='Comma-separated article tags')
series=commands.add_parser('series'); series.add_argument('id')
for field in ['name','tagline','tag']:series.add_argument('--'+field)
series.add_argument('--topic-tags',help='Comma-separated series topic tags (2–3 unique tags)')
series.add_argument('--level',choices=['Beginner','Intermediate','Advanced'])
args=parser.parse_args(); assert args.db.exists(),'Master content.db not found'
db=sqlite3.connect(args.db); db.row_factory=sqlite3.Row
if args.kind=='article':
    table='articles'; where='series=? AND code=?'; key=[args.series,args.code]
    fields={'title':args.title,'description':args.description,'topic':args.topic,'content_path':args.content_path,'hero_image':args.hero_image,'reading_order':args.order,'tags_json':json.dumps([x.strip() for x in args.tags.split(',') if x.strip()]) if args.tags is not None else None}
    if args.content_path is not None:
        path=(blog/args.content_path).resolve(); assert path.is_relative_to((blog/'src/content/blog').resolve()) and path.is_file(),'Content path must identify an existing Markdown file under src/content/blog'
    if args.order is not None:assert args.order>0,'Order must be positive'
else:
    topic_tags=[t.strip() for t in args.topic_tags.split(',') if t.strip()] if args.topic_tags is not None else None
    if topic_tags is not None:assert 2<=len(topic_tags)<=3 and len(topic_tags)==len(set(topic_tags)),'Choose 2–3 unique series topic tags'
    table='series';where='id=?';key=[args.id];fields={'name':args.name,'tagline':args.tagline,'subject_tag':args.tag,'level':args.level,'topic_tags_json':json.dumps(topic_tags) if topic_tags is not None else None}
fields={k:v for k,v in fields.items() if v is not None}
assert db.execute('SELECT 1 FROM '+table+' WHERE '+where,key).fetchone(),'Database record not found'
if fields:
    with db:db.execute('UPDATE '+table+' SET '+','.join(k+'=?' for k in fields)+' WHERE '+where,list(fields.values())+key)
row=dict(db.execute('SELECT * FROM '+table+' WHERE '+where,key).fetchone())
if args.kind=='article':row['public_url']=db.execute('SELECT public_url FROM article_links WHERE '+where,key).fetchone()[0]
print(json.dumps(row,ensure_ascii=False,indent=2));db.close()
