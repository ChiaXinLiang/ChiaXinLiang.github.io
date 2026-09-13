"""Export published metadata from the editorial DB; validate the CI snapshot."""
from pathlib import Path
import sqlite3,json,os,argparse,tempfile

def validate(db,blog):
    series={r[0] for r in db.execute('select id from series')}; ids=set(); paths=set()
    for sid,raw in db.execute('select id,topic_tags_json from series'):
        tags=json.loads(raw)
        assert isinstance(tags,list) and 2<=len(tags)<=3 and len(tags)==len(set(tags)) and all(isinstance(t,str) and t.strip() for t in tags),f'Invalid series topic tags: {sid}'
    for r in db.execute('select id,series,content_path,tags_json,public_url from articles'):
        article_id,sid,path,tags,url=r
        assert article_id not in ids and path not in paths,'Duplicate article id or content path'
        ids.add(article_id); paths.add(path)
        candidate=(blog/path).resolve()
        assert candidate.is_relative_to((blog/'src/content/blog').resolve()) and candidate.is_file(),f'Missing or unsafe content path: {path}'
        assert candidate.suffix in ['.md','.mdx'] and sid in series|{None},f'Invalid content record: {article_id}'
        assert isinstance(json.loads(tags),list) and all(isinstance(x,str) for x in json.loads(tags)),f'Invalid tags: {article_id}'
        assert url=='https://chiaxinliang.github.io/blog/'+article_id+'/',f'Unexpected URL: {article_id}'
    assert ids,'Published metadata database is empty'
    for sid,count in db.execute('select series,count(*) from articles where series is not null group by series'):
        assert count<=30,f'{sid} exceeds the 30-article limit'
    return len(ids)

def export(source,target,blog):
    if not source.exists():
        assert target.exists(),'No editorial database or published database snapshot found'
        with sqlite3.connect('file:'+str(target)+'?mode=ro',uri=True) as db: count=validate(db,blog)
        print(f'Validated committed published database: {count} articles (editorial DB unavailable on this host).'); return
    src=sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True); src.row_factory=sqlite3.Row
    rows=list(src.execute("select a.*, s.public_listing from articles a join series s on s.id=a.series where a.status in ('published','promoted') order by a.slug"))
    assert all(r['content_path'] for r in rows),'Published article missing content_path'
    active={r['series'] for r in rows if r['public_listing']}; series=list(src.execute('select * from series order by reading_order'))
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='content-',suffix='.db',dir=target.parent); os.close(fd)
    try:
        dst=sqlite3.connect(temp)
        dst.execute('pragma foreign_keys=ON')
        dst.execute('create table series(id TEXT PRIMARY KEY,name TEXT NOT NULL,tagline TEXT NOT NULL,tag TEXT NOT NULL,level TEXT NOT NULL,reading_order INTEGER NOT NULL,topic_tags_json TEXT NOT NULL)')
        dst.execute('create table articles(id TEXT PRIMARY KEY,series TEXT REFERENCES series(id),code TEXT NOT NULL,title TEXT NOT NULL,description TEXT NOT NULL,pub_date TEXT NOT NULL,updated_date TEXT,linkedin_date TEXT,topic TEXT,level TEXT NOT NULL,reading_order INTEGER NOT NULL,hero_image TEXT,tags_json TEXT NOT NULL,content_path TEXT NOT NULL UNIQUE,public_url TEXT NOT NULL UNIQUE)')
        for s in series:
            if s['id'] in active: dst.execute('insert into series values(?,?,?,?,?,?,?)',(s['id'],s['name'],s['tagline'],s['subject_tag'],s['level'],s['reading_order'],s['topic_tags_json']))
        for r in rows:
            dst.execute('insert into articles values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(r['slug'],r['series'] if r['public_listing'] else None,r['code'],r['title'],r['description'],r['pub_date'],r['updated_date'],r['linkedin_date'],r['topic'],r['level'] or 'beginner',r['reading_order'],r['hero_image'],r['tags_json'] or '[]',r['content_path'],'https://chiaxinliang.github.io/blog/'+r['slug']+'/'))
        dst.commit(); count=validate(dst,blog); dst.close(); src.close()
        if target.exists() and target.read_bytes()==Path(temp).read_bytes(): os.unlink(temp)
        else: os.replace(temp,target)
        print(f'Exported {count} published articles and {len(active)} series from content.db. Editorial notes and plans excluded.')
    finally:
        if Path(temp).exists(): os.unlink(temp)

if __name__=='__main__':
    blog=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,default=blog.parent/'content.db'); parser.add_argument('--target',type=Path,default=blog/'data/content.db'); args=parser.parse_args()
    export(args.source,args.target,blog)
