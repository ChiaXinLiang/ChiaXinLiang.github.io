import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const require=createRequire(import.meta.url);
const testRequire=process.argv[2] ? createRequire(path.resolve(process.argv[2], 'package.json')) : require;
const {JSDOM}=testRequire('jsdom');
const {transform}=require('esbuild');
const b=fileURLToPath(new URL('../', import.meta.url)).replace(/\/$/, '');
const code=(await transform(fs.readFileSync(b+'/src/scripts/article-visuals.ts','utf8'),{loader:'ts',format:'iife',globalName:'ArticleVisualModule'})).code;
let downloads=[],copied=[];
function setup(html,clipboard=true){
 const dom=new JSDOM(html,{url:'https://example.org/blog/test/',runScripts:'outside-only'}); const w=dom.window;
 w.HTMLDialogElement.prototype.showModal=function(){this.open=true};
 w.HTMLDialogElement.prototype.close=function(){this.open=false;this.dispatchEvent(new w.Event('close'))};
 w.fetch=async()=>({ok:true,blob:async()=>new w.Blob(['image'],{type:'image/png'})});
 w.URL.createObjectURL=()=> 'blob:image';w.URL.revokeObjectURL=()=>{};
 w.setTimeout=()=>0;
 w.HTMLAnchorElement.prototype.click=function(){downloads.push({name:this.download,href:this.href})};
 Object.defineProperty(w,'isSecureContext',{value:true});
 if(clipboard){w.ClipboardItem=class{constructor(data){this.data=data}};Object.defineProperty(w.navigator,'clipboard',{value:{write:async items=>copied.push(await items[0].data['image/png'])}})}
 w.eval(code);w.ArticleVisualModule.initArticleVisuals();return {w,d:w.document,dom};
}
const dir=b+'/dist/blog';const paths=fs.readdirSync(dir).filter(x=>fs.existsSync(dir+'/'+x+'/index.html'));let articles=0,figures=0;
for(const slug of paths){
 const html=fs.readFileSync(dir+'/'+slug+'/index.html','utf8');const {w,d,dom}=setup(html);const root=d.querySelector('[data-article-visuals]');if(!root){dom.window.close();continue}
 const imgs=[...d.querySelectorAll('.article-content img')];const n=imgs.length||d.querySelectorAll('.hero-image img').length;
 assert.equal(root.hidden,n===0,slug+' enhancement');
 if(!n){dom.window.close();continue}
 articles++;figures+=n;
 assert.equal(d.querySelector('[data-slide-count]').textContent,`1 / ${n}`);
 assert.equal(d.querySelectorAll('[data-slide-pages] button').length,n);
 assert.equal(d.querySelector('[data-slide-prev]').disabled,true);
 if(n>1){d.querySelector('[data-slide-next]').click();assert.equal(d.querySelector('[data-slide-count]').textContent,`2 / ${n}`);root.dispatchEvent(new w.KeyboardEvent('keydown',{key:'ArrowLeft',bubbles:true}));assert.equal(d.querySelector('[data-slide-count]').textContent,`1 / ${n}`)}
 const buttons=d.querySelectorAll('[data-slide-pages] button');buttons[n-1].click();assert.equal(d.querySelector('[data-slide-next]').disabled,true);assert.equal(buttons[n-1].getAttribute('aria-current'),'true');
 const opener=d.querySelector('[data-open-image]');opener.click();const dialog=d.querySelector('dialog');assert.equal(dialog.open,true);assert.equal(d.body.style.overflow,'hidden');assert.equal(d.querySelector('[data-image-count]').textContent,`${n} / ${n}`);assert.equal(d.activeElement,d.querySelector('[data-image-close]'));
 d.querySelector('[data-image-zoom]').click();assert.equal(d.querySelector('[data-image-viewport]').classList.contains('is-zoomed'),true);assert.equal(d.querySelector('[data-image-zoom]').getAttribute('aria-pressed'),'true');dialog.dispatchEvent(new w.KeyboardEvent('keydown',{key:'ArrowLeft',bubbles:true}));assert.equal(d.querySelector('[data-image-count]').textContent,`${n} / ${n}`);
 d.querySelector('[data-image-zoom]').click();if(n>1){dialog.dispatchEvent(new w.KeyboardEvent('keydown',{key:'ArrowLeft',bubbles:true}));assert.equal(d.querySelector('[data-image-count]').textContent,`${n-1} / ${n}`)}
 d.querySelector('[data-image-close]').click();assert.equal(dialog.open,false);assert.equal(d.body.style.overflow,'');assert.equal(d.activeElement,opener);
 if(imgs.length){imgs[imgs.length-1].dispatchEvent(new w.KeyboardEvent('keydown',{key:'Enter',bubbles:true}));assert.equal(dialog.open,true);assert.equal(d.querySelector('[data-lightbox-image]').alt,imgs[imgs.length-1].alt);dialog.click();assert.equal(dialog.open,false);assert.equal(d.activeElement,imgs[imgs.length-1]);}
 w.ArticleVisualModule.initArticleVisuals();assert.equal(d.querySelectorAll('[data-slide-pages] button').length,n);
 dom.window.close();
}
const html=fs.readFileSync(dir+'/maximum-likelihood-estimation/index.html','utf8');const {w,d}=setup(html);const root=d.querySelector('[data-article-visuals]');const next=d.querySelector('[data-slide-next]');
function touch(type,list,key){const e=new w.Event(type,{bubbles:true});Object.defineProperty(e,key,{value:list});d.querySelector('[data-open-image]').dispatchEvent(e)}
touch('touchstart',[{clientX:200,clientY:100}],'touches');touch('touchend',[{clientX:40,clientY:105}],'changedTouches');assert.equal(d.querySelector('[data-slide-count]').textContent.startsWith('2 /'),true);d.querySelector('[data-open-image]').click();assert.equal(d.querySelector('dialog').open,false,'swipe synthetic click suppressed');
// Direct inline opening is not suppressed after carousel swipe.
d.querySelector('.article-content img').click();assert.equal(d.querySelector('dialog').open,true);
d.querySelector('[data-image-copy]').click();await new Promise(r=>setImmediate(r));assert.equal(copied.length,1);assert.equal(copied[0].type,'image/png');assert.match(d.querySelector('[data-image-status]').textContent,/copied/);
d.querySelector('[data-image-download]').click();await new Promise(r=>setImmediate(r));assert.equal(downloads.length,1);assert.match(downloads[0].name,/image-1\.png$/);
w.fetch=async()=>({ok:false});d.querySelector('[data-image-copy]').click();await new Promise(r=>setImmediate(r));assert.match(d.querySelector('[data-image-status]').textContent,/Could not copy/);assert.equal(d.querySelector('[data-image-copy]').disabled,false);d.querySelector('[data-image-download]').click();await new Promise(r=>setImmediate(r));assert.match(d.querySelector('[data-image-status]').textContent,/Could not download/);
const unavailable=setup(html,false);assert.equal(unavailable.d.querySelector('[data-image-copy]').disabled,true);assert.equal(unavailable.d.querySelector('[data-image-download]').disabled,false);
const empty=setup(html.replace(/<img\b[^>]*>/g,''));assert.equal(empty.d.querySelector('[data-article-visuals]').hidden,true);assert.equal(empty.d.body.classList.contains('has-visual-walkthrough'),false);
console.log(`PASS: ${articles} articles, ${figures} body/fallback figures; selection, boundaries, keyboard, focus, modal/backdrop close, zoom, swipe/click suppression, copy/download, failure handling, unsupported clipboard, empty article, idempotent initialization.`);
// Cover the SVG/non-PNG clipboard conversion branch and preserved download format.
const converted=setup(html);const cw=converted.w, cd=converted.d;let drew=false;
cw.fetch=async()=>({ok:true,blob:async()=>new cw.Blob(['<svg/>'],{type:'image/svg+xml'})});
cw.HTMLImageElement.prototype.decode=async function(){};
Object.defineProperty(cw.HTMLImageElement.prototype,'naturalWidth',{get:()=>1536});Object.defineProperty(cw.HTMLImageElement.prototype,'naturalHeight',{get:()=>1024});
cw.HTMLCanvasElement.prototype.getContext=function(){return {drawImage:()=>{drew=true}}};cw.HTMLCanvasElement.prototype.toBlob=function(callback,type){assert.equal(this.width,1536);assert.equal(this.height,1024);callback(new cw.Blob(['png'],{type}))};
cd.querySelector('[data-open-image]').click();cd.querySelector('[data-image-copy]').click();await new Promise(r=>setImmediate(r));assert.equal(drew,true);assert.equal(copied.at(-1).type,'image/png');cd.querySelector('[data-image-download]').click();await new Promise(r=>setImmediate(r));assert.match(downloads.at(-1).name,/\.svg$/);
console.log('PASS: SVG copied through native canvas PNG conversion; SVG download retains its served format.');
