"""Offline anonymous model viewing and human score collection."""

import base64
import hashlib
import io
import json
import random
import tempfile
from pathlib import Path

from PIL import Image

from minecraft_skin_gan.preview import create_preview
from minecraft_skin_gan.skin_layout import atlas_layout

_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skin studio · review</title><style>
:root{color-scheme:dark;--ink:#edf2f7;--muted:#a5b4c6;--panel:#172231;--accent:#83e8bc}
*{box-sizing:border-box}body{margin:0;background:#0e1722;color:var(--ink);font:15px system-ui,sans-serif}
header{padding:22px 28px;border-bottom:1px solid #2e3b4b;display:flex;gap:20px;align-items:center;justify-content:space-between;flex-wrap:wrap}
h1{font-size:24px;margin:0 0 5px}p{color:var(--muted);margin:5px 0;line-height:1.5}
button,select,input,textarea{font:inherit;color:var(--ink);background:#243347;border:1px solid #45566c;border-radius:8px;padding:9px}
button{cursor:pointer}button:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,input:focus-visible,textarea:focus-visible{outline:2px solid var(--accent);outline-offset:3px}button:disabled{opacity:.4;cursor:default}
.primary{background:var(--accent);color:#10271f;border:0;font-weight:700}.toolbar{display:flex;gap:8px;flex-wrap:wrap}
main{max-width:1500px;margin:auto;padding:24px;display:grid;grid-template-columns:180px minmax(320px,1fr) 320px;gap:20px}
.panel{background:var(--panel);border:1px solid #2e3b4b;border-radius:16px;padding:18px;min-width:0}
h2{font-size:18px;margin:0 0 14px}#candidates{display:grid;grid-template-columns:1fr 1fr;gap:8px;max-height:650px;overflow:auto;padding:3px}
.tile{padding:8px;font-size:12px}.tile img{display:block;width:100%;image-rendering:pixelated;margin-bottom:6px;background:#334252}.tile[aria-current=true]{border:2px solid var(--accent)}.tile.done::after{content:' ✓';color:var(--accent)}
iframe{width:100%;height:470px;border:0;border-radius:10px;background:#fff}.heading{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap}.heading h2{margin:0}
#atlas{width:128px;height:128px;image-rendering:pixelated;background:repeating-conic-gradient(#526070 0% 25%,#324050 0% 50%) 0/16px 16px}.inspect{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:14px}
label{display:block;margin:14px 0 6px;font-weight:600}select,textarea{width:100%}textarea{resize:vertical;min-height:80px}small{color:var(--muted);line-height:1.4;display:block;margin-top:5px}.score-row{display:grid;grid-template-columns:1fr 85px;gap:10px;align-items:center;margin-top:14px}.score-row label{margin:0}.score-row small{grid-column:1/-1;margin-top:-5px}
#status{min-height:24px;color:var(--accent);font-size:13px}#progress{color:var(--accent)}#reviewer{width:160px}footer{padding:0 26px 24px;max-width:1500px;margin:auto}#import{max-width:220px}.zoom{display:flex;gap:8px;align-items:center}#zoom{width:100px}
@media(max-width:1150px){main{grid-template-columns:minmax(280px,1fr) 280px}.candidates{grid-column:1/-1;padding:12px}.candidates h2,.candidates p{display:inline-block;margin:0 12px 8px 0}#candidates{display:flex;max-height:110px;overflow:auto}.tile{flex:0 0 68px}.tile img{width:45px;height:45px}.score-row{max-width:550px}}
@media(max-width:700px){main{padding:12px;grid-template-columns:1fr}.candidates{order:3}header{padding:18px}.ratings{grid-column:auto}.inspect{flex-wrap:wrap}}
#rating-panel{scroll-margin-top:16px}.ratings h2{color:var(--accent)}.mouse-hint{margin:0 0 10px;font-size:13px}
</style></head><body>
<header><div><h1>Skin studio</h1><p>Inspect the model. Score what you see.</p></div><div class="toolbar"><button id="save-html" class="primary">Save rated HTML</button><button id="csv">Export scores CSV</button><button id="backup">Save JSON backup</button></div></header>
<main><aside class="panel candidates"><h2>Candidates</h2><p id="progress"></p><div id="candidates"></div></aside>
<section class="panel"><div class="heading"><h2 id="title"></h2><div><button id="rate" class="primary">Rate this skin</button> <button id="previous" aria-label="Previous candidate">←</button> <button id="next" aria-label="Next candidate">→</button></div></div>
<p class="mouse-hint">Drag the model to rotate · scroll to zoom</p>
<iframe id="viewer" title="Minecraft skin model" sandbox="allow-same-origin"></iframe>
<div class="inspect"><img id="atlas" alt="Original skin atlas"><div><label class="zoom" for="zoom">Zoom <input id="zoom" type="range" min="0.6" max="1.5" step="0.1" value="1"></label><small id="profile"></small><a id="png" download>Download original PNG</a></div></div></section>
<section class="panel ratings" id="rating-panel"><h2>Rate this skin</h2><p>Choose a score for each category.</p><p>1 = poor · 3 = mixed · 5 = strong</p><form id="scores">
<div id="criteria"></div><label for="decision">Would you use this skin?</label><select id="decision"><option value="">Not scored</option><option value="accept">Accept</option><option value="edit">Edit first</option><option value="discard">Discard</option></select>
<label for="reason">Notes <small>What works, or what needs fixing?</small></label><textarea id="reason" maxlength="2000"></textarea></form>
<label for="reviewer">Reviewer alias (optional)</label><input id="reviewer" maxlength="80" placeholder="e.g. reviewer-1"><p id="status" role="status" aria-live="polite"></p>
<label for="import">Restore a JSON backup</label><input type="file" id="import" accept="application/json,.json"><small>Only backups from this exact review packet are accepted.</small></section></main>
<footer><p>All candidates are embedded in this file. Save rated HTML to keep your scores with the viewer, or export CSV/JSON.</p><p>Browser drafts save automatically when storage is available. Missing ratings stay blank. Agree quality thresholds before unblinding results.</p></footer>
<script type="application/json" id="packet">__PACKET__</script><script>
'use strict';
const packet=JSON.parse(document.getElementById('packet').textContent);
const definitions=[['detail_1_to_5','Detail','Readable face and clothing pixels'],['coherence_1_to_5','Body coherence','Head, limbs and clothing form a character'],['palette_1_to_5','Palette','Consistent, deliberate colors'],['seams_1_to_5','Seams','Joins are intentional or unobtrusive'],['wearability_1_to_5','Wearability','Would use with little or no editing']];
const fields=definitions.map(x=>x[0]);
const empty=()=>Object.fromEntries([...fields,'accept_edit_discard','reason'].map(x=>[x,'']));
let index=0,reviewer='',scores=packet.files.map(empty);
const key='skin-review:'+packet.packet_id;
const el=id=>document.getElementById(id);
const message=text=>{el('status').textContent=text;};
function snapshot(){return {schema:'minecraft-skin-gan.review-scores/v1',packet_id:packet.packet_id,model_type:packet.model_type,alpha_mode:packet.alpha_mode,reviewer,exported_at:new Date().toISOString(),scores:packet.files.map((file,i)=>({candidate:file.candidate,sha256:file.sha256,...scores[i]}))};}
function validate(data){
 if(!data||data.schema!=='minecraft-skin-gan.review-scores/v1'||data.packet_id!==packet.packet_id||data.model_type!==packet.model_type||data.alpha_mode!==packet.alpha_mode||typeof data.reviewer!=='string'||data.reviewer.length>80||!Array.isArray(data.scores)||data.scores.length!==packet.files.length)throw Error('Backup does not match this packet.');
 const clean=data.scores.map((row,i)=>{
  const file=packet.files[i];
  if(!row||row.candidate!==file.candidate||row.sha256!==file.sha256||fields.some(k=>!['','1','2','3','4','5'].includes(row[k]))||!['','accept','edit','discard'].includes(row.accept_edit_discard)||typeof row.reason!=='string'||row.reason.length>2000)throw Error('Backup contains invalid scores.');
  return Object.fromEntries([...fields,'accept_edit_discard','reason'].map(k=>[k,row[k]]));
 });return {scores:clean,reviewer:data.reviewer};
}
function persist(){try{localStorage.setItem(key,JSON.stringify(snapshot()));message('Draft saved in this browser.');}catch{message('Browser storage unavailable. Save a JSON backup to keep scores.');}}
function complete(row){return fields.every(k=>row[k]!=='')&&row.accept_edit_discard!=='';}
function progress(){const count=scores.filter(complete).length;el('progress').textContent=`${count} / ${scores.length} scored`;document.querySelectorAll('.tile').forEach((tile,i)=>tile.classList.toggle('done',complete(scores[i])));}
function collect(){const row=scores[index];fields.forEach(k=>row[k]=el(k).value);row.accept_edit_discard=el('decision').value;row.reason=el('reason').value;reviewer=el('reviewer').value;persist();progress();}
function zoom(){try{const character=el('viewer').contentDocument.querySelector('.character');if(character){character.style.scale=el('zoom').value;character.style.transformOrigin='50% 50%';}}catch{}}
function mouseControls(){try{
 const doc=el('viewer').contentDocument,viewport=doc.querySelector('.viewport'),character=doc.querySelector('.character');
 let drag=null,yaw=0,pitch=0;
 viewport.addEventListener('pointerdown',event=>{if(event.button!==0)return;doc.getElementById('rotate').checked=false;drag={id:event.pointerId,x:event.clientX,y:event.clientY};viewport.setPointerCapture(event.pointerId);viewport.classList.add('dragging');});
 viewport.addEventListener('pointermove',event=>{if(!drag||drag.id!==event.pointerId)return;yaw+=(event.clientX-drag.x)*0.6;pitch=Math.max(-80,Math.min(80,pitch-(event.clientY-drag.y)*0.6));drag.x=event.clientX;drag.y=event.clientY;character.style.transform=`rotateX(${pitch}deg) rotateY(${yaw}deg)`;});
 const stop=event=>{if(drag&&event.pointerId===drag.id){drag=null;viewport.classList.remove('dragging');}};
 viewport.addEventListener('pointerup',stop);viewport.addEventListener('pointercancel',stop);
 for(const id of ['front','back'])doc.getElementById(id).addEventListener('change',()=>{yaw=id==='back'?180:0;pitch=0;character.style.transform='';});
 viewport.addEventListener('wheel',event=>{event.preventDefault();const control=el('zoom');control.value=String(Math.max(0.6,Math.min(1.5,Number(control.value)+(event.deltaY<0?0.1:-0.1))));zoom();},{passive:false});
 zoom();
 }catch{message('Mouse controls unavailable here. Open the viewer in Firefox.');}}
function show(next){index=next;const file=packet.files[index];el('title').textContent=`Candidate ${index+1} of ${packet.files.length}`;el('viewer').srcdoc=file.preview_html;el('atlas').src=file.texture;el('png').href=file.texture;el('png').download=file.candidate+'.png';el('previous').disabled=index===0;el('next').disabled=index===scores.length-1;fields.forEach(k=>el(k).value=scores[index][k]);el('decision').value=scores[index].accept_edit_discard;el('reason').value=scores[index].reason;document.querySelectorAll('.tile').forEach((tile,i)=>tile.setAttribute('aria-current',String(i===index)));progress();}
for(const [id,label,hint] of definitions){const row=document.createElement('div');row.className='score-row';const name=document.createElement('label');name.htmlFor=id;name.textContent=label;const select=document.createElement('select');select.id=id;for(const value of ['','1','2','3','4','5']){const option=document.createElement('option');option.value=value;option.textContent=value||'—';select.append(option);}const small=document.createElement('small');small.textContent=hint;row.append(name,select,small);el('criteria').append(row);}
packet.files.forEach((file,i)=>{const tile=document.createElement('button');tile.className='tile';tile.type='button';tile.setAttribute('aria-label','View candidate '+(i+1));const image=document.createElement('img');image.src=file.texture;image.alt='';tile.append(image,document.createTextNode(String(i+1)));tile.onclick=()=>show(i);el('candidates').append(tile);});
el('profile').textContent=packet.model_type+' model · '+packet.alpha_mode+' alpha';
el('scores').addEventListener('input',collect);el('scores').addEventListener('submit',event=>event.preventDefault());el('reviewer').addEventListener('input',collect);el('previous').onclick=()=>show(index-1);el('next').onclick=()=>show(index+1);el('zoom').oninput=zoom;el('viewer').onload=mouseControls;
el('rate').onclick=()=>{el('rating-panel').scrollIntoView({block:'start'});el(fields[0]).focus({preventScroll:true});};
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type}));const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csvCell(value){let text=String(value);if(/^[\s]*[=+@-]/.test(text))text="'"+text;return '"'+text.replaceAll('"','""')+'"';}
el('csv').onclick=()=>{const columns=['candidate',...fields,'accept_edit_discard','reason','packet_id','sha256','reviewer','model_type','alpha_mode'];const saved=snapshot();const rows=saved.scores.map(row=>({...row,packet_id:saved.packet_id,reviewer:saved.reviewer,model_type:saved.model_type,alpha_mode:saved.alpha_mode}));download('scores.csv',[columns.join(','),...rows.map(row=>columns.map(k=>csvCell(row[k])).join(','))].join('\r\n')+'\r\n','text/csv;charset=utf-8');};
el('backup').onclick=()=>download('scores-backup.json',JSON.stringify(snapshot(),null,2),'application/json');
el('save-html').onclick=()=>{
 const clone=document.documentElement.cloneNode(true);
 clone.querySelector('#criteria').replaceChildren();clone.querySelector('#candidates').replaceChildren();clone.querySelector('#viewer').removeAttribute('srcdoc');clone.querySelector('#import').value='';
 let saved=clone.querySelector('#saved-scores');if(!saved){saved=document.createElement('script');saved.type='application/json';saved.id='saved-scores';clone.querySelector('#packet').before(saved);}
 saved.textContent=JSON.stringify(snapshot()).replaceAll('<','\\u003c');
 download('skin-review-rated.html','<!doctype html>\n'+clone.outerHTML,'text/html;charset=utf-8');message('Rated HTML downloaded. Open it to continue anywhere.');
};
el('import').onchange=async event=>{try{const file=event.target.files[0];if(!file)return;if(file.size>2*1024*1024)throw Error('Backup is too large.');const clean=validate(JSON.parse(await file.text()));scores=clean.scores;reviewer=clean.reviewer;el('reviewer').value=reviewer;show(index);persist();message('Matching backup restored.');}catch(error){message(error instanceof SyntaxError?'Backup is not valid JSON.':error.message);}finally{el('import').value='';}};
let restoredAt=-1;
function restore(data,label){const clean=validate(data);const timestamp=Date.parse(data.exported_at)||0;if(timestamp>=restoredAt){scores=clean.scores;reviewer=clean.reviewer;restoredAt=timestamp;message(label);}}
try{const embedded=el('saved-scores');if(embedded)restore(JSON.parse(embedded.textContent),'Ratings restored from this HTML file.');}catch{message('Embedded scores are invalid. Restore a JSON backup.');}
try{const saved=localStorage.getItem(key);if(saved)restore(JSON.parse(saved),'Previous browser draft restored.');}catch{if(restoredAt<0)message('No usable browser draft. You can restore a JSON backup.');}
el('reviewer').value=reviewer;show(0);
</script></body></html>
"""


def create_review(
    source_directory: Path | str,
    output_directory: Path | str,
    *,
    model_type: str = "classic",
    alpha_mode: str = "original",
    seed: int = 1976,
) -> Path:
    """Create a bounded offline scoring packet; never overwrite existing outputs."""
    atlas_layout(model_type)
    if alpha_mode not in ("original", "opaque-base"):
        raise ValueError("alpha_mode must be original or opaque-base")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    source, output = Path(source_directory), Path(output_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    paths = sorted(source.glob("*.png"))
    if not 1 <= len(paths) <= 256:
        raise ValueError("Review requires 1..256 PNGs; choose a smaller batch")
    random.Random(seed).shuffle(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".review-", dir=output.parent) as directory:
        staging = Path(directory) / "packet"
        staging.mkdir()
        files, mapping = [], []
        for index, path in enumerate(paths, 1):
            if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("Review skin must be a regular PNG below 4 MiB")
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                if (
                    image.format != "PNG"
                    or image.size != (64, 64)
                    or getattr(image, "n_frames", 1) != 1
                ):
                    raise ValueError("Review requires single-frame 64x64 PNG skins")
                image.load()
            candidate = f"candidate-{index:03d}"
            (staging / f"{candidate}.png").write_bytes(data)
            create_preview(
                staging / f"{candidate}.png",
                staging / f"{candidate}.html",
                model_type=model_type,
                alpha_mode=alpha_mode,
            )
            preview_path = staging / f"{candidate}.html"
            preview_path.write_text(
                preview_path.read_text().replace(
                    "</style>",
                    "body{margin:12px;max-width:none;font-size:14px}"
                    "h1,p{display:none}label{padding:6px}"
                    ".viewport{height:390px;background:#fff;cursor:grab;touch-action:none}"
                    ".viewport.dragging{cursor:grabbing}</style>",
                    1,
                )
            )
            digest = hashlib.sha256(data).hexdigest()
            files.append({"candidate": candidate, "sha256": digest})
            mapping.append({"candidate": candidate, "source_name": path.name, "sha256": digest})
        manifest = {
            "schema": "minecraft-skin-gan.review/v1",
            "model_type": model_type,
            "alpha_mode": alpha_mode,
            "shuffle_seed": seed,
            "files": files,
        }
        packet_id = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest["packet_id"] = packet_id
        (staging / "review.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "candidate-map.json").write_text(json.dumps(mapping, indent=2) + "\n")
        page_files = [
            {
                **file,
                "preview": file["candidate"] + ".html",
                "preview_html": (staging / (file["candidate"] + ".html")).read_text(),
                "texture": "data:image/png;base64,"
                + base64.b64encode((staging / (file["candidate"] + ".png")).read_bytes()).decode(),
            }
            for file in files
        ]
        payload = json.dumps({**manifest, "files": page_files}, separators=(",", ":"))
        (staging / "index.html").write_text(
            _PAGE.replace("__PACKET__", payload.replace("<", "\\u003c"))
        )
        output.mkdir()
        try:
            staging.replace(output)
        except BaseException:
            output.rmdir()
            raise
    return output / "index.html"
