"""Build stamp for every published map page.

`stamp_html` writes a `<meta name="build-id">` plus a small guard script into a map page. On load (and when the tab becomes visible again) the guard
fetches the small sidecar `<page>.build` (written next to the page, `cache: 'no-store'`) and compares the build id: when the published page is newer than the one running, it
deletes the Cache Storage / service workers, re-downloads the page bypassing the HTTP cache and reloads once. The stamp is idempotent (a second call
replaces the first), so it can be applied to freshly built pages and to already deployed ones.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

_STAMP_RE = re.compile(r"<!--build-stamp-->.*?<!--/build-stamp-->\n", re.S)

GUARD_JS = r"""(function(){
  var m=document.querySelector('meta[name="build-id"]'), f=document.querySelector('meta[name="build-file"]'); if(!m||!f||location.protocol==='file:') return; var B=m.content;
  var url=new URL(f.content+'.build',document.baseURI).href;
  function purge(){var p=[];try{if(window.caches)p.push(caches.keys().then(function(k){return Promise.all(k.map(function(x){return caches.delete(x);}));}));}catch(e){}
    try{if(navigator.serviceWorker)p.push(navigator.serviceWorker.getRegistrations().then(function(r){return Promise.all(r.map(function(x){return x.unregister();}));}));}catch(e){}
    return Promise.all(p).catch(function(){});}
  var busy=false;
  function check(){
    if(busy)return; busy=true;
    fetch(url,{cache:'no-store'}).then(function(r){return r.ok?r.text():'';}).then(function(t){
      busy=false; t=t.trim(); if(!/^[0-9a-f]{12}$/.test(t)||t===B)return;
      var key='map_reload_'+f.content+location.pathname, tried=null; try{tried=sessionStorage.getItem(key);}catch(e){}
      if(tried===t)return;   // already reloaded once for this build: never loop
      try{sessionStorage.setItem(key,t);}catch(e){}
      purge().then(function(){location.reload();});   // a reload revalidates the page itself; no second download of the (large) page
    }).catch(function(){busy=false;});
  }
  check(); document.addEventListener('visibilitychange',function(){if(!document.hidden)check();});
})();"""


ATTR_JS = r"""(function(){
  // the (i) attribution button of every MapLibre map starts closed, until the visitor opens it; its text carries the map version date and the author.
  // Only the attribution controls themselves are watched (never the whole page), and only for a few seconds.
  var dm=document.querySelector('meta[name="build-date"]'), bm=document.querySelector('meta[name="build-id"]'),
      INFO='Map version '+(dm?dm.content:'')+' (build '+(bm?bm.content:'')+') · Map by Miguel Ureña Pliego', touched=false, seen=[];
  document.addEventListener('click',function(ev){ if(ev.target.closest&&ev.target.closest('.maplibregl-ctrl-attrib-button')) touched=true; },true);
  function info(el){ var i=el.querySelector('.maplibregl-ctrl-attrib-inner'); if(i&&!i.querySelector('.map-author-info')){ var s=document.createElement('div'); s.className='map-author-info'; s.style.cssText='margin-top:2px;color:#444'; s.textContent=INFO; i.appendChild(s); } }
  function close(el){ if(!touched){ el.classList.remove('maplibregl-compact-show'); el.removeAttribute('open'); } }
  function attach(el){
    if(seen.indexOf(el)>=0) return; seen.push(el); info(el); close(el); var t0=Date.now(), busy=false;
    new MutationObserver(function(){ if(busy) return; busy=true; setTimeout(function(){ busy=false; info(el); if(Date.now()-t0<20000) close(el); },50); }).observe(el,{attributes:true,attributeFilter:['class','open'],childList:true,subtree:true});
  }
  var n=0,t=setInterval(function(){ [].forEach.call(document.getElementsByClassName('maplibregl-ctrl-attrib'),attach); if(++n>60) clearInterval(t); },500);
})();"""

FETCH_JS = r"""(function(){
  // every same-origin data request of the page carries the page's build id (?v=...): a new build means new URLs, so cached data of an older build is never used
  var m=document.querySelector('meta[name="build-id"]'), of=window.fetch; if(!m||!of||location.protocol==='file:') return; var V=m.content;
  function ver(u){ try{ var x=new URL(u,document.baseURI); if(x.origin!==location.origin||/\.(html|build)$/.test(x.pathname)||x.searchParams.has('v')) return null; x.searchParams.set('v',V); return x.href; }catch(e){ return null; } }
  window.fetch=function(i,o){ try{ if(typeof i==='string'||i instanceof URL){ var n=ver(i); if(n) i=n; } else if(i&&i.url){ var n2=ver(i.url); if(n2) i=new Request(n2,i); } }catch(e){} return of.call(this,i,o); };
})();"""

STYLE_JS = r"""(function(){
  // the hexagons / circles / census / none radio buttons of the layer panel become one dropdown (the radios stay in the page, hidden, and keep driving the map)
  function build(){
    if(document.getElementById('baseGroupSelect')) return true;
    var radios=[].slice.call(document.querySelectorAll('input[type=radio][name=__base_group]')); if(!radios.length) return false;
    var sel=document.createElement('select'); sel.id='baseGroupSelect'; sel.style.cssText='width:100%;font:11px sans-serif;margin:2px 0;';
    radios.forEach(function(r){ var o=document.createElement('option'); o.value=r.value; o.textContent=r.parentElement.textContent.trim(); sel.appendChild(o); });
    var first=radios[0].closest('label')||radios[0]; first.parentNode.insertBefore(sel,first);
    radios.forEach(function(r){ var l=r.closest('label')||r; l.style.display='none'; if(l.nextSibling&&l.nextSibling.tagName==='BR') l.nextSibling.style.display='none'; });
    function fromRadios(e){ if(e&&e.target===sel) return; var c=radios.filter(function(r){return r.checked;})[0]; if(c&&sel.value!==c.value) sel.value=c.value; }
    sel.addEventListener('change',function(){ var r=radios.filter(function(x){return x.value===sel.value;})[0]; if(r&&!r.checked) r.click(); });
    document.addEventListener('change',fromRadios,true); setInterval(fromRadios,1000); fromRadios(); return true;
  }
  function go(){ if(build()) return; var n=0,t=setInterval(function(){ if(build()||++n>60) clearInterval(t); },500); }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',go); else go();
})();"""

DARK_JS = r"""(function(){
  // legend polish (compact chips, bigger colour bars) and dark mode (button in the legend header, URL item ?dark=1 / theme=dark)
  var CSS=''
   +'#layer-switcher-body label{font-size:11px;}'
   +'#layer-switcher-body label:has(> input[type=checkbox]){display:inline-flex;align-items:center;gap:3px;margin:1px 3px 1px 0;padding:1px 7px 1px 4px;border:1px solid #dfe3e8;border-radius:11px;background:#f6f8fa;cursor:pointer;}'
   +'#layer-switcher-body label:has(> input[type=checkbox]) + br{display:none;}'
   +'#layer-switcher-body input[type=checkbox]{margin:0;accent-color:#2b6cb0;}'
   +'#layer-switcher-body input[type=range]{width:100%;height:3px;margin:3px 0 7px;accent-color:#2b6cb0;cursor:pointer;}'
   +'#layer-switcher{width:min(252px,92vw) !important;overflow-x:hidden !important;box-sizing:border-box;}#layer-switcher-body{width:auto !important;max-width:100% !important;}#mapLegend{font-size:12px !important;width:auto !important;max-width:100% !important;box-sizing:border-box;}#layer-switcher select,#layer-switcher input[type=range]{max-width:100%;box-sizing:border-box;}'
   +'#mapLegend div[style*="linear-gradient"]{height:12px !important;border-radius:3px !important;}'
   +'#mapLegend [style*="font-size:8.5px"]{font-size:10px !important;}'
   +'#mapDarkBtn{border:1px solid #c9ced6;background:#fff;border-radius:6px;cursor:pointer;font:12px sans-serif;padding:0 5px;margin-right:4px;height:22px;line-height:20px;}'
   +'#placeRankList .colInfoBtn{display:none !important}'
   +'#legendAreaRow .__csel-btn{border:1px solid #9aa5b1 !important;border-radius:5px !important;background:#fff !important;padding:2px 22px 2px 8px !important;cursor:pointer;box-shadow:0 1px 2px rgba(0,0,0,.12);}#legendAreaRow .__csel-btn:hover{border-color:#2b6cb0 !important;}html.mapdark #legendAreaRow .__csel-btn{background:#2a2f37 !important;color:#e8e8e8;border-color:#5b6572 !important;}'
   +'#statsPanel{overflow:hidden !important;height:min(calc(100vh - 150px),780px) !important;}'
   +'#statsPanel[style*="display: block"],#statsPanel[style*="display:block"]{display:flex !important;flex-direction:column;}'
   +'#statsPanel > *{flex:0 0 auto;}'
   +'#statsPanel > div[id^=statsTab_][style*="block"]{display:flex !important;flex-direction:column;flex:1 1 auto !important;min-height:0;}'
   +'#statsPanel > div[id^=statsTab_] > *{flex:0 0 auto;}'
   +'#placeRankList,#distBars,#regR2Bars,#anovaBars{flex:1 1 auto !important;min-height:90px;overflow-y:auto !important;overflow-x:hidden !important;}#statsPanel{width:min(480px,94vw);}'
   +'#mapDarkBtn.pill{border:none;background:transparent;font-size:13px;line-height:1;height:auto;padding:0 0 0 8px;margin:0 0 0 8px;border-left:1px solid #ddd;border-radius:0;opacity:.75;}#mapDarkBtn.pill:hover{opacity:1;}'
   +'html.mapdark #citySelectorBar,html.mapdark #cityDropdown{background:#23272e;color:#e8e8e8;}html.mapdark #cityDropdown .opt{border-color:#3a3f48;}html.mapdark #cityDropdown .opt:hover,html.mapdark #cityDropdown .opt.active{background:#2f3540;}'
   +'#statsButton,#downloadButton{bottom:12px !important;}#statsPanel{bottom:58px !important;}#downloadPanel{bottom:12px !important;}'
   +'#mapDarkBtn.fab{position:fixed;left:98px;bottom:10px;z-index:99999;width:34px;height:34px;border-radius:50%;border:none;box-shadow:0 1px 5px rgba(0,0,0,.35);font-size:16px;line-height:34px;padding:0;}'
   +'html.mapdark #layer-switcher,html.mapdark #statsPanel,html.mapdark #top-center-bar,html.mapdark #downloadPanel,html.mapdark .maplibregl-popup-content,html.mapdark .maplibregl-ctrl-group,html.mapdark #colInfoPanel,html.mapdark #editSummaryBar,html.mapdark #route-toolbar,html.mapdark #statsButton,html.mapdark #mapDarkBtn.fab,html.mapdark #cityDropdown,html.mapdark #routeEditToggleBtn,html.mapdark button[title*="ownload"],html.mapdark #downloadBtn{filter:invert(.92) hue-rotate(180deg);}'
   +'html.mapdark #mapLegend [style*="linear-gradient"],html.mapdark #mapLegend [style*="rgba(80,80,80"]{filter:invert(1) hue-rotate(180deg);}'
   +'html.mapdark #mapLegend [id^=circleLegendC]{filter:invert(1);}'
   +'html.mapdark body,html.mapdark #overviewMap,html.mapdark #overviewMap .maplibregl-canvas-container{background:#0b0e14 !important;}'
   +'html.mapdark-full{filter:invert(.92) hue-rotate(180deg);background:#fff;}';
  var st=document.createElement('style'); st.textContent=CSS;
  function qs(s){ return document.querySelector(s); }
  function isDark(){ return document.documentElement.classList.contains('mapdark'); }
  function group(a){ return (a==='official'||a==='municipality')?'official':'metro'; }
  // ---- legend ----
  function textFix(root){
    var w=document.createTreeWalker(root,NodeFilter.SHOW_TEXT,null),n,hits=[];
    while((n=w.nextNode())) if(/\(per stop\)|\(raster basemaps only\)/.test(n.nodeValue)) hits.push(n);
    hits.forEach(function(t){ t.nodeValue=t.nodeValue.replace(/\s*\(per stop\)/,'').replace(/\s*\(raster basemaps only\)/,''); });
  }
  function fixLegend(){
    var body=qs('#layer-switcher-body'); if(!body) return false;
    textFix(body);
    var sel=qs('#baseGroupSelect'); if(sel){ var h=sel.previousElementSibling; if(h&&h.tagName==='HR') h.style.display='none'; var h2=sel.nextElementSibling; if(h2&&h2.tagName==='HR') h2.style.display='none'; sel.style.cssText+=';border:1px solid #c9ced6;border-radius:4px;'; }
    var arow=qs('#legendAreaRow'), shp=qs('#baseGroupSelect');   // the Area dropdown sits just above the circles / hexagons / census dropdown
    if(arow&&shp&&!arow.dataset.moved){ var anc=shp.closest('.__csel-wrap')||shp; if(anc.parentNode){ anc.parentNode.insertBefore(arow,anc); arow.dataset.moved='1'; arow.style.margin='5px 0 3px 0'; } }
    if(arow){ var ph=arow.previousElementSibling; if(ph&&ph.tagName==='HR') ph.style.display='none'; }   // no rule above the Area row or around the shape dropdown
    if(shp){ var an2=shp.closest('.__csel-wrap')||shp; [an2.nextElementSibling,an2.previousElementSibling].forEach(function(h){ if(h&&h.tagName==='HR') h.style.display='none'; }); }
    var ar=qs('#legendAreaSelect');
    if(ar && ar.options.length>2 && !ar.dataset.two){   // only "UN" or "Official": the stats panel keeps all four
      ar.dataset.two='1'; ar.innerHTML='<option value="metro">UN</option><option value="official">Official</option>';
      ar.value=group(window.__currentArea||'metro');
      ar.title='UN or official analysis area: sets the stats-panel Area, the top score and the mask';
      setInterval(function(){ var g=group(window.__currentArea||'metro'); if(ar.value!==g && document.activeElement!==ar) ar.value=g; },1000);
    }
    return true;
  }
  // ---- dark mode ----
  function basemap(v){
    var bs=qs('#basemapSelect'); if(!bs) return false;
    var has=[].some.call(bs.options,function(o){return o.value==='openfreemap_dark';}); if(!has) return false;
    if(v){ if(bs.value!=='openfreemap_dark'){ bs.dataset.prev=bs.value; bs.value='openfreemap_dark'; bs.dispatchEvent(new Event('change',{bubbles:true})); } }
    else if(bs.dataset.prev){ bs.value=bs.dataset.prev; delete bs.dataset.prev; bs.dispatchEvent(new Event('change',{bubbles:true})); }
    return true;
  }
  function overview(v){
    var m=window.__overviewMap; if(!m||!m.setStyle) return;
    m.setStyle(v?'https://tiles.openfreemap.org/styles/dark':'https://tiles.openfreemap.org/styles/positron');
    m.once('style.load',function(){ try{ m.setProjection({type:'globe'}); }catch(e){} try{ m.setSky(v?{'sky-color':'#0b0e14','horizon-color':'#0b0e14','fog-color':'#0b0e14','fog-ground-blend':0,'horizon-fog-blend':0,'sky-horizon-blend':0,'atmosphere-blend':0}:{'atmosphere-blend':['interpolate',['linear'],['zoom'],0,1,5,1,7,0]}); }catch(e){} });
  }
  function url(v){ try{ var u=new URL(location.href); if(v) u.searchParams.set('dark','1'); else { u.searchParams.delete('dark'); u.searchParams.delete('theme'); } history.replaceState(null,'',u.href); }catch(e){} }
  function paint(){ var b=qs('#mapDarkBtn'); if(b){ b.textContent=isDark()?'☀️':'🌙'; b.title=isDark()?'Light mode':'Dark mode'; } }
  function set(v,fromParent){
    window.__mapDark=v; document.documentElement.classList.toggle('mapdark',v);
    var hasBm=basemap(v); document.documentElement.classList.toggle('mapdark-full',v&&!hasBm&&!qs('#layer-switcher')&&!window.__overviewMap);
    overview(v); paint(); if(!fromParent) url(v);
    if(!fromParent&&window.parent!==window){ try{ window.parent.__setDark(v); }catch(e){} }
    [].forEach.call(document.querySelectorAll('iframe'),function(f){ try{ if(f.contentWindow.__setDark) f.contentWindow.__setDark(v); }catch(e){} });
  }
  window.__setDark=function(v){ set(v,true); };
  function wantDark(){ try{ var q=new URLSearchParams(location.search); if(q.get('dark')==='1'||q.get('theme')==='dark') return true; }catch(e){} try{ if(window.parent!==window&&window.parent.__mapDark) return true; }catch(e){} return false; }
  var inFrame=window.parent!==window;
  function placeButton(){
    if(qs('#mapDarkBtn')) return;
    var head=qs('#layer-switcher-head'), b=document.createElement('button'); b.id='mapDarkBtn'; b.type='button'; b.addEventListener('click',function(){ set(!isDark(),false); });
    var pill=qs('#citySelectorBar');
    if(head){ head.insertBefore(b,head.querySelector('#layer-switcher-toggle')||null); }
    else if(!inFrame&&pill){ b.className='pill'; b.addEventListener('click',function(e){ e.stopPropagation(); }); pill.appendChild(b); }
    else if(!inFrame){ b.className='fab'; document.body.appendChild(b); }
    else return;
    paint();
  }
  function init(){
    document.head.appendChild(st);
    setInterval(function(){ var f=qs('#cityIframe'), b=qs('#mapDarkBtn.fab'); if(!f||!b) return; var cs=getComputedStyle(f); var vis=f.offsetWidth>0&&f.offsetHeight>0&&cs.display!=='none'&&cs.visibility!=='hidden'; b.style.display=vis?'none':''; },700);   // a city is open: its legend carries the button
    var n=0,t=setInterval(function(){ placeButton(); var ok=fixLegend(); if(isDark()) basemap(true); if(window.__dkInit===undefined && wantDark()){ window.__dkInit=1; set(true,true); } else if(window.__dkInit===undefined) window.__dkInit=0; if((ok&&qs('#mapDarkBtn')&&(!wantDark()||qs('#basemapSelect')))||++n>80) clearInterval(t); },500);
  }
  try{ if(wantDark()) document.documentElement.classList.add('mapdark'); }catch(e){}
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init); else init();
})();"""

STAMP_VER = 'data-stamp="23"'


def build_id(html: str) -> str:
    """Content hash of the page without its stamp: unchanged pages keep their id, any change gives a new one.

    Args:
        html: Page HTML.

    Returns:
        12-character id.
    """
    return hashlib.sha1(_STAMP_RE.sub("", html).encode("utf-8")).hexdigest()[:12]


META_JS = r"""(function(){
  // Metadata tab, compact: one line per column (name, source); a click opens the rest (the table's other cells under their header names).
  // Built from whatever table the page carries (static rows, or rows a city map fills in later), so every map gets the same view.
  var CSS='#statsTab_metadata > div:not(#metaCompact){display:none !important;}'
   +'#statsPanel #statsTab_metadata > #metaCompact{display:flex !important;flex-direction:column;flex:1 1 auto !important;min-height:0;gap:6px;}'
   +'#metaSearch{box-sizing:border-box;width:100%;padding:4px 6px;font:12px sans-serif;}'
   +'#statsPanel #metaList{flex:1 1 auto !important;min-height:0;overflow-y:auto;border-top:1px solid #e3e6ea;}'
   +'.mc-row{border-bottom:1px solid #eef0f3;padding:4px 2px;cursor:pointer;font:12px sans-serif;}'
   +'.mc-row:hover{background:#f6f8fa;}'
   +'.mc-top{display:flex;gap:8px;align-items:baseline;}'
   +'.mc-name{flex:1 1 auto;min-width:0;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}'
   +'.mc-src{flex:0 1 45%;min-width:0;color:#667;font-size:11px;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}'
   +'.mc-det{display:none;margin:4px 0 2px;font-size:11px;color:#333;}'
   +'.mc-row.open .mc-det{display:grid;grid-template-columns:max-content 1fr;gap:2px 8px;}'
   +'.mc-row.open .mc-name,.mc-row.open .mc-src{white-space:normal;}'
   +'.mc-k{color:#778;}.mc-d{grid-column:1 / -1;margin-top:3px;color:#444;line-height:1.35;}';
  var st=document.createElement('style'); st.textContent=CSS; (document.head||document.documentElement).appendChild(st);
  function esc(t){ return String(t).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];}); }
  var sig=null;
  function build(){
    var tab=document.getElementById('statsTab_metadata'); if(!tab) return;
    var table=tab.querySelector('table'); if(!table) return;
    var heads=[].map.call(table.querySelectorAll('thead th'),function(th){return th.textContent.trim();});
    var trs=[].slice.call(table.querySelectorAll('tbody tr'));
    var s=trs.length+':'+(trs[0]?trs[0].textContent.length:0)+':'+(trs.length?trs[trs.length-1].textContent.length:0);
    var box=document.getElementById('metaCompact');
    if(box && s===sig) return; sig=s;
    if(!box){ box=document.createElement('div'); box.id='metaCompact'; box.innerHTML='<input id="metaSearch" type="search" placeholder="Search..." autocomplete="off"><div id="metaList"></div>'; tab.insertBefore(box,tab.firstChild);
      document.getElementById('metaSearch').addEventListener('input',filter);
      document.getElementById('metaList').addEventListener('click',function(e){ var r=e.target.closest('.mc-row'); if(r) r.classList.toggle('open'); }); }
    var iSrc=heads.indexOf('Source'), iDesc=heads.indexOf('Description'), h='';
    trs.forEach(function(tr){
      var td=[].map.call(tr.children,function(c){return c.textContent.trim();});
      if(td.length<2) return;
      var det='';
      td.forEach(function(v,i){ if(i===0||i===iSrc||i===iDesc||!v||v==='--'||v==='/'||v===' / ') return; det+='<span class="mc-k">'+esc(heads[i]||'')+'</span><span>'+esc(v)+'</span>'; });
      if(iDesc>=0&&td[iDesc]&&td[iDesc]!=='--') det+='<div class="mc-d">'+esc(td[iDesc])+'</div>';
      h+='<div class="mc-row" data-k="'+esc((td[0]+' '+(td[iSrc]||'')).toLowerCase())+'"><div class="mc-top"><span class="mc-name">'+esc(td[0])+'</span><span class="mc-src">'+esc(iSrc>=0?td[iSrc]:'')+'</span></div><div class="mc-det">'+det+'</div></div>';
    });
    document.getElementById('metaList').innerHTML=h; filter();
  }
  function filter(){ var q=(document.getElementById('metaSearch')||{}).value; q=(q||'').toLowerCase().trim();
    [].forEach.call(document.querySelectorAll('#metaList .mc-row'),function(r){ r.style.display=(!q||r.getAttribute('data-k').indexOf(q)>=0)?'':'none'; }); }
  function watch(){
    var tab=document.getElementById('statsTab_metadata'); if(!tab) return false;
    build(); var body=document.getElementById('metadataTabBody');
    new MutationObserver(function(){ build(); }).observe(body||tab,{childList:true,subtree:!body});
    return true;
  }
  function go(){ if(watch()) return; var n=0,t=setInterval(function(){ if(watch()||++n>60) clearInterval(t); },500); }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',go); else go();
})();"""


def stamp_html(html: str, name: str = "map.html", date: str | None = None) -> tuple[str, str]:
    """`(stamped html, build id)`: the stamp sits right after `<head>` and replaces an earlier one.

    Args:
        html: Page HTML.
        name: File name written in the stamp.
        date: Build date; ``None`` = today.
    """
    html = _STAMP_RE.sub("", html)
    bid = build_id(html)
    date = date or time.strftime("%Y-%m-%d")
    block = f'<!--build-stamp--><meta name="build-id" content="{bid}"><meta name="build-file" content="{name}"><meta name="build-date" content="{date}"><script {STAMP_VER}>{FETCH_JS}{GUARD_JS}</script><script>{ATTR_JS}</script><script>{STYLE_JS}</script><script>{DARK_JS}</script><script>{META_JS}</script><!--/build-stamp-->\n'
    m = re.search(r"<head[^>]*>", html, re.I)
    return (html[: m.end()] + block + html[m.end():] if m else block + html), bid


_DATA_MARK = re.compile(r'<!--data:[0-9a-f]{12}-->')


def data_marker(html: str, page_path) -> str:
    """Puts a hash of the page's small data files (`results/*.json|geojson|js`: stats, area outlines and masks, vehicle-km, column info ...) into the page as a comment, so the
    build id changes whenever that data changes even when the page itself does not. Data requested with `?v=<build id>` is cached for a year: without this a changed data file
    could be shadowed by an older copy cached under an unchanged id. Pages without a `results` folder are left alone.

    Args:
        html: Page HTML.
        page_path: Path of the page.

    Returns:
        The HTML with the data marker.
    """
    parent = Path(page_path).parent
    dirs = [parent / "results"] if (parent / "results").is_dir() else sorted(d for d in parent.glob("*/results") if d.is_dir())   # a wrapper page (all cities) reads every city's results
    if not dirs:
        return html
    h = hashlib.sha1()
    for res in dirs:
        for f in sorted(res.iterdir()):
            if f.is_file() and f.suffix in (".json", ".geojson", ".js"):
                h.update((res.parent.name + "/" + f.name).encode()); h.update(hashlib.sha1(f.read_bytes()).digest())
    mark = f"<!--data:{h.hexdigest()[:12]}-->"
    if mark in html:
        return html
    html = _DATA_MARK.sub("", html)
    m = re.search(r"</head>", html, re.I)
    return html[: m.start()] + mark + html[m.start():] if m else html + mark


_STATS_INLINE = re.compile(r'<script>\s*window\.__statsData = (?=\{)')


def externalize_stats(html: str, page_path) -> str:
    """Moves the inline `window.__statsData = {...}` (tens of MB on the big cities) to `results/stats_data.js`, loaded by a plain script tag whose URL carries the content hash:
    the page shrinks, the browser fetches the data in parallel and keeps it (immutable) across visits. Pages without the inline block, or without a results folder, are unchanged.

    Args:
        html: Page HTML.
        page_path: Path of the page.

    Returns:
        The HTML (shorter when the data were moved).
    """
    m = _STATS_INLINE.search(html)
    res = Path(page_path).parent / "results"
    if not m or not res.is_dir():
        return html
    import json
    try:
        _, n = json.JSONDecoder().raw_decode(html[m.end():])
    except ValueError:
        return html
    end = m.end() + n
    tail = re.match(r';\s*</script>', html[end:])
    if not tail:
        return html
    body = "window.__statsData = " + html[m.end():end] + ";\n"
    h = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    (res / "stats_data.js").write_text(body, encoding="utf-8")
    return html[:m.start()] + f'<script src="results/stats_data.js?v={h}"></script>' + html[end + tail.end():]


def write_stamped(path, html: str) -> str:
    """Writes the stamped page and its `<page>.build` sidecar (the build id, fetched by the guard); returns the id.

    Args:
        path: Page path.
        html: Page HTML.
    """
    p = Path(path)
    out, bid = stamp_html(data_marker(externalize_stats(html, p), p), p.name)
    p.write_text(out, encoding="utf-8")
    Path(str(p) + ".build").write_text(bid, encoding="ascii")
    return bid


def stamp_file(path) -> str:
    """Stamps an already written page in place (idempotent); rewrites nothing when the page and its sidecar are current.

    Args:
        path: Page path.

    Returns:
        The build id.
    """
    p = Path(path); sc = Path(str(p) + ".build")
    html = p.read_text(encoding="utf-8")
    ext = data_marker(externalize_stats(html, p), p)
    if ext != html:
        tmp = Path(str(p) + ".ext"); tmp.write_text(ext, encoding="utf-8"); os.replace(tmp, p); html = ext
    cur = re.search(r'<meta name="build-id" content="([0-9a-f]{12})"', html)
    bid = build_id(html)
    if not (cur and cur.group(1) == bid and f'<meta name="build-file" content="{p.name}">' in html and STAMP_VER in html):
        mt = p.stat().st_mtime_ns
        dm = re.search(r'<meta name="build-date" content="([0-9-]{10})"', html)
        out, bid = stamp_html(html, p.name, dm.group(1) if dm and cur and cur.group(1) == bid else None)
        if p.stat().st_mtime_ns != mt:   # replaced while stamping: leave the newer file for the next pass
            return ""
        tmp = Path(str(p) + ".stamp"); tmp.write_text(out, encoding="utf-8"); os.replace(tmp, p)
    if not sc.exists() or sc.read_text().strip() != bid:
        sc.write_text(bid, encoding="ascii")
    return bid
