"""Presentation mode of a MapLibre page, driven by URL parameters (and a "copy presentation link" button in the normal view).

    ?present=1                 larger text, minimal chrome, legend reduced to the colour map, bigger stats plots, no scrolling, simplified census
    shape=circles|hexagons|census|none   base layer           badges=1   route badges per stop (+stops)
    z=<zoom> lat=<lat> lng=<lng>         camera               circle=<column>   circle size column
    tab=distribution|regression|census|discretization|metadata   opens the stats panel on that tab
    col=<column>               column of the opened tab (Distribute by / regression field)
    opacity=<column> alpha=<0..1>        opacity-by column and the flat shape opacity
    state=<name>               scope (US / Spain pages, handled by the scope panel)
Every parameter works without `present=1` too (a shareable view); `present=1` only changes the chrome.
"""

PRESENTATION_JS = r"""
(function() {
  var P = new URLSearchParams(location.search);
  var present = P.get('present') === '1' || P.get('present') === 'true';
  var st = document.createElement('style');
  st.textContent =
    'body.presentation #statsButton,body.presentation #downloadButton,body.presentation #routeEditToggleBtn,body.presentation #route-toolbar,' +
    'body.presentation #top-center-bar,body.presentation #editSummaryBar,body.presentation #addressSearch,body.presentation #layer-switcher-body,' +
    'body.presentation #layer-switcher-head,body.presentation #circleLegend,body.presentation #opacityLegend,body.presentation #devLegend,' +
    'body.presentation #statsAreaBtnRow,body.presentation #distScenarioWrap,body.presentation #regScenarioWrap,body.presentation #anovaScenarioWrap,' +
    'body.presentation #distAreaWrap,body.presentation #regAreaWrap,body.presentation #anovaAreaWrap,body.presentation [id$=ComparePopover],' +
    'body.presentation label:has(> [id$=CompareEnable]),body.presentation #extCensusRow,body.presentation .statsTabBtn:not(.active),' +
    'body.presentation #distAggSelect,body.presentation .maplibregl-ctrl-top-left,body.presentation #copyPresentationLink{display:none!important}' +
    'body.presentation #layer-switcher{padding:10px 14px;zoom:1.35}body.presentation #legend-slot{display:block!important}' +
    'body.presentation #statsPanel{zoom:1.25;overflow:hidden!important;width:min(560px,46vw)!important;height:calc(78vh/1.25)!important;font-size:13px}' +
    'body.presentation #statsPanel *{scrollbar-width:none}body.presentation{overflow:hidden}' +
    'body.presentation #regR2Bars > div:nth-child(n+5),body.presentation #statsTab_anova > div > div:nth-child(n+9){display:none!important}' +
    '#copyPresentationLink{display:block;margin:6px 0 2px;font:11px sans-serif;border:1px solid #ccc;background:#f4f4f4;border-radius:3px;padding:2px 6px;cursor:pointer}';
  document.head.appendChild(st);
  if (present) document.body.classList.add('presentation');

  function until(test, fn, tries) {
    var n = 0, t = setInterval(function() {
      var r = null; try { r = test(); } catch (e) {}
      if (r) { clearInterval(t); try { fn(r); } catch (e) { console.warn('presentation', e); } }
      else if (++n > (tries || 60)) clearInterval(t);
    }, 400);
  }
  function setSel(id, v) {
    var s = document.getElementById(id); if (!s) return false;
    if (!Array.prototype.some.call(s.options, function(o) { return o.value === v; })) return false;
    s.value = v; s.dispatchEvent(new Event('change', {bubbles: true})); return true;
  }
  var shape = P.get('shape');
  if (shape) until(function() { return document.querySelector('input[name=__base_group][value="' + shape + '"]'); }, function(r) { r.click(); });
  if (P.get('badges') === '1') until(function() { return document.getElementById('ovStopRouteBadgesCheckbox'); }, function(c) {
    var s = document.getElementById('ovStopsCheckbox'); if (s && !s.checked) s.click();
    if (!c.checked) c.click();
  });
  if (P.get('z') || P.get('lat') || P.get('lng')) until(function() { return window.__mainMap; }, function(m) {
    var c = m.getCenter(), o = {center: [parseFloat(P.get('lng') || c.lng), parseFloat(P.get('lat') || c.lat)]};
    if (P.get('z')) o.zoom = parseFloat(P.get('z'));
    m.jumpTo(o); setTimeout(function() { m.jumpTo(o); }, 3500); setTimeout(function() { m.jumpTo(o); }, 7000);   // the page's own start-up fit may run after the first jump
  });
  if (P.get('circle')) until(function() { return document.getElementById('circleFieldSelect'); }, function() { setSel('circleFieldSelect', P.get('circle')); });
  if (P.get('opacity')) until(function() { return document.getElementById('opacitySelect'); }, function() { setSel('opacitySelect', P.get('opacity')); });
  if (P.get('alpha') != null && P.get('alpha') !== '') until(function() { return document.getElementById('shapeOpacitySlider'); }, function(sl) {
    sl.value = Math.round(parseFloat(P.get('alpha')) * 100); sl.dispatchEvent(new Event('input', {bubbles: true}));
  });
  var tab = P.get('tab'); if (tab === 'census') tab = 'anova';
  if (tab) until(function() { return document.querySelector('#statsPanel .statsTabBtn[data-tab="' + tab + '"]'); }, function(b) {
    var panel = document.getElementById('statsPanel');
    if (panel.style.display === 'none') document.getElementById('statsButton').click();
    b.click();
    var col = P.get('col');
    if (col) setTimeout(function() { setSel(tab === 'regression' ? 'regFieldSelect' : 'distMainSelect', col); }, 600);
  }, 90);

  // ---- "copy presentation link" ----
  function currentLink() {
    var u = new URL(location.href), q = u.searchParams; q.set('present', '1');
    var g = document.querySelector('input[name=__base_group]:checked'); if (g) q.set('shape', g.value);
    var b = document.getElementById('ovStopRouteBadgesCheckbox'); if (b && b.checked) q.set('badges', '1'); else q.delete('badges');
    var m = window.__mainMap; if (m) { var c = m.getCenter(); q.set('z', m.getZoom().toFixed(2)); q.set('lat', c.lat.toFixed(5)); q.set('lng', c.lng.toFixed(5)); }
    var cs = document.getElementById('circleFieldSelect'); if (cs && g && g.value === 'circles') q.set('circle', cs.value);
    var os = document.getElementById('opacitySelect'); if (os && os.value !== 'none') q.set('opacity', os.value); else q.delete('opacity');
    var sl = document.getElementById('shapeOpacitySlider'); if (sl) q.set('alpha', (sl.value / 100).toFixed(2));
    var sp = document.getElementById('statsPanel');
    if (sp && sp.style.display !== 'none') {
      var tb = sp.querySelector('.statsTabBtn.active'); if (tb) q.set('tab', tb.getAttribute('data-tab'));
      var sel = document.getElementById(q.get('tab') === 'regression' ? 'regFieldSelect' : 'distMainSelect'); if (sel && sel.value) q.set('col', sel.value);
    } else { q.delete('tab'); q.delete('col'); }
    return u.toString();
  }
  window.__presentationLink = currentLink;
  if (!present) until(function() { return document.getElementById('layer-switcher-body'); }, function(body) {
    var btn = document.createElement('button'); btn.id = 'copyPresentationLink'; btn.type = 'button'; btn.textContent = 'Copy presentation link';
    btn.addEventListener('click', function() {
      var link = currentLink();
      var done = function() { btn.textContent = 'Link copied'; setTimeout(function() { btn.textContent = 'Copy presentation link'; }, 1800); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(link).then(done, function() { window.prompt('Presentation link', link); });
      else window.prompt('Presentation link', link);
    });
    body.appendChild(btn);
  });
})();
"""


def presentation_script() -> str:
    """The presentation-mode script of the maps.

    Returns:
        ``<script>`` block.
    """
    return "<script>" + PRESENTATION_JS + "</script>\n"
