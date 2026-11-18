"""Headless-Chromium checks of the map UI todo items (1, 7, 9, 11, 13, 17, 22/23, 25, 27).

Run against a served (byte-range server) rebuilt map page:

    python test_ui_todo.py http://localhost:8771/gipuzkoa/map.html city
    python test_ui_todo.py http://localhost:8772/map_uitest.html us
    python test_ui_todo.py http://localhost:8773/map_uitest.html spain

(`kind` only switches the checks that exist on one kind of page: the US/Spain scope picker, the Spain language switch.)
Under pytest the tests are skipped unless UI_TODO_URL (and optionally UI_TODO_KIND) is set.
"""
import tempfile
import os
import sys

from playwright.sync_api import sync_playwright

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("  PASS " if ok else "  FAIL ") + name + ("" if ok else "  -> " + str(detail)[:300]), flush=True)


# MapLibre expression evaluator for the subset the circle radius uses (+ * ^ max min / coalesce get literal numbers).
EVAL_JS = """
function ev(e, props) {
  if (typeof e === 'number') return e;
  var op = e[0];
  if (op === 'get') return props[e[1]];
  if (op === 'coalesce') { for (var i = 1; i < e.length; i++) { var v = ev(e[i], props); if (v != null) return v; } return null; }
  var a = e.slice(1).map(function(x) { return ev(x, props); });
  if (op === '+') return a.reduce(function(p, c) { return p + c; }, 0);
  if (op === '*') return a.reduce(function(p, c) { return p * c; }, 1);
  if (op === '/') return a[0] / a[1];
  if (op === '^') return Math.pow(a[0], a[1]);
  if (op === 'max') return Math.max.apply(null, a);
  if (op === 'min') return Math.min.apply(null, a);
  throw new Error('op ' + op);
}
"""


def open_page(b, url, w=int(os.environ.get("UI_W", 1400)), h=int(os.environ.get("UI_H", 900))):
    pg = b.new_page(viewport={"width": w, "height": h})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:150]))
    pg.goto(url, wait_until="domcontentloaded", timeout=120000)
    pg.wait_for_function("window.__mainMap && window.__mainMap.isStyleLoaded()", timeout=90000)
    pg.wait_for_timeout(5000)
    pg.errs = errs
    return pg


def fly(pg, lng, lat, z):
    pg.evaluate("([x,y,z]) => window.__mainMap.jumpTo({center:[x,y], zoom:z})", [lng, lat, z])
    pg.wait_for_timeout(1800)


def t01_controls(b, url, kind):
    pg = open_page(b, url)
    r = pg.evaluate("""() => {
      const rect = s => { const e = document.querySelector(s); if (!e) return null; const r = e.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height,r:r.right,b:r.bottom}; };
      const zi = rect('.maplibregl-ctrl-zoom-in'), zo = rect('.maplibregl-ctrl-zoom-out'), cp = rect('.maplibregl-ctrl-compass');
      return {zi, zo, cp, tl: rect('.maplibregl-ctrl-top-left'), trKids: document.querySelectorAll('.maplibregl-ctrl-top-right .maplibregl-ctrl').length,
              legend: rect('#layer-switcher'), lang: rect('#lang-switch'), W: innerWidth, addr: rect('#addressSearch'), edit: rect('#routeEditToggleBtn')};
    }""")
    check("1 zoom/north controls are in the top-left corner", r["zi"] and r["zi"]["x"] < 300 and r["zi"]["y"] < 60 and r["trKids"] == 0, r)
    check("1 controls form a horizontal row", abs(r["zi"]["y"] - r["zo"]["y"]) < 2 and abs(r["zi"]["y"] - r["cp"]["y"]) < 2 and r["zo"]["x"] > r["zi"]["x"], r)
    check("1 legend moved to the right edge", r["legend"]["r"] >= r["W"] - 14, r["legend"])
    if r["edit"] and r["edit"]["w"] > 0:
        check("1 controls do not overlap the 'Draw a new line' button", r["zi"]["x"] >= r["edit"]["r"] - 1, r)
    if r["lang"]:
        check("1 language button in the top-left row, right of the zoom controls", r["lang"]["x"] >= r["cp"]["r"] - 1 and r["lang"]["x"] < 450 and abs(r["lang"]["y"] - r["zi"]["y"]) < 20, r)
        if r["addr"]:
            check("1 address box does not overlap the language button", r["addr"]["x"] >= r["lang"]["r"] - 1 or r["addr"]["y"] > r["lang"]["b"], r)
    # stats panel (plots) must not overlap the row
    pg.click("#statsButton")
    pg.wait_for_timeout(800)
    sp = pg.evaluate("(() => { const r = document.getElementById('statsPanel').getBoundingClientRect(); return {y: r.y, x: r.x}; })()")
    check("1 plots panel does not overlap the control row", sp["y"] > r["zi"]["y"] + r["zi"]["h"] + 5, (sp, r["zi"]))
    pg.close()


def t07_popup_lowzoom(b, url, kind):
    pg = open_page(b, url)
    zoom0 = pg.evaluate("window.__mainMap.getZoom()")
    c = pg.evaluate("window.__mainMap.getCenter().toArray()")
    fly(pg, c[0], c[1], 3.2)
    pg.mouse.click(700, 450)
    pg.wait_for_timeout(2500)
    n = pg.evaluate("document.querySelectorAll('.maplibregl-popup').length")
    check("7 no hexagon popup at very low zoom (3.2)", n == 0, n)
    fly(pg, c[0], c[1], 5.0)
    pg.mouse.click(700, 450)
    pg.wait_for_timeout(2500)
    n = pg.evaluate("document.querySelectorAll('.maplibregl-popup').length")
    check("7 no hexagon popup at zoom 5.0", n == 0, n)
    pg.close()


def t07b_popup_highzoom(b, url, kind):
    pg = open_page(b, url)
    c = pg.evaluate("window.__mainMap.getCenter().toArray()")
    z = 11 if kind == "city" else 9
    fly(pg, c[0], c[1], z)
    pg.evaluate("document.querySelectorAll('.maplibregl-popup-close-button').forEach(b=>b.click())")
    pt = pg.evaluate("""() => { const m = window.__mainMap;
      const ls = m.getStyle().layers.filter(l => /circle|fill/.test(l.type) && /h3_/.test(l.id) && m.getLayer(l.id) && m.getLayoutProperty(l.id,'visibility') !== 'none').map(l => l.id);
      const f = m.queryRenderedFeatures({layers: ls}).filter(x => { const q = m.project(x.geometry.type === 'Point' ? x.geometry.coordinates : m.getCenter().toArray()); return q.x > 420 && q.x < 1000 && q.y > 100 && q.y < 800; });
      if (!f.length) return null; const q = m.project(f[0].geometry.coordinates); return [q.x, q.y]; }""")
    if pt:
        pg.mouse.click(*pt)
        pg.wait_for_timeout(3500)
        n = pg.evaluate("document.querySelectorAll('.maplibregl-popup').length")
        check("7 popup still opens at working zoom", n >= 1, n)
        pg.wait_for_timeout(1500)
        pg.evaluate("window.__colInfoScan && window.__colInfoScan()")
        pn = pg.evaluate("document.querySelectorAll('.maplibregl-popup .colInfoBtn').length")
        check("13 popup table rows carry 'i' buttons", pn >= 1, pn)
    else:
        print("  SKIP 7 (no clickable cell found at working zoom)")
    pg.close()


def t09_level_selector(b, url, kind):
    if kind not in ("us", "spain"):
        return
    pg = open_page(b, url)
    pg.wait_for_timeout(4000)
    txt = pg.evaluate("[...document.querySelectorAll('.scopeMainBtn')].map(b => b.innerText).join(' | ')")
    check("9 level selector button shows the level without pop / veh-km", txt and "veh-km" not in txt and "Pop." not in txt and "Level" in txt, txt)
    pg.click("#statsButton")
    pg.wait_for_timeout(800)
    pg.locator(".scopeMainBtn").first.click()
    pg.wait_for_timeout(500)
    pg.locator(".scopePop button[data-act=state]").first.click()
    pg.wait_for_timeout(1200)
    items = pg.evaluate("[...document.querySelectorAll('.scopeItem')].slice(0,6).map(e => e.innerText)")
    check("9 picker options carry no population figure", items and all(len(i.split("\n")) == 1 and not any(ch.isdigit() for ch in i) for i in items), items)
    pg.close()


def t11_attribution(b, url, kind):
    pg = open_page(b, url)
    pg.wait_for_timeout(3000)
    r = pg.evaluate("""() => { const a = document.querySelector('.maplibregl-ctrl-attrib'); if (!a) return null;
      const inner = a.querySelector('.maplibregl-ctrl-attrib-inner'); const btn = a.querySelector('.maplibregl-ctrl-attrib-button');
      return {open: a.hasAttribute('open'), cls: a.className, innerVisible: inner ? inner.getBoundingClientRect().width > 0 : false, hasBtn: !!btn}; }""")
    check("11 attribution collapsed by default (compact)", r and "maplibregl-compact" in r["cls"] and not r["open"] and not r["innerVisible"] and r["hasBtn"], r)
    pg.locator(".maplibregl-ctrl-attrib-button").click()
    pg.wait_for_timeout(500)
    o = pg.evaluate("(() => { const a = document.querySelector('.maplibregl-ctrl-attrib'); const i = a.querySelector('.maplibregl-ctrl-attrib-inner'); return a.hasAttribute('open') && i.getBoundingClientRect().width > 0; })()")
    check("11 attribution expands on click", o)
    pg.close()


def t13_col_info(b, url, kind):
    pg = open_page(b, url)
    has_meta = pg.evaluate("Object.keys(window.__columnMeta || {}).length")
    check("13 column metadata embedded", has_meta > 5, has_meta)
    # layer-switcher selectors: open the circle size selector
    pg.wait_for_timeout(1500)
    pg.evaluate("window.__colInfoScan && window.__colInfoScan()")
    n_leg = pg.evaluate("document.querySelectorAll('#layer-switcher .colInfoBtn').length")
    check("13 info buttons next to column names in the legend / selectors", n_leg >= 1, n_leg)
    pg.click("#statsButton")
    pg.wait_for_timeout(1000)
    for tab, minimum in (("Distribution", 1), ("Regression", 1), ("Census", 3)):
        pg.locator("#statsPanel button", has_text=tab).first.click()
        pg.wait_for_timeout(1800)
        pg.evaluate("window.__colInfoScan && window.__colInfoScan()")
        n = pg.evaluate("document.querySelectorAll('#statsPanel .colInfoBtn').length")
        check(f"13 info buttons in the {tab} tab", n >= minimum, n)
        if tab == "Census" and n:
            vis = pg.evaluate("""() => { const bs = [...document.querySelectorAll('#statsTab_anova .colInfoBtn')].filter(x => x.offsetParent);
              return bs.slice(0, 40).map(x => x.parentElement.textContent.trim()); }""")
            check("13 census panel rows carry an 'i' (visible rows: %s)" % vis[:3], len(vis) >= 1, vis)
            btn = pg.locator("#statsTab_anova .colInfoBtn:visible").first
            btn.click()
            pg.wait_for_timeout(400)
            panel = pg.evaluate("(() => { const p = document.getElementById('colInfoPanel'); return p && p.style.display === 'block' ? p.innerText : null; })()")
            check("13 clicking 'i' opens a panel with source / native level / description", panel and all(k in panel.upper() for k in ("SOURCE", "NATIVE LEVEL", "DESCRIPTION")), panel)
            pg.keyboard.press("Escape")
            pg.wait_for_timeout(200)
            check("13 panel closes with Escape", pg.evaluate("document.getElementById('colInfoPanel').style.display === 'none'"))
    # popup table (cell popup)
    c = pg.evaluate("window.__mainMap.getCenter().toArray()")
    pg.click("#statsButton")
    pg.wait_for_timeout(300)
    pg.close()


def t17_badges(b, url, kind):
    pg = open_page(b, url)
    r = pg.evaluate("""() => { const m = window.__mainMap; const l = m.getLayer('__routes_label'); return {labelMin: l ? l.minzoom : null}; }""")
    check("17 route name badges on lines start at zoom >= 16", r["labelMin"] is None or r["labelMin"] >= 16, r)
    src = pg.content()
    check("17 per-stop route badges gated at zoom 17.5", "ROUTE_BADGE_MIN_ZOOM = 17.5" in src)
    if pg.query_selector("#ovStopRouteBadgesCheckbox"):
        c = pg.evaluate("window.__mainMap.getCenter().toArray()")
        pg.check("#ovStopRouteBadgesCheckbox")
        # zoom onto a rendered stop
        stop = pg.evaluate("""() => { const m = window.__mainMap; const f = m.querySourceFeatures('__stops_overlay').filter(x => !x.properties.point_count); return f.length ? f[0].geometry.coordinates : null; }""")
        if stop is None:
            fly(pg, c[0], c[1], 14)
            stop = pg.evaluate("""() => { const m = window.__mainMap; const f = m.querySourceFeatures('__stops_overlay').filter(x => !x.properties.point_count); return f.length ? f[0].geometry.coordinates : null; }""")
        if stop:
            fly(pg, stop[0], stop[1], 16.6)
            lo = pg.evaluate("document.querySelectorAll('.maplibregl-marker').length")
            fly(pg, stop[0], stop[1], 18.2)
            pg.wait_for_timeout(1500)
            hi = pg.evaluate("document.querySelectorAll('.maplibregl-marker').length")
            check("17 stop route badges hidden at 16.6, shown at 18.2", lo == 0 and hi > 0, (lo, hi))
    pg.close()


def t22_strict_bands(b, url, kind):
    pg = open_page(b, url)
    bad = []
    for z in [x / 2 for x in range(0, 41)]:
        res = pg.evaluate(
            """(z) => { const m = window.__mainMap; const out = {};
              m.getStyle().layers.forEach(l => { const mt = /^(hexagons|circles|census):(h3_\\d+|census_\\w+?)_(fill|circle)/.exec(l.id); if (!mt) return;
                if ((l.minzoom || 0) <= z && z < (l.maxzoom == null ? 24 : l.maxzoom)) (out[mt[1]] = out[mt[1]] || new Set()).add(mt[2]); });
              const o = {}; Object.keys(out).forEach(k => o[k] = [...out[k]]); return o; }""",
            z,
        )
        for g, lv in res.items():
            if len(lv) > 1:
                bad.append((z, g, lv))
    check("22/23 at every zoom (0..20, step .5) each hexagon / circle / census group has at most one level in range", not bad, bad[:4])
    # also at a high zoom, rendered features exist for only one resolution of the active group
    c = pg.evaluate("window.__mainMap.getCenter().toArray()")
    fly(pg, c[0], c[1], 15)
    lv = pg.evaluate("""() => { const m = window.__mainMap; const s = new Set(); m.queryRenderedFeatures().forEach(f => { if (/^(hexagons|circles):h3_/.test(f.layer.id)) s.add(f.layer.id.split(':')[1].split('_').slice(0,2).join('_')); }); return [...s]; }""")
    check("22/23 only one H3 resolution is rendered at zoom 15", len(lv) <= 1, lv)
    fly(pg, c[0], c[1], 17.5)
    lv = pg.evaluate("""() => { const m = window.__mainMap; const s = new Set(); m.queryRenderedFeatures().forEach(f => { if (/^(hexagons|circles):h3_/.test(f.layer.id)) s.add(f.layer.id.split(':')[1].split('_').slice(0,2).join('_')); }); return [...s]; }""")
    check("22/23 only one H3 resolution is rendered at zoom 17.5", len(lv) <= 1, lv)
    pg.close()


def t27_circle_legend(b, url, kind):
    pg = open_page(b, url)
    pg.evaluate("document.querySelector('input[name=__base_group][value=circles]').click()")
    pg.wait_for_timeout(1500)
    c = pg.evaluate("window.__mainMap.getCenter().toArray()")
    zooms = [float(z) for z in os.environ["UI_ZOOMS"].split(",")] if os.environ.get("UI_ZOOMS") else [4, 5.5, 7, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 15, 17]
    for z in zooms:
        fly(pg, c[0], c[1], z)
        pg.wait_for_timeout(1200)
        r = pg.evaluate(
            "(z) => {" + EVAL_JS + """
            const m = window.__mainMap, field = window.__circleField;
            const cl = [1,2,3].map(i => document.getElementById('circleLegendC' + i));
            const lab = ['Min','Mid','Max'].map(k => document.getElementById('circleLegend' + k).textContent);
            const res = cl[0].dataset.res;
            const layerId = m.getStyle().layers.map(l => l.id).find(id => new RegExp('^circles:h3_' + res + '_circle').test(id));
            const expr = m.getPaintProperty(layerId, 'circle-radius');
            const vals = cl.map(c => Number(c.dataset.value)), diam = cl.map(c => c.getBoundingClientRect().width);
            const props = v => { const o = {}; o[field] = v; return o; };
            const drawn = vals.map(v => 2 * ev(expr, props(v)));
            // layer on screen at this zoom
            const vis = m.getStyle().layers.filter(l => /^circles:h3_/.test(l.id) && (l.minzoom || 0) <= z && z < (l.maxzoom == null ? 24 : l.maxzoom)).map(l => l.id.split(':')[1].split('_')[1]).filter((x, i, a) => a.indexOf(x) === i);
            // real rendered features: radius of each from its own value, must lie within [3, max]
            const feats = m.queryRenderedFeatures({layers: m.getStyle().layers.filter(l => /^circles:h3_/.test(l.id) && m.getLayer(l.id)).map(l => l.id)});
            const rad = feats.map(f => ev(m.getPaintProperty(f.layer.id, 'circle-radius'), f.properties)).filter(x => x != null);
            return {z, res, vis, lab, vals, diam, drawn, nFeat: feats.length, radMin: rad.length ? Math.min.apply(null, rad) : null, radMax: rad.length ? Math.max.apply(null, rad) : null, shown: getComputedStyle(document.getElementById('circleLegend')).display};
            }""",
            z,
        )
        ok_vis = r["vis"] == [r["res"]]
        ok_nonzero = all(l not in ("0", "", "0%") for l in r["lab"]) and all(v > 0 for v in r["vals"])
        ok_match = all(abs(d - w) < 0.6 for d, w in zip(r["diam"], r["drawn"]))
        ok_incr = r["vals"][0] < r["vals"][1] < r["vals"][2]
        check(f"27 zoom {z}: legend res {r['res']} == layer res on screen {r['vis']}", ok_vis, r)
        check(f"27 zoom {z}: legend labels non-zero {r['lab']}", ok_nonzero and ok_incr, r)
        check(f"27 zoom {z}: legend diameters {[round(x,1) for x in r['diam']]} == drawn diameters {[round(x,1) for x in r['drawn']]}", ok_match, r)
        if r["nFeat"]:
            check(f"27 zoom {z}: drawn feature radii ({r['radMin']:.1f}..{r['radMax']:.1f}px) inside the legend's range", r["radMin"] >= 3 - 1e-6 and r["radMax"] <= max(r["drawn"]) / 2 + 3 + 1e-6, r)
    pg.close()


def t25_development(b, url, kind):
    pg = open_page(b, url)
    info = pg.evaluate("""() => { const m = window.__mainMap; return m.getStyle().layers.filter(l => /development/.test(l.id)).map(l => [l.id, l.minzoom]); }""")
    check("25 development layer has a minzoom >= 8", info and all(z >= 8 for _, z in info), info)
    reqs = []
    pg.on("request", lambda r: reqs.append(r.url) if "development" in r.url else None)
    c = pg.evaluate("window.__mainMap.getCenter().toArray()")
    pg.evaluate("document.querySelector('input[name=__overlay_group][value=development]').click()")
    fly(pg, c[0], c[1], 5.0 if kind != "city" else 7.0)
    pg.wait_for_timeout(2500)
    TILES = """() => { const m = window.__mainMap; let n = 0;
      m.getStyle().layers.filter(l => /development/.test(l.id)).forEach(l => { n += m.querySourceFeatures(l.source, {sourceLayer: l['source-layer']}).length; }); return n; }"""
    low_tiles = pg.evaluate(TILES)
    low_vis = pg.evaluate("(() => { const m = window.__mainMap, z = m.getZoom(); return m.getStyle().layers.some(l => /development/.test(l.id) && l.layout.visibility !== 'none' && (l.minzoom||0) <= z); })()")
    check("25 checkbox on but low zoom: development layer is out of its zoom range and holds no tiles", not low_vis and low_tiles == 0, (low_vis, low_tiles, len(reqs)))
    fly(pg, c[0], c[1], 11)
    pg.wait_for_timeout(6000)
    pg.evaluate('window.__mainMap.panBy([30, 20])')
    pg.wait_for_timeout(3000)
    high_vis = pg.evaluate("(() => { const m = window.__mainMap, z = m.getZoom(); return m.getStyle().layers.some(l => /development/.test(l.id) && l.layout.visibility !== 'none' && (l.minzoom||0) <= z); })()")
    high_tiles = pg.evaluate(TILES)
    check("25 checkbox on and high zoom: development layer is active (tiles loaded: %d; 0 only if the dev tiles are missing on disk)" % high_tiles, high_vis, (high_vis, high_tiles))
    pg.evaluate("document.querySelector('input[name=__overlay_group][value=development]').click()")
    pg.close()


def t30_simplified_columns(b, url, kind):
    pg = open_page(b, url)
    S = pg.evaluate("window.__colSelect")
    check("30 simplified column set + relevance ranks embedded", S and len(S["simple"]) > 3 and len(S["r2"]) > 3, S and {k: len(v) for k, v in S.items() if isinstance(v, list)})
    total = len(S["r2"])
    check("30 simplified set is smaller than the full list (%d of %d)" % (len(S["simple"]), total), len(S["simple"]) < total or total < 15, (len(S["simple"]), total))
    # parties: at most 4 per election in the simplified set
    import re
    groups = {}
    for c in S["simple"]:
        m = re.match(r"^(.*elec_(?:national|state|municipal)_\d{4})_party_(.+)_(?:votes|share)$", c)
        if m:
            groups.setdefault(m.group(1), set()).add(m.group(2))
    check("3 election parties: at most 4 per election in the simplified list %s" % {k: len(v) for k, v in groups.items()}, all(len(v) <= 4 for v in groups.values()))
    pg.click("#statsButton"); pg.wait_for_timeout(800)
    pg.locator("#statsPanel button", has_text="Census").first.click(); pg.wait_for_timeout(2500)
    n0 = pg.evaluate("document.querySelectorAll('#statsTab_anova > div > div').length")
    pg.locator("#extCensusCheck").check(); pg.wait_for_timeout(2500)
    n1 = pg.evaluate("document.querySelectorAll('#statsTab_anova > div > div').length")
    check("30 'Extended census' reveals more census rows (%d -> %d)" % (n0, n1), n1 >= n0 and (n1 > n0 or len(S["simple"]) >= total), (n0, n1))
    pg.locator("#extCensusCheck").uncheck(); pg.wait_for_timeout(500)
    # dropdown order + filtering: open the regression field dropdown
    pg.locator("#statsPanel button", has_text="Regression").first.click(); pg.wait_for_timeout(1500)
    pg.locator("#regFieldSelect + .__csel-btn, #regFieldSelect ~ .__csel-btn").first.click(); pg.wait_for_timeout(400)
    items = pg.evaluate("[...document.querySelectorAll('.__csel-list.__csel-open .__csel-item')].map(e => e.textContent)")
    check("30 regression dropdown shows only the simplified columns (%d items)" % len(items), 0 < len(items) <= max(len(S["simple"]) + 1, 1), len(items))
    pg.close()


def t31_scope_rules(b, url, kind):
    if kind not in ("us", "spain"):
        return
    pg = open_page(b, url)
    pg.wait_for_timeout(3000)
    r = pg.evaluate("""() => ({lv: window.__scopeLevel && window.__scopeLevel(),
       state: window.__colVisible('ine_elec_state_2023_party_pp_share'), muni: window.__colVisible('ine_elec_municipal_2023_party_pp_share'),
       natparty: window.__colVisible('ine_elec_national_2023_party_pp_share'), left: window.__colVisible('ine_elec_national_2023_left_votes') })""")
    if kind == "spain":
        check("5 national scope: no state / municipal / national party columns, left-right proxy allowed", r["lv"] == "nation" and not r["state"] and not r["muni"] and not r["natparty"], r)
    pg.close()


def t33_ranking(b, url, kind):
    if kind not in ("us", "spain"):
        return
    pg = open_page(b, url)
    pg.click("#statsButton"); pg.wait_for_timeout(800)
    btn = pg.locator("#statsPanel .statsTabBtn[data-tab=ranking]")
    check("15 Ranking tab exists", btn.count() == 1)
    btn.click(); pg.wait_for_timeout(2500)
    txt = pg.evaluate("document.getElementById('statsTab_ranking').innerText")
    check("15 ranking lists the units (states) with values", "1." in txt and "weighted by" in txt, txt[:200])
    pg.select_option("#rkLevel", "county"); pg.wait_for_timeout(1500)
    txt = pg.evaluate("document.getElementById('statsTab_ranking').innerText")
    check("15 ranking by county / province works", "1." in txt, txt[:200])
    check("15 unavailable levels are stated, not faked", "not available" in txt.lower(), txt[-200:])
    pg.select_option("#rkMethod", "median"); pg.wait_for_timeout(800)
    check("15 median method selectable", pg.evaluate("document.getElementById('rkMethod').value") == "median")
    pg.close()


def t34_presentation(b, url, kind):
    pg = open_page(b, url)
    link = pg.evaluate("window.__presentationLink()")
    check("36 'Copy presentation link' button in the normal view", pg.evaluate("!!document.getElementById('copyPresentationLink')") and "present=1" in link, link)
    pg.close()
    c = open_page(b, url).evaluate("window.__mainMap.getCenter().toArray()")
    q = "?present=1&shape=circles&badges=1&z=11&lat=%f&lng=%f&tab=regression&alpha=0.6" % (c[1], c[0])
    pg = open_page(b, url + q)
    pg.wait_for_timeout(7000)
    r = pg.evaluate("""() => ({pres: document.body.classList.contains('presentation'), body: getComputedStyle(document.getElementById('layer-switcher-body')).display,
       shape: document.querySelector('input[name=__base_group]:checked').value, z: Math.round(window.__mainMap.getZoom()),
       panel: getComputedStyle(document.getElementById('statsPanel')).display, tab: window.__statsTab,
       alpha: document.getElementById('shapeOpacitySlider').value, noscroll: (() => { const p = document.getElementById('statsPanel'); return getComputedStyle(p).overflow; })(),
       ctrl: getComputedStyle(document.querySelector('.maplibregl-ctrl-top-left')).display, cmap: !!document.getElementById('mapLegend') && getComputedStyle(document.getElementById('mapLegend')).display !== 'none'})""")
    check("36 URL params: shape, zoom, tab, alpha applied", r["shape"] == "circles" and r["z"] == 11 and r["panel"] != "none" and r["tab"] == "regression" and r["alpha"] == "60", r)
    check("36 presentation chrome: layer controls hidden, colormap legend kept, no scrolling", r["pres"] and r["body"] == "none" and r["cmap"] and r["noscroll"] == "hidden" and r["ctrl"] == "none", r)
    pg.screenshot(path=os.path.join(tempfile.gettempdir(), "present_%s.png" % kind))
    pg.close()


def t35_translated_i(b, url, kind):
    if kind != "spain":
        return
    pg = open_page(b, url)
    pg.evaluate("window.__setMapLanguage('es')"); pg.wait_for_timeout(1500)
    pg.click("#statsButton"); pg.wait_for_timeout(800)
    pg.locator("#statsPanel .statsTabBtn[data-tab=anova]").click(); pg.wait_for_timeout(2500)
    pg.evaluate("window.__colInfoScan()")
    n = pg.evaluate("document.querySelectorAll('#statsTab_anova .colInfoBtn').length")
    check("13 'i' buttons also next to translated (Spanish) column labels", n >= 3, n)
    pg.close()


TESTS = [t01_controls, t07_popup_lowzoom, t07b_popup_highzoom, t09_level_selector, t11_attribution, t13_col_info, t17_badges, t22_strict_bands, t27_circle_legend, t25_development, t30_simplified_columns, t31_scope_rules, t33_ranking, t34_presentation, t35_translated_i]


def run(url, kind, only=None):
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--disable-gpu", "--use-gl=swiftshader", "--disable-dev-shm-usage"])
        for t in TESTS:
            if only and not any(t.__name__.startswith("t" + o.zfill(2)) for o in only):
                continue
            print("==", t.__name__, flush=True)
            try:
                t(b, url, kind)
            except Exception as e:  # noqa: BLE001
                check(t.__name__ + " ran without exception", False, repr(e))
        b.close()
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed" + (": FAILED " + "; ".join(bad[:12]) if bad else ""))
    return not bad


def test_ui_todo():  # pytest entry
    import pytest

    url = os.environ.get("UI_TODO_URL")
    if not url:
        pytest.skip("set UI_TODO_URL (and UI_TODO_KIND=city|us|spain) to run the browser checks")
    assert run(url, os.environ.get("UI_TODO_KIND", "city"))


if __name__ == "__main__":
    ok = run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "city", sys.argv[3].split(",") if len(sys.argv) > 3 else None)
    sys.exit(0 if ok else 1)
