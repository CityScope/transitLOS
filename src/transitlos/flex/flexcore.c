/* flexcore.c -- the heavy loops of the flexible scoring prototype (compiled and loaded by flex/core.py with ctypes).
 *
 *  1. scoring       stop_score / walk decay / PTLOS, identical to flex/scoring.py (parameter vector `sp`)
 *  2. select_core   the CAP: equilibrium, tolerant extremes of 1, 2, 3.. axes, every mode, then diversity fill that prefers good values
 *  3. select_profiles   floors + dominance + cap for the (mode, headway, speed) profiles of one station
 *  4. multilabel_search multi-source Dijkstra where every node keeps up to Kc labels (one per station), pruned by dominance
 *  5. process_cells     edge-touch distances + R, floors, dominance and cap -> the K stations of every cell
 *
 * Parameter vectors: sp = scoring (layout in the scoring section below), cfg = caps/floors:
 *   cfg[0] tau_band  cfg[1] gamma  cfg[2] delta  cfg[3] dom_tau_h  cfg[4] dom_tau_v  cfg[5] h_max  cfg[6] v_min  cfg[7] s_min  cfg[8] p_min
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define AX_DIST 0
#define AX_HEAD 1
#define AX_MODE 2
#define AX_SPEED 3

/* ------------------------------------------------------------------ scoring */
/* Parameter vector sp (doubles). Header:
 *   0..2 mrc bus,tram,rail  3 v_anchor 4 v_bonus 5 v_n  6 h_sat 7 h_p 8 h_q  9 walk_speed 10 plateau_m  11..13 t0_base bus,tram,rail  14 k  15 p_shape
 *   16 w_mode 17 w_speed 18 w_headway  19 agg (0 nested, 1 power mean, 2 product)  20 agg_p
 *   curves (6 doubles each: kind 0 default / 1 table, lo, hi, log scale 0/1, n, offset of the table in sp): 21 headway, 27 speed, 33 walk
 *   tables follow from index 39.  Must match flex/core.py::sp_array and flex/scoring.py. */
#define CV(c) (21 + 6 * (c))
static double table_eval(const double *sp, int c, double x) {
    const double *t = sp + CV(c);
    int n = (int)t[4];
    const double *y = sp + (long)t[5];
    double pos = t[3] != 0.0 ? (log(x < 1e-300 ? 1e-300 : x) - log(t[1])) / (log(t[2]) - log(t[1])) : (x - t[1]) / (t[2] - t[1]);
    pos *= (n - 1);
    if (pos < 0) pos = 0; if (pos > n - 1) pos = n - 1;
    int i = (int)floor(pos); if (i > n - 2) i = n - 2;
    double fr = pos - i;
    return y[i] * (1.0 - fr) + y[i + 1] * fr;
}
static double headway_score(double h, const double *sp) {
    if (sp[CV(0)] == 1.0) return table_eval(sp, 0, h);
    double sat = sp[6];
    if (h <= sat) return 1.0;
    double eps = sp[7] + sp[8] * log(sat), lr = log(h / sat);
    return exp(-eps * lr - (sp[8] / 2.0) * lr * lr);
}
static double speed_score(double v, const double *sp) {
    if (v <= 0) return 0.0;
    if (sp[CV(1)] == 1.0) return table_eval(sp, 1, v);
    double m = 1.0 + sp[4], n = sp[5], kk = sp[3] * pow(m - 1.0, 1.0 / n);
    double vn = pow(v, n);
    return m * vn / (vn + pow(kk, n));
}
static double stop_score_c(int mode, double h, double v, const double *sp) {
    double m = sp[mode], s = speed_score(v, sp), f = headway_score(h, sp);
    if (m <= 0 || s <= 0 || f <= 0) return 0.0;
    double r;
    if (sp[19] == 2.0) {   /* product: MRC^wm * V^ws * H^wh */
        r = pow(m, sp[16]) * pow(s, sp[17]) * pow(f, sp[18]);
    } else if (sp[19] == 1.0) {   /* weighted generalised mean of (MRC, V, H) */
        double w0 = sp[16], w1 = sp[17], w2 = sp[18], ws = w0 + w1 + w2, p = sp[20];
        if (ws <= 0) return 0.0;
        w0 /= ws; w1 /= ws; w2 /= ws;
        if (p == 0.0) r = exp(w0 * log(m) + w1 * log(s) + w2 * log(f));
        else r = pow(w0 * pow(m, p) + w1 * pow(s, p) + w2 * pow(f, p), 1.0 / p);
    } else {               /* nested: ( MRC^wm * V^ws )^(1/(wm+ws)) * H^wh */
        double wms = sp[16] + sp[17];
        double core = (wms == 0.0) ? 1.0 : pow(pow(m, sp[16]) * pow(s, sp[17]), 1.0 / wms);
        r = core * pow(f, sp[18]);
    }
    return r > 1.0 ? 1.0 : r;
}
/* walk decay at walking time t (min) for a stop of score ss and mode */
static double walk_decay_t(double t, double ss, int mode, const double *sp) {
    double tp = sp[10] / sp[9] / 60.0;
    if (t <= tp) return 1.0;
    double u = (t - tp) / (sp[11 + mode] * (1.0 + sp[14] * ss));
    if (sp[CV(2)] == 1.0) return table_eval(sp, 2, u);
    return pow(1.0 + u, -sp[15]);
}
static double walk_decay_c(double d, double ss, int mode, const double *sp) { return walk_decay_t(d / sp[9] / 60.0, ss, mode, sp); }
static double ptlos_c(double d, int mode, double h, double v, const double *sp) {
    double ss = stop_score_c(mode, h, v, sp);
    return ss * walk_decay_c(d, ss, mode, sp);
}

void ptlos_array(int n, const double *d, const int *mode, const double *h, const double *v, const double *sp, double *out) {
    for (int i = 0; i < n; i++) out[i] = ptlos_c(d[i], mode[i], h[i], v[i], sp);
}
void stop_score_array(int n, const int *mode, const double *h, const double *v, const double *sp, double *out) {
    for (int i = 0; i < n; i++) out[i] = stop_score_c(mode[i], h[i], v[i], sp);
}

/* --------------------------------------------------------------- selection */
typedef struct { int mode; double h, v, d, q; } Item;

static double clampg(double x) { return x < 1e-3 ? 1e-3 : (x > 1.0 ? 1.0 : x); }

/* Cap `cap` items out of n. axes_mask bit AX_* says which axes exist. g (goodness 0..1, 1 = best) is computed per axis over the n candidates.
 * Order: equilibrium -> subsets of 1, 2 .. (na-1) axes (lexicographic in axis order dist, headway, mode, speed; subsets that contain mode run once
 * per mode level rail, tram, bus) -> diversity fill  gain = (q/qmax)^gamma * d_min, stopping when d_min < delta.
 * Returns the number chosen (indexes into the item array in `out`, in selection order). */
static int select_core(int n, const Item *it, int axes_mask, int cap, const double *cfg, int *out) {
    if (n <= 0) return 0;
    double tau = cfg[0], gamma = cfg[1], delta = cfg[2];
    int ax[4], na = 0;
    for (int a = 0; a < 4; a++) if (axes_mask & (1 << a)) ax[na++] = a;
    double *g = (double *)malloc(sizeof(double) * 4 * n);
    double dmin = 1e18, dmax = -1e18, hmin = 1e18, hmax = -1e18, vmin = 1e18, vmax = -1e18, qmax = -1e18;
    for (int i = 0; i < n; i++) {
        if (it[i].d < dmin) dmin = it[i].d; if (it[i].d > dmax) dmax = it[i].d;
        if (it[i].h < hmin) hmin = it[i].h; if (it[i].h > hmax) hmax = it[i].h;
        if (it[i].v < vmin) vmin = it[i].v; if (it[i].v > vmax) vmax = it[i].v;
        if (it[i].q > qmax) qmax = it[i].q;
    }
    double ld = log(dmax + 50.0) - log(dmin + 50.0), lh = log(hmax) - log(hmin), lv = vmax - vmin;
    for (int i = 0; i < n; i++) {
        g[AX_DIST * n + i] = ld > 1e-9 ? (log(dmax + 50.0) - log(it[i].d + 50.0)) / ld : 1.0;
        g[AX_HEAD * n + i] = lh > 1e-9 ? (log(hmax) - log(it[i].h)) / lh : 1.0;
        g[AX_MODE * n + i] = it[i].mode / 2.0;
        g[AX_SPEED * n + i] = lv > 1e-9 ? (it[i].v - vmin) / lv : 1.0;
    }
    char *chosen = (char *)calloc(n, 1);
    int cnt = 0;
#define ADD(i) do { if (!chosen[i] && cnt < cap) { chosen[i] = 1; out[cnt++] = (i); } } while (0)
    /* 1. equilibrium: best default score (ties -> lowest index, i.e. the nearest for cells) */
    int bi = 0;
    for (int i = 1; i < n; i++) if (it[i].q > it[bi].q) bi = i;
    ADD(bi);
    /* the nearest item is always kept EXACTLY (not within a tolerance): 'is there any transit within X m' must stay answerable whatever the scoring */
    if (axes_mask & (1 << AX_DIST)) {
        int nb = 0;
        for (int i = 1; i < n; i++) if (it[i].d < it[nb].d) nb = i;
        ADD(nb);
    }
    /* 2. tolerant extremes of subsets of 1..na-1 axes */
    for (int size = 1; size < na && cnt < cap; size++) {
        int idx[4];
        for (int j = 0; j < size; j++) idx[j] = j;
        for (;;) {
            int has_mode = 0;
            for (int j = 0; j < size; j++) if (ax[idx[j]] == AX_MODE) has_mode = 1;
            for (int lv_i = 0; lv_i < (has_mode ? 3 : 1) && cnt < cap; lv_i++) {
                int level = has_mode ? 2 - lv_i : -1;
                double best = -1.0;
                for (int i = 0; i < n; i++) {
                    if (level >= 0 && it[i].mode != level) continue;
                    double s = 0;
                    for (int j = 0; j < size; j++) s += log(clampg(g[ax[idx[j]] * n + i]));
                    s = exp(s / size);
                    if (s > best) best = s;
                }
                if (best < 0) continue;
                int pick = -1;
                for (int i = 0; i < n; i++) {
                    if (level >= 0 && it[i].mode != level) continue;
                    double s = 0;
                    for (int j = 0; j < size; j++) s += log(clampg(g[ax[idx[j]] * n + i]));
                    s = exp(s / size);
                    if (s >= best * (1.0 - tau) && (pick < 0 || it[i].q > it[pick].q)) pick = i;
                }
                if (pick >= 0) ADD(pick);
            }
            int j = size - 1;   /* next combination */
            while (j >= 0 && idx[j] == na - size + j) j--;
            if (j < 0) break;
            idx[j]++;
            for (int k = j + 1; k < size; k++) idx[k] = idx[k - 1] + 1;
        }
    }
    /* 3. diversity fill, always preferring good values */
    while (cnt < cap) {
        int pick = -1; double bgain = -1.0, bdmin = 0;
        for (int i = 0; i < n; i++) {
            if (chosen[i]) continue;
            double dm = 1e18;
            for (int c = 0; c < cnt; c++) {
                double s = 0;
                for (int j = 0; j < na; j++) { double dd = g[ax[j] * n + i] - g[ax[j] * n + out[c]]; s += dd * dd; }
                s = sqrt(s / na);
                if (s < dm) dm = s;
            }
            double qq = qmax > 0 ? it[i].q / qmax : 1.0;
            double gain = pow(qq < 0 ? 0 : qq, gamma) * dm;
            if (gain > bgain) { bgain = gain; pick = i; bdmin = dm; }
        }
        if (pick < 0 || bdmin < delta) break;
        ADD(pick);
    }
    free(g); free(chosen);
    return cnt;
}

/* dominance of item b by item a (a at least as good on every axis, within tolerance), used for profiles */
static int prof_dominates(const Item *a, const Item *b, const double *cfg) {
    return a->mode >= b->mode && a->h <= b->h * (1.0 + cfg[3]) && a->v >= b->v * (1.0 - cfg[4]);
}

/* Profiles of ONE station: floors -> exact dominance -> tolerance merge (best first) -> cap. in: n candidates. out: indexes (<= P). */
static int select_profiles_one(int n, const int *mode, const float *h, const float *v, int P, const double *sp, const double *cfg, int *out, int *dropped) {
    Item *it = (Item *)malloc(sizeof(Item) * (n > 0 ? n : 1));
    int *map = (int *)malloc(sizeof(int) * (n > 0 ? n : 1));
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (!(h[i] <= cfg[5]) || !(v[i] >= cfg[6])) continue;
        double q = stop_score_c(mode[i], h[i], v[i], sp);
        if (q < cfg[7]) continue;   /* floors are optional (default 0): a zero-quality pattern is still a pattern */
        it[m].mode = mode[i]; it[m].h = h[i]; it[m].v = v[i]; it[m].d = 0; it[m].q = q; map[m++] = i;
    }
    /* exact dominance (no tolerance), then tolerance merge best-first */
    char *keep = (char *)calloc(m > 0 ? m : 1, 1);
    for (int i = 0; i < m; i++) {
        keep[i] = 1;
        for (int j = 0; j < m && keep[i]; j++) {
            if (i == j) continue;
            int ge = it[j].mode >= it[i].mode && it[j].h <= it[i].h && it[j].v >= it[i].v;
            int eq = it[j].mode == it[i].mode && it[j].h == it[i].h && it[j].v == it[i].v;
            if (ge && (!eq || j < i)) keep[i] = 0;
        }
    }
    int *ord = (int *)malloc(sizeof(int) * (m > 0 ? m : 1)); int no = 0;
    for (int i = 0; i < m; i++) if (keep[i]) ord[no++] = i;
    for (int a = 1; a < no; a++) { int x = ord[a], b = a - 1; while (b >= 0 && it[ord[b]].q < it[x].q) { ord[b + 1] = ord[b]; b--; } ord[b + 1] = x; }
    int nk = 0; int *kept = (int *)malloc(sizeof(int) * (no > 0 ? no : 1));
    for (int a = 0; a < no; a++) {
        int dom = 0;
        for (int b = 0; b < nk && !dom; b++) if (prof_dominates(&it[kept[b]], &it[ord[a]], cfg)) dom = 1;
        if (!dom) kept[nk++] = ord[a];
    }
    Item *it2 = (Item *)malloc(sizeof(Item) * (nk > 0 ? nk : 1));
    for (int a = 0; a < nk; a++) it2[a] = it[kept[a]];
    int *sel = (int *)malloc(sizeof(int) * (P > 0 ? P : 1));
    int c = nk <= P ? nk : select_core(nk, it2, (1 << AX_HEAD) | (1 << AX_MODE) | (1 << AX_SPEED), P, cfg, sel);
    if (nk <= P) for (int a = 0; a < nk; a++) sel[a] = a;
    for (int a = 0; a < c; a++) out[a] = map[kept[sel[a]]];
    *dropped = nk - c;
    free(it); free(map); free(keep); free(ord); free(kept); free(it2); free(sel);
    return c;
}

/* Many stations. off[ns+1] delimits each station's candidate profiles. out_idx[ns*P] = chosen candidate index or -1; out_n[ns]; out_drop[ns]. */
void select_profiles(int ns, const int *off, const int *mode, const float *h, const float *v, int P, const double *sp, const double *cfg,
                     int *out_idx, int *out_n, int *out_drop) {
    int *tmp = (int *)malloc(sizeof(int) * (P + 1));
    for (int s = 0; s < ns; s++) {
        int dropped = 0;
        int c = select_profiles_one(off[s + 1] - off[s], mode + off[s], h + off[s], v + off[s], P, sp, cfg, tmp, &dropped);
        for (int a = 0; a < P; a++) out_idx[s * P + a] = a < c ? off[s] + tmp[a] : -1;
        out_n[s] = c; out_drop[s] = dropped;
    }
    free(tmp);
}

/* generic cap for the tests: items given as arrays; returns count, indexes in `out` */
int select_items(int n, const int *mode, const double *h, const double *v, const double *d, const double *q, int axes_mask, int cap, const double *cfg, int *out) {
    Item *it = (Item *)malloc(sizeof(Item) * (n > 0 ? n : 1));
    for (int i = 0; i < n; i++) { it[i].mode = mode[i]; it[i].h = h[i]; it[i].v = v[i]; it[i].d = d[i]; it[i].q = q[i]; }
    int c = select_core(n, it, axes_mask, cap, cfg, out);
    free(it);
    return c;
}

/* ----------------------------------------------------- station dominance */
typedef struct {
    const int *np;            /* profiles per station */
    const int8_t *pmode;      /* [S*P] */
    const float *ph, *pv;     /* [S*P] */
    const int8_t *smode;      /* best mode of the station */
    const float *shmin, *svmax;
    int P;
    double tau_h, tau_v;
} Prof;

/* station a dominates station b: every profile of b is matched by a profile of a (mode >=, headway <= within tol, speed >= within tol) */
static int st_dominates(const Prof *pf, int a, int b) {
    if (pf->smode[a] < pf->smode[b]) return 0;
    if (pf->shmin[a] > pf->shmin[b] * (1.0 + pf->tau_h)) return 0;
    if (pf->svmax[a] < pf->svmax[b] * (1.0 - pf->tau_v)) return 0;
    for (int j = 0; j < pf->np[b]; j++) {
        int ok = 0;
        for (int i = 0; i < pf->np[a] && !ok; i++) {
            if (pf->pmode[a * pf->P + i] >= pf->pmode[b * pf->P + j] && pf->ph[a * pf->P + i] <= pf->ph[b * pf->P + j] * (1.0 + pf->tau_h) &&
                pf->pv[a * pf->P + i] >= pf->pv[b * pf->P + j] * (1.0 - pf->tau_v)) ok = 1;
        }
        if (!ok) return 0;
    }
    return 1;
}

static double st_ptlos(const Prof *pf, int s, double d, const double *sp) {
    double best = 0;
    for (int i = 0; i < pf->np[s]; i++) {
        double p = ptlos_c(d, pf->pmode[s * pf->P + i], pf->ph[s * pf->P + i], pf->pv[s * pf->P + i], sp);
        if (p > best) best = p;
    }
    return best;
}

/* -------------------------------------------------------- label search */
typedef struct { float d; int node; int src; } HItem;
typedef struct { HItem *a; long n, cap; } Heap;
static void hpush(Heap *h, HItem x) {
    if (h->n == h->cap) { h->cap = h->cap ? h->cap * 2 : 1024; h->a = (HItem *)realloc(h->a, sizeof(HItem) * h->cap); }
    long i = h->n++;
    while (i > 0) { long p = (i - 1) / 2; if (h->a[p].d <= x.d) break; h->a[i] = h->a[p]; i = p; }
    h->a[i] = x;
}
static HItem hpop(Heap *h) {
    HItem top = h->a[0], x = h->a[--h->n];
    long i = 0;
    for (;;) {
        long c = 2 * i + 1; if (c >= h->n) break;
        if (c + 1 < h->n && h->a[c + 1].d < h->a[c].d) c++;
        if (h->a[c].d >= x.d) break;
        h->a[i] = h->a[c]; i = c;
    }
    if (h->n) h->a[i] = x;
    return top;
}

/* Multi-source, multi-label Dijkstra. Sources: ns (node, offset, station). Every node keeps <= Kc labels (distinct stations, in order of distance);
 * a label dominated by a closer label at the same node is dropped and not propagated. stats[0] = labels stored, [1] = dropped by dominance, [2] = nodes that hit the Kc cap. */
void multilabel_search(int n, const int *indptr, const int *adj, const float *w, int ns, const int *src_node, const float *src_off, const int *src_station,
                       int Kc, float maxd, int P, const int *np, const int8_t *pmode, const float *ph, const float *pv, const int8_t *smode,
                       const float *shmin, const float *svmax, const double *cfg, int *lab_station, float *lab_dist, int *lab_n, uint8_t *lab_cap, long *stats) {
    Prof pf = {np, pmode, ph, pv, smode, shmin, svmax, P, cfg[3], cfg[4]};
    Heap hp = {0, 0, 0};
    for (int i = 0; i < ns; i++) { HItem x = {src_off[i], src_node[i], src_station[i]}; if (x.d <= maxd) hpush(&hp, x); }
    long stored = 0, dom = 0, capped = 0;
    while (hp.n) {
        HItem x = hpop(&hp);
        int v = x.node, k = lab_n[v], dup = 0, dominated = 0;
        for (int i = 0; i < k && !dup; i++) {
            if (lab_station[(long)v * Kc + i] == x.src) dup = 1;
            else if (st_dominates(&pf, lab_station[(long)v * Kc + i], x.src)) dominated = 1;
        }
        if (dup) continue;
        if (dominated) { dom++; continue; }
        if (k >= Kc) { capped++; if (lab_cap[v] < 255) lab_cap[v]++; continue; }
        lab_station[(long)v * Kc + k] = x.src; lab_dist[(long)v * Kc + k] = x.d; lab_n[v] = k + 1; stored++;
        for (int e = indptr[v]; e < indptr[v + 1]; e++) {
            float nd = x.d + w[e];
            if (nd <= maxd) { HItem y = {nd, adj[e], x.src}; hpush(&hp, y); }
        }
    }
    free(hp.a);
    stats[0] = stored; stats[1] = dom; stats[2] = capped;
}

/* ------------------------------------------------------------- cells */
/* Edge-cell records (sorted by cell): for cell c, records rec_off[c]..rec_off[c+1]: edge endpoints u,v and the interval [a,b] the edge spends in the cell
 * (metres from u) and the edge length L. Candidate distance of station s = min over records of min(d_u + a, d_v + L - b), plus R. Then floors (p_min),
 * station dominance in distance order, the cap. out_station[nc*K], out_dist[nc*K] (uint16, 65535 = empty), out_flag[nc] bit0 = capped (candidates > K),
 * bit1 = more than Kcand candidates were found (list cut). */
void process_cells(int nc, const int *rec_off, const int *ru, const int *rv, const float *ra, const float *rb, const float *rl, float R,
                   const int *lab_station, const float *lab_dist, const int *lab_n, const uint8_t *lab_cap, int Kc, int K, int Kcand, int P, const int *np, const int8_t *pmode,
                   const float *ph, const float *pv, const int8_t *smode, const float *shmin, const float *svmax, const double *sp, const double *cfg,
                   int *out_station, uint16_t *out_dist, uint8_t *out_flag, uint8_t *out_cropped) {
    Prof pf = {np, pmode, ph, pv, smode, shmin, svmax, P, cfg[3], cfg[4]};
    int cap_tmp = 2 * Kc * 64 + 16;
    #pragma omp parallel
    {
    int *cs = (int *)malloc(sizeof(int) * cap_tmp);
    float *cd = (float *)malloc(sizeof(float) * cap_tmp);
    #pragma omp for schedule(dynamic, 256)
    for (int c = 0; c < nc; c++) {
        int m = 0, nodecap = 0;
        for (int r = rec_off[c]; r < rec_off[c + 1]; r++) {
            for (int side = 0; side < 2; side++) {
                int node = side ? rv[r] : ru[r];
                float add = side ? rl[r] - rb[r] : ra[r];
                if (add < 0) add = 0;
                if (lab_cap[node]) nodecap = 1;
                for (int i = 0; i < lab_n[node]; i++) {
                    int s = lab_station[(long)node * Kc + i]; float d = lab_dist[(long)node * Kc + i] + add + R;
                    int f = -1;
                    for (int j = 0; j < m; j++) if (cs[j] == s) { f = j; break; }
                    if (f >= 0) { if (d < cd[f]) cd[f] = d; }
                    else if (m < cap_tmp) { cs[m] = s; cd[m] = d; m++; }
                }
            }
        }
        /* sort by distance (insertion), floors, dominance */
        for (int a = 1; a < m; a++) { int xs = cs[a]; float xd = cd[a]; int b = a - 1; while (b >= 0 && cd[b] > xd) { cs[b + 1] = cs[b]; cd[b + 1] = cd[b]; b--; } cs[b + 1] = xs; cd[b + 1] = xd; }
        int nk = 0; int kept[256]; double kq[256];
        uint8_t flag = nodecap ? 4 : 0; int extra = 0;
        for (int a = 0; a < m && nk < 256; a++) {
            double q = st_ptlos(&pf, cs[a], cd[a], sp);
            if (q < cfg[8]) continue;
            int dom = 0;
            for (int b = 0; b < nk && !dom; b++) if (st_dominates(&pf, cs[kept[b]], cs[a])) dom = 1;
            if (dom) continue;
            if (nk >= Kcand) { flag |= 2; extra++; continue; }   /* a non-dominated candidate beyond the list limit: genuinely cropped */
            kept[nk] = a; kq[nk] = q; nk++;
        }
        { int cr = (nk > K ? nk - K : 0) + extra; out_cropped[c] = cr > 255 ? 255 : (uint8_t)cr; }
        for (int j = 0; j < K; j++) { out_station[(long)c * K + j] = -1; out_dist[(long)c * K + j] = 65535; }
        if (nk <= K) {
            for (int j = 0; j < nk; j++) { out_station[(long)c * K + j] = cs[kept[j]]; float dd = cd[kept[j]]; out_dist[(long)c * K + j] = dd > 65534 ? 65534 : (uint16_t)(dd + 0.5f); }
        } else {
            flag |= 1;
            Item *it = (Item *)malloc(sizeof(Item) * nk);
            for (int j = 0; j < nk; j++) { int s = cs[kept[j]]; it[j].mode = smode[s]; it[j].h = shmin[s]; it[j].v = svmax[s]; it[j].d = cd[kept[j]]; it[j].q = kq[j]; }
            int *sel = (int *)malloc(sizeof(int) * K);
            int cnt = select_core(nk, it, (1 << AX_DIST) | (1 << AX_HEAD) | (1 << AX_MODE) | (1 << AX_SPEED), K, cfg, sel);
            for (int a = 1; a < cnt; a++) { int x = sel[a], b = a - 1; while (b >= 0 && sel[b] > x) { sel[b + 1] = sel[b]; b--; } sel[b + 1] = x; }  /* distance order */
            for (int j = 0; j < cnt; j++) { out_station[(long)c * K + j] = cs[kept[sel[j]]]; float dd = cd[kept[sel[j]]]; out_dist[(long)c * K + j] = dd > 65534 ? 65534 : (uint16_t)(dd + 0.5f); }
            free(it); free(sel);
        }
        out_flag[c] = flag;
    }
    free(cs); free(cd);
    }
}

/* Fast recomputation: cell value = max over its stations and their profiles of PTLOS(distance) under the parameter vector sp (0-100 scale).
 * out_slot / out_prof = which station slot and which profile won (-1 when the cell has none). */
void recompute_cells(int nc, int K, int P, const int *cell_station, const uint16_t *cell_dist, const int *np, const int8_t *pmode, const float *ph, const float *pv,
                     const double *sp, float *out_los, int8_t *out_slot, int8_t *out_prof) {
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < nc; c++) {
        double best = 0; int bs = -1, bp = -1;
        for (int k = 0; k < K; k++) {
            int s = cell_station[(long)c * K + k];
            if (s < 0) continue;
            double d = cell_dist[(long)c * K + k];
            for (int j = 0; j < np[s]; j++) {
                double p = ptlos_c(d, pmode[(long)s * P + j], ph[(long)s * P + j], pv[(long)s * P + j], sp);
                if (p > best) { best = p; bs = k; bp = j; }
            }
        }
        out_los[c] = (float)(best * 100.0); out_slot[c] = (int8_t)bs; out_prof[c] = (int8_t)bp;
    }
}

/* stop scores [S*P] (0 for empty slots) */
void recompute_stops(int S, int P, const int8_t *pmode, const float *ph, const float *pv, const double *sp, float *out) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)S * P; i++) out[i] = pmode[i] < 0 ? 0.0f : (float)stop_score_c(pmode[i], ph[i], pv[i], sp);
}

/* Same as recompute_cells but with the stop scores [S*P] already computed (recompute_stops): a cell then only needs the walk decay. */
void recompute_cells_ss(int nc, int K, int P, const int *cell_station, const uint16_t *cell_dist, const int *np, const int8_t *pmode, const float *ss,
                        const double *sp, float *out_los, int8_t *out_slot, int8_t *out_prof) {
    double ws = sp[9];
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < nc; c++) {
        double best = 0; int bs = -1, bp = -1;
        for (int k = 0; k < K; k++) {
            int s = cell_station[(long)c * K + k];
            if (s < 0) continue;
            double t = cell_dist[(long)c * K + k] / ws / 60.0;
            for (int j = 0; j < np[s]; j++) {
                double q = ss[(long)s * P + j];
                double p = q * walk_decay_t(t, q, pmode[(long)s * P + j], sp);
                if (p > best) { best = p; bs = k; bp = j; }
            }
        }
        out_los[c] = (float)(best * 100.0); out_slot[c] = (int8_t)bs; out_prof[c] = (int8_t)bp;
    }
}
