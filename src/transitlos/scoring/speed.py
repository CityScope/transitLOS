"""Speed scoring: converts an observed average operating speed into a score.

Implements `V(speed)`, scoring.md §2.4 (2026-09-29 THIRD SHAPE-FIX revision --
the fourth distinct calibration of this function in one day; see "Revision
history" below for the full honest record of all four):

    V(v) = M * v^n / (v^n + K^n)

a single continuous, infinitely differentiable ("Hill function" / generalized
Michaelis-Menten family) curve on `v >= 0`, solved so it passes exactly
through one literature-grounded anchor point -- `V(anchor_kmh) = 1.0` -- with
`M` (the asymptote) and `n` (the shape exponent) both fixed by design
judgment rather than fitted to a second point.

With the defaults (anchor=30, bonus=0.15 so M=1.15, n=2):

    K = anchor * (M-1)**(1/n) = 30 * sqrt(0.15) ~= 11.619
    V(v) = 1.15 * v^2 / (v^2 + 11.619^2)

giving `V(30) = 1.0` exactly and `V(v) -> 1.15` as `v -> infinity` (the
curve keeps rising, very slowly, past 1.15 in absolute terms it never
reaches -- e.g. `V(100) ~= 1.135`, `V(200) ~= 1.144`). There is no hard cap
and no requirement that the curve hit any specific value at any specific
high speed -- see "Revision history," Change 4, for why this project tried
and rejected pinning a second point twice before settling on this.

**IMPORTANT -- what "speed" means here**: `avg_speed_kmh` is COMMERCIAL /
OPERATING speed averaged over the whole route, INCLUDING dwell time at stops
-- not pure between-stops cruising speed (which is substantially higher and
not what this function, or the anchor benchmark it uses, are measuring). See
"Anchor" in scoring.md §2.4 for the specific metro sources and this
distinction stated per-source.

## Revision history (all on 2026-09-29; kept as an honest, non-overwritten record)

**Revision 1 (SHAPE): piecewise-linear -> plain Michaelis-Menten.** The
original two-segment ramp (`V(v)=v/25` below a 25 km/h anchor, a shallower
capped ramp above it) was replaced with a smooth curve, `V(v)=M*v/(v+K)`,
fitted through `(25, 1.0)` and `(100, 1.15)`, after a user review found the
piecewise kink inelegant.

**Revision 2 (SHAPE-FIX): plain Michaelis-Menten -> Hill function, n=1 -> n=2.**
A user review found `V(5)~=0.59`/`V(10)~=0.79` -- unreasonably generous for
barely-faster-than-walking speeds. Diagnosis: a 2-parameter curve forced
through two exact points has no freedom left to independently control
low-speed behavior. Fix: relaxed the second point to a pure asymptote,
freeing a 3rd parameter `n`, fixed at `n=2` by design judgment (checked
numerically against `n=1, 1.5, 2, 2.5, 3`).

**Revision 3 (SECOND SHAPE-FIX/RE-ANCHOR): re-anchored 25->30 km/h against
real metro commercial-speed data, and re-introduced an exact second fitted
point at 80 km/h, matching German Regional-Express's real average commercial
speed (~70-90 km/h).** A follow-up request asked to ground *every*
calibration point in literature, including the anchor and ceiling, not just
the low-speed shape. The anchor move (25->30 km/h) was well-supported: four
major dense-network metro systems (London Underground ~33, Paris Metro
legacy lines ~30-35, NYC Subway ~30, Tokyo Metro ~30 km/h) independently
converge on this band, bracketed by UITP's general 25-45 km/h figure. The
second-point move (100->80 km/h) was NOT equally well-supported, as
Revision 4 (immediately below) concluded on further review.

**Revision 4 (THIRD SHAPE-FIX, this revision): dropped the exact second
fitted point entirely; the ceiling is now an explicit, honestly-flagged
design-judgment asymptote, not pinned to any named speed.** Direct user
review of Revision 3 rejected the German Regional-Express justification:
it is a single country's single named service tier (not corroborated by
other countries' equivalent fast-regional-rail categories the way the
30 km/h metro anchor was corroborated by four independent systems), and
using it to "pin" 80 km/h risked repeating the exact mistake Revision 3 was
supposed to fix in Revision 1's "100 km/h" -- a round-seeming number dressed
up as literature-derived when the underlying evidence doesn't actually bear
that weight. A follow-up search for a better-corroborated second point was
run specifically for this revision: German S-Bahn/RER-type urban rail
(Berlin S-Bahn ~40 km/h average, Berlin S-Bahn's own stated commercial-speed
*objective* of 55 km/h, Paris RER A's stated objective of ~49 km/h) turned
out to be SLOWER than the already-adopted 30 km/h metro anchor in most
cases, not meaningfully faster, so it cannot serve as a credible "faster
than metro" second point; and a search across Japan/Korea/China's semi-fast
"limited express" regional-rail tiers found only high-speed-rail-adjacent
services (Korea's ITX-Cheongchun ~180 km/h average) with no clean,
internationally-recognized "just above metro, well below HSR" tier
comparable in convergence-strength to the 4-system metro anchor. **No
second point was found that is corroborated the way the anchor is.**
Conclusion: rather than force a single-source citation (Regional-Express, or
any other single system/country) to carry more evidentiary weight than it
can bear, the exact second-point-fitting approach is abandoned. The ceiling
is instead kept as what it has always honestly been underneath -- a
design-judgment asymptote, `M = 1+ceiling_bonus = 1.15`, applied directly
(no second point to solve for) -- and this is now stated plainly rather than
dressed up as a second literature-fitted coordinate. This is structurally
identical to Revision 2's asymptote-only approach, just re-derived from the
new 30 km/h anchor instead of the old 25 km/h one; Revision 3's brief
excursion into an exact second point is kept in the record (scoring.md
§2.4, `parameters/07...md` §4i) as an honest account of an approach that was
tried and found not to hold up, not silently erased.

The `n=2` low-speed-shape fix from Revision 2 carries forward unchanged and
was re-verified at the current anchor (`V(5)~=0.180`, `V(10)~=0.489` under
`anchor=30, M=1.15, n=2`).

**A same-day follow-up sanity-checked (not re-derived) the `M=1.15` ceiling
against real high-speed rail commercial speeds**, without reinstating a
second fitted point: Japan's Tokaido Shinkansen (Nozomi, ~220 km/h), French
TGV Paris-Lyon (~200-207 km/h), and China's Beijing-Shanghai HSR (~292 km/h)
-- three independently-sourced, different-country systems, all real
timetabled commercial speeds -- evaluate to `V(v) ~= 1.146-1.148` under this
curve, all within 0.2 percentage points of the asymptote. This confirms the
existing ceiling is not obviously wrong (real HSR sits essentially at the
ceiling, not far below or requiring implausible speeds to approach it), but
it is a confirmation-by-inspection, not a new derivation -- no source states
a specific "HSR should score ~1.15" claim. `M`, `anchor`, and `n` are
unchanged by this check. Full writeup: scoring.md §2.4 Hyperparameters,
`parameters/07_speed_based_service_classification.md` §4k.

This retains everything from the prior revisions that is not about the
second-point/ceiling calibration: every positive speed still gets a
positive score (`V` is never zero except at `v=0`, a genuinely stationary
vehicle) since `stop_score` is a geometric-mean-family aggregation (§1.2)
where an exact zero on any one factor is disqualifying -- correct for "no
service," wrong for "merely slow." The old ITDP 11/20 km/h banded
classification remains documented, unused by this function, as an
independent §6 comparison method only
(`parameters/07_speed_based_service_classification.md` §1, §4c).

`V` is still allowed to exceed 1.0 (approaching `M=1.15` in the limit, from
below, monotonically, never reaching it at any finite speed):
`stop_score` is not guaranteed to stay within [0, 1] purely from this
factor's own boundedness -- see `stop_score.py`'s module docstring for how
the overall geometric mean is nonetheless kept a sensible bounded quantity
in practice.

Cross-checked against `transitLOS/documentation/scoring/scripts/
compare_methods.py`'s own `V_speed` reference implementation.
"""

from __future__ import annotations

from .parameters import REGIONS
from ._warnings import warn_if_unvalidated_region


def speed_score(
    avg_speed_kmh: float,
    region: str = "global",
    anchor_kmh: float | None = None,
    ceiling_bonus: float | None = None,
    shape_n: float | None = None,
) -> float:
    """Compute `V(speed)` from average operating speed.

    Args:
        avg_speed_kmh: Average operating (commercial) speed in km/h --
            i.e. distance/time averaged over the whole route, INCLUDING
            dwell time at stops, not pure between-stops cruising speed. Must
            be >= 0.
        region: Region key into `parameters.REGIONS`. Kept for backward
            compatibility -- every region now shares identical, universal
            parameters (see `parameters.py`'s module docstring).
        anchor_kmh: Speed at which `V` reaches exactly 1.0. Defaults to the
            region's `v_anchor_kmh` (30.0, re-derived from real metro
            commercial-speed data -- see module docstring and scoring.md
            §2.4 Anchor).
        ceiling_bonus: The curve's asymptotic ceiling above 1.0, approached
            as avg_speed_kmh -> infinity, never hit at any specific finite
            speed and NOT pinned to any named high-speed benchmark (an
            earlier same-day revision tried pinning it to German
            Regional-Express's speed and was reverted -- see module
            docstring, Revision 4). Defaults to the region's
            `v_ceiling_bonus` (0.15), an explicit design-judgment choice.
        shape_n: Hill-function exponent controlling how quickly the curve
            rises below the anchor (higher n => lower scores at low speed).
            Defaults to the region's `v_shape_n` (2.0).

    Returns:
        A float, strictly increasing in avg_speed_kmh, exactly 0.0 only at
        avg_speed_kmh == 0, asymptotically approaching (but never reaching)
        `1 + ceiling_bonus` as avg_speed_kmh -> infinity.

    Raises:
        ValueError: If avg_speed_kmh is negative.
    """
    if avg_speed_kmh < 0:
        raise ValueError(f"avg_speed_kmh must be >= 0, got {avg_speed_kmh}")

    warn_if_unvalidated_region(region)
    params = REGIONS[region]
    anchor = anchor_kmh if anchor_kmh is not None else params.v_anchor_kmh
    bonus = ceiling_bonus if ceiling_bonus is not None else params.v_ceiling_bonus
    n = shape_n if shape_n is not None else params.v_shape_n

    # Hill-function (generalized/power-law Michaelis-Menten) curve
    # V(v) = M*v^n/(v^n+K^n): M=1+bonus is the asymptote (an explicit
    # design-judgment ceiling, not fitted to any second point -- see module
    # docstring, Revision 4, for why a second fitted point was tried twice
    # and dropped both times), K is solved so V(anchor)=1.0 exactly, and n
    # is a fixed shape parameter chosen to keep the curve from rising too
    # steeply at low speed.
    m = 1.0 + bonus
    k = anchor * (m - 1.0) ** (1.0 / n)
    return m * avg_speed_kmh**n / (avg_speed_kmh**n + k**n)
