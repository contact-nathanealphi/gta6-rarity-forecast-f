#!/usr/bin/env python3
"""
GTA VI trophy rarity forecast -- make.fun contest gta6-trophy-rarity-forecast-01.

Idea in one sentence: the earn rate of a trophy is (holders) / (players
tracked), and holders and players move for different reasons, so we forecast
them separately and divide at the end.

  * Players N(t) grow in jumps: the stats page refreshes every few hours and
    a batch of newly tracked players appears at once. The jumps shrink over
    time (the first jump is ~6x, later ones ~1.2-1.6x), and total growth is
    capped at 5x the last observed count.
  * Holders are tracked per COHORT. A cohort is the batch of players who
    arrived together at one refresh. Each cohort starts holding a trophy at
    the launch new-cohort rate rho0 (low: a player who just joined has only
    played a few hours) and then progresses toward the trophy's asymptote A
    at speed h. This matters because the old cohort -- the players who joined
    on day 1 and have played for a week -- is what lifts the rate to
    20-50% for the first chapter by day 7, while the fresh batches keep
    arriving near 0%. A single pooled hazard would give the brand-new
    players the launch cohort's speed and double-count progression.
  * The new cohort's rate at arrival is the launch value rho0, unless the
    history itself shows it trending (two or more refreshes with measurable
    new-cohort rates), in which case the fitted trend is extrapolated.
  * When the history actually shows progression (rates creeping between
    refreshes, several refreshes), the fitted hazard / new-cohort trend /
    player growth override the priors. When the history is flat (the page
    has not refreshed), the priors do the work. The blend is done in log
    space so a large disagreement between fit and prior still lands near
    the fit when the history is informative. There is no absolute hazard
    floor: a trophy's speed is scaled by its own asymptote, so rare
    trophies get almost no growth and no ghost holders are invented.

Uncertainty: a small deterministic ensemble of scenarios (refresh timing,
growth, progression speed, cohort maturity varied one factor at a time)
gives the median and the dispersion. The reported band is symmetric in
logit space and widens with the horizon, so a long-horizon forecast never
bleeds several points on a single trophy because its band was narrow in
the wrong place.

Standard library only. No network. No model calls. Deterministic: the same
input always produces the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)  # quantiles 5/25/50/75/95


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    p = clamp(p, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x)) if x < 700 else 1.0
    ex = math.exp(x)
    return ex / (1.0 + ex)


def parse_time(s):
    """ISO-8601 with Z or offset, or a bare date, -> aware datetime (UTC)."""
    if s is None:
        return None
    s = str(s).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def linfit(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return (my - b * mx, b)


# --------------------------------------------------------------------------
# trophy classes.
# Each class carries: asymptote A (share of players who ever earn it),
# prior hazard h (per hour, speed of a cohort approaching A), launch
# new-cohort rate rho0 (share of a fresh batch holding it on launch
# night), and a q95 cap (a sanity ceiling applied after the band is
# widened). Values come from the contest spec facts: story length
# ~68-80h, first chapter reachable in hours, final mission and platinum
# take days/weeks, and the launch-night dilution arithmetic (a fresh
# batch holds the first chapter at ~0.34%, fishing at ~0.99%, most
# others near 0%).
# --------------------------------------------------------------------------

# How far a fresh batch's rate may rise toward the asymptote as the player
# base matures (the "rho rise"). Early trophies (small play-hour
# threshold) rise a lot; late trophies (final mission, platinum) barely
# rise -- a player who joined yesterday has not played 60-250 hours yet.
RISE_FRAC = {
    "platinum": 0.01, "completion": 0.01, "first": 0.0,
    "ch1": 0.15, "ch2": 0.10, "ch3": 0.08, "ch4": 0.06, "final": 0.02,
    "story": 0.10, "strand": 0.03, "fishing": 0.20, "easy": 0.15,
    "side": 0.08, "grind": 0.02,
}
CLASSES = {
    "platinum":  (0.008, 0.0008, 0.0000, 0.020),
    "completion": (0.010, 0.0007, 0.0000, 0.020),
    "first":     (1.000, 0.0000, 1.0000, 0.9999),
    "ch1":       (0.70,  0.0085, 0.0034, 0.55),
    "ch2":       (0.60,  0.0050, 0.0006, 0.45),
    "ch3":       (0.55,  0.0045, 0.0006, 0.45),
    "ch4":       (0.50,  0.0040, 0.0006, 0.45),
    "final":     (0.25,  0.0035, 0.0006, 0.30),
    "story":     (0.45,  0.0040, 0.0006, 0.50),
    "strand":    (0.15,  0.0020, 0.0002, 0.30),
    "fishing":   (0.40,  0.0080, 0.0099, 0.40),
    "easy":      (0.25,  0.0040, 0.0004, 0.40),
    "side":      (0.12,  0.0020, 0.0006, 0.30),
    "grind":     (0.03,  0.0007, 0.0002, 0.10),
}

GRIND_WORDS = ("stunt", "collectible", "challenge", "vault", "launder",
               "garage", "gold medal", "drone race", "races", "meditate",
               "hunt", "waink", "100 locations", "50 challenges",
               "shooting range", "100% completion")
EASY_WORDS = ("arcade", "cocktail", "showroom", "bowling", "departure",
              "rhino", "choices made", "melee", "partnership",
              "second time", "what the hell", "discover", "explore",
              "activity once", "combat focus", "wanted level",
              "jewelry store", "armored truck", "illegal operation",
              "vehicle", "shops", "ocean beach")


def classify(trophy, last_rate):
    """Assign a trophy to a class from its description, tier and rate."""
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    if tier == "platinum":
        return "platinum"
    if "100% completion" in desc:
        return "completion"
    if last_rate is not None and last_rate >= 99.5:
        return "first"
    if "chapter 1" in desc:
        return "ch1"
    if "chapter 2" in desc:
        return "ch2"
    if "chapter 3" in desc:
        return "ch3"
    if "chapter 4" in desc:
        return "ch4"
    if "final mission" in desc:
        if " for " in desc or "strands" in desc:
            return "strand"
        return "final"
    if "fishing" in desc:
        return "fishing"
    if any(w in desc for w in GRIND_WORDS):
        return "grind"
    if any(w in desc for w in EASY_WORDS):
        return "easy"
    if desc.startswith("complete"):
        return "story"
    return "side"


# --------------------------------------------------------------------------
# history analysis
# --------------------------------------------------------------------------

class History(object):
    """What the case history actually tells us, fitted where possible."""

    def __init__(self, case):
        trophies = case.get("trophies") or []
        self.n = len(trophies)
        self.classes = []
        self.asym = []
        self.h_prior = []
        self.rho0 = []
        self.cap = []
        reads = sorted(case.get("history") or [],
                       key=lambda r: parse_time(r.get("at")) or datetime.min.replace(tzinfo=timezone.utc))
        self.reads = reads
        if reads:
            last = reads[-1]
            self.t_last_read = parse_time(last.get("at"))
            self.n_last = float(last.get("players_tracked") or 0) or None
            self.r_last = [float(x) / 100.0 for x in (last.get("rates") or [])]
        else:
            self.t_last_read = parse_time(case.get("at"))
            self.n_last = None
            self.r_last = [0.0] * self.n
        # The forecast starts at the case checkpoint `at`, NOT at the last
        # read: the horizon is measured from `at`. The state (N, rates) is
        # the last read's, because the page has not refreshed since.
        self.t_last = parse_time(case.get("at")) or self.t_last_read
        while len(self.r_last) < self.n:
            self.r_last.append(0.0)
        for i, t in enumerate(trophies):
            cls = classify(t, self.r_last[i] * 100.0 if i < len(self.r_last) else 0.0)
            self.classes.append(cls)
            a, h, r0, cp = CLASSES[cls]
            self.asym.append(a)
            self.h_prior.append(h)
            self.rho0.append(r0)
            self.cap.append(cp)
        rel = parse_time((case.get("game") or {}).get("release_date"))
        if reads:
            first_t = parse_time(reads[0].get("at"))
            if first_t is not None and (rel is None or first_t < rel):
                rel = first_t
        self.release = rel or self.t_last
        self._analyse(reads)

    def _analyse(self, reads):
        """Find refreshes (N jumps), same-N holder growth, and fit trends."""
        self.jumps = []          # (time, n_before, n_after, [rho per trophy])
        self.growth = []         # (dt_hours, [dE per trophy], N, [r1 per trophy])
        self.n_refresh_times = []
        prev = None
        for r in reads:
            t = parse_time(r.get("at"))
            n = r.get("players_tracked")
            rates = r.get("rates") or []
            if t is None or n is None or len(rates) < self.n:
                prev = (t, n, rates)
                continue
            n = float(n)
            if prev is not None and prev[0] is not None and prev[1]:
                dt_h = (t - prev[0]).total_seconds() / 3600.0
                if dt_h <= 0:
                    prev = (t, n, rates)
                    continue
                if abs(n - prev[1]) > 0.5:
                    rhos = []
                    for i in range(self.n):
                        r1 = float(prev[2][i]) / 100.0
                        r2 = float(rates[i]) / 100.0
                        rho = (r2 * n - r1 * prev[1]) / (n - prev[1])
                        rhos.append(clamp(rho, 0.0, 1.0))
                    self.jumps.append((t, prev[1], n, rhos))
                    self.n_refresh_times.append(t)
                else:
                    dE = []
                    r1s = []
                    moved = False
                    for i in range(self.n):
                        r1 = float(prev[2][i]) / 100.0
                        r2 = float(rates[i]) / 100.0
                        r1s.append(r1)
                        dE.append((r2 - r1) * n)
                        if abs(r2 - r1) > 1e-9:
                            moved = True
                    if moved:
                        self.growth.append((dt_h, dE, n, r1s))
            prev = (t, n, rates)

        ng = len(self.growth)
        nj = len(self.jumps)
        self.w_haz = min(0.9, ng / (ng + 3.0)) if ng else 0.0
        self.w_rho = min(0.9, nj / (nj + 2.0)) if nj else 0.0
        self.w_n = self.w_rho

        # Fitted hazard per trophy, from same-N holder growth. Zero or
        # negative growth counts as zero: a trophy that never grows in a
        # history full of informative intervals has a ~zero hazard, not
        # the prior. (Only used when the history is informative.)
        # Median of per-interval ratios: robust to the page rounding
        # rates to 2 decimals (an aggregate sum accumulates the positive
        # rounding steps and invents ghost holders for rare trophies).
        self.h_fit = [None] * self.n
        for i in range(self.n):
            obs = []
            for dt_h, dE, n, r1s in self.growth:
                if dt_h <= 0:
                    continue
                room = self.asym[i] * n - r1s[i] * n
                if room > 1.0:
                    obs.append(max(0.0, (dE[i] / dt_h) / room))
            if obs:
                self.h_fit[i] = median(obs)

        # Observed new-cohort rates, per trophy (median across jumps).
        self.rho_obs = [None] * self.n
        for i in range(self.n):
            vals = [j[3][i] for j in self.jumps]
            if vals:
                self.rho_obs[i] = median(vals)
        # Fitted trend of the new-cohort rate, pooled per class in log
        # space: slope of log(rho) against maturity in days.
        self.rho_slope = {}     # class -> slope per day
        self.rho_mat_last = {}  # class -> maturity (days) at the last jump
        by_class = {}
        for (t, n1, n2, rhos) in self.jumps:
            mat = (t - self.release).total_seconds() / 86400.0
            for i, rho in enumerate(rhos):
                if rho > 0:
                    by_class.setdefault(self.classes[i], []).append((mat, math.log(rho)))
        for cls, pts in by_class.items():
            if len(pts) >= 2:
                fit = linfit([p[0] for p in pts], [p[1] for p in pts])
                if fit is not None and -1.0 < fit[1] < 1.0:
                    self.rho_slope[cls] = fit[1]
                    self.rho_mat_last[cls] = max(p[0] for p in pts)

        # Fitted N(t): log N = a + b*log(1 + hours since release)
        self.n_fit = None
        pts = []
        for r in reads:
            t = parse_time(r.get("at"))
            n = r.get("players_tracked")
            if t is None or not n:
                continue
            hrs = max(0.0, (t - self.release).total_seconds() / 3600.0)
            pts.append((math.log1p(hrs), math.log(float(n))))
        if len(pts) >= 3:
            fit = linfit([p[0] for p in pts], [p[1] for p in pts])
            if fit is not None and 0.05 < fit[1] < 1.5:
                self.n_fit = fit

        gaps = []
        for a, b in zip(self.n_refresh_times, self.n_refresh_times[1:]):
            g = (b - a).total_seconds() / 3600.0
            if 0 < g < 72:
                gaps.append(g)
        self.gap_obs = median(gaps) if gaps else None

    # -- blended parameters (log space) ------------------------------------

    def hazard(self, i):
        """Per-trophy progression speed, fit blended with prior."""
        hp = self.h_prior[i]
        hf = self.h_fit[i]
        if hf is None or self.w_haz <= 0:
            return hp
        hf = max(hf, 1e-6)
        lh = self.w_haz * math.log(hf) + (1.0 - self.w_haz) * math.log(max(hp, 1e-9))
        return math.exp(lh)

    def rho_base(self, i):
        """New-cohort rate at arrival, observed value blended with prior."""
        r0 = self.rho0[i]
        ro = self.rho_obs[i]
        if ro is None or self.w_rho <= 0:
            return r0
        ro = max(ro, 1e-5)
        if r0 <= 0:
            return ro
        lr = self.w_rho * math.log(ro) + (1.0 - self.w_rho) * math.log(r0)
        return math.exp(lr)

    def rho_at(self, i, when):
        """New-cohort rate for a batch arriving at `when`."""
        base = self.rho_base(i)
        cls = self.classes[i]
        if cls == "first":
            # the opening trophy: every batch holds it (~100%).
            return base
        slope = self.rho_slope.get(cls)
        if slope is not None:
            mat = (when - self.release).total_seconds() / 86400.0
            mat_last = self.rho_mat_last.get(cls, 0.0)
            val = base * math.exp(slope * max(0.0, mat - mat_last))
            return clamp(val, 0.0, self.asym[i])
        # No fitted trend: the new cohort's rate rises as the player base
        # matures -- a batch arriving on day 2-3 has played longer before
        # being tracked than a launch-night batch. Delayed 12h (the first
        # refresh is untouched) and rising only toward a class-dependent
        # fraction of the asymptote (RISE_FRAC), so late trophies barely
        # rise at all.
        hrs = (when - self.release).total_seconds() / 3600.0
        m = 1.0 - math.exp(-max(0.0, hrs - 12.0) / 48.0)
        target = self.asym[i] * RISE_FRAC.get(cls, 0.05)
        val = base + (target - base) * m
        return clamp(val, 0.0, self.asym[i])


# --------------------------------------------------------------------------
# simulation (cohort model)
# --------------------------------------------------------------------------

def simulate(hist, horizon_h, gap_mul, grow_mul, haz_mul):
    """One scenario: hourly steps from the checkpoint to the target.

    Cohorts: each refresh adds a cohort (size dN, holders rho*dN); every
    cohort progresses toward its own asymptote at the trophy hazard.

    The answer the contest scores is the stats PAGE state, and the page
    only changes at refreshes. So the returned rate is the state at the
    last refresh at or before the target -- or the first refresh inside
    (target, target+60min] when one lands there -- never the real-time
    holder count after that refresh. Progress made after the last
    refresh is invisible to the page and must not be forecast.
    """
    n = hist.n
    t0 = hist.t_last if hist.t_last is not None else hist.release
    n_cur = hist.n_last if hist.n_last else 638.0
    # cohorts: list of [size, [holders per trophy]]
    cohorts = [[n_cur, [hist.r_last[i] * n_cur for i in range(n)]]]
    haz = [hist.hazard(i) * haz_mul for i in range(n)]

    def current_rates():
        e_tot = [0.0] * n
        for size, e in cohorts:
            for i in range(n):
                e_tot[i] += e[i]
        return [clamp(e_tot[i] / n_cur, 0.0, 1.0) if n_cur > 0 else 0.0
                for i in range(n)]

    target_t = t0.timestamp() + horizon_h * 3600.0
    # snapshots of the page state: (time, rates). Starts at the checkpoint.
    snaps = [(t0.timestamp(), list(hist.r_last), n_cur)]

    gap = (hist.gap_obs if hist.gap_obs else 10.0) * gap_mul
    if hist.jumps:
        last_ref = hist.n_refresh_times[-1].timestamp()
        if last_ref + gap * 3600.0 > t0.timestamp():
            nxt = last_ref + gap * 3600.0
            missed = 0
        else:
            # Stale history: refreshes happened between the last observed
            # refresh and the checkpoint that we never saw. Count them so
            # the next jump ratio is a late one, not the second-refresh
            # ratio, and schedule the next refresh on the same cadence.
            missed = int((t0.timestamp() - last_ref) / (gap * 3600.0))
            nxt = last_ref + (missed + 1) * gap * 3600.0
            if nxt <= t0.timestamp():
                nxt = t0.timestamp() + gap * 3600.0
    else:
        missed = 0
        # First refresh lands ~4.5h after release (spec fact). It cannot
        # be before the checkpoint: the history shows no refresh has
        # happened yet, so if the estimate falls before the checkpoint the
        # refresh is overdue and lands shortly after it.
        first = hist.release.timestamp() + 4.5 * 3600.0 * gap_mul
        nxt = max(first, t0.timestamp() + 0.5 * 3600.0 * gap_mul)
    prior_ratios = [6.06, 1.60, 1.42, 1.34, 1.29, 1.25, 1.21, 1.18, 1.15]
    jump_idx = len(hist.jumps) + missed
    n_start = n_cur
    n_cap = n_start * 5.0
    first_jump_done = bool(hist.jumps)

    steps = int(math.ceil(horizon_h))
    for s in range(steps):
        t_now = t0.timestamp() + s * 3600.0
        t_end = t0.timestamp() + min(s + 1, horizon_h) * 3600.0
        dt = (t_end - t_now) / 3600.0
        if dt <= 0:
            break

        def progress(frac):
            if frac <= 0:
                return
            for c in cohorts:
                size, e = c
                for i in range(n):
                    room = hist.asym[i] * size - e[i]
                    if room > 0:
                        e[i] += haz[i] * room * frac

        if t_now < nxt <= t_end:
            f = (nxt - t_now) / 3600.0
            progress(f)
            if hist.n_fit is not None and hist.w_n > 0:
                hrs = max(0.0, (nxt - hist.release.timestamp()) / 3600.0)
                n_fit = math.exp(hist.n_fit[0] + hist.n_fit[1] * math.log1p(hrs))
                ratio_fit = clamp(n_fit / n_cur, 1.0, 5.0) if n_cur > 0 else 1.0
            else:
                ratio_fit = None
            idx = min(jump_idx, len(prior_ratios) - 1)
            ratio_prior = prior_ratios[idx]
            if ratio_fit is not None:
                lr = hist.w_n * math.log(ratio_fit) + (1.0 - hist.w_n) * math.log(ratio_prior)
                ratio = math.exp(lr)
            else:
                ratio = ratio_prior
            ratio = 1.0 + (ratio - 1.0) * grow_mul
            # The 5x cap is on extrapolated growth. The first refresh is
            # a spec fact (6.06x), not an extrapolation, so it is exempt;
            # once it lands, the cap is 5x the new player count.
            if first_jump_done and n_cur * ratio > n_cap:
                ratio = n_cap / n_cur if n_cur > 0 else 1.0
            if ratio > 1.0:
                d_n = n_cur * (ratio - 1.0)
                when = datetime.fromtimestamp(nxt, tz=timezone.utc)
                new_e = [hist.rho_at(i, when) * d_n for i in range(n)]
                cohorts.append([d_n, new_e])
                n_cur *= ratio
                jump_idx += 1
                snaps.append((nxt, current_rates(), n_cur))
                nxt += gap * 3600.0
                if not first_jump_done:
                    first_jump_done = True
                    n_cap = n_cur * 5.0
            progress(dt - f)
        else:
            progress(dt)
    # the page state the contest scores: first refresh inside
    # (target, target+60min] if one exists, else the last refresh at or
    # before the target, else the checkpoint state.
    for (tt, rr, nn) in snaps:
        if target_t < tt <= target_t + 3600.0:
            return rr, nn
    best = snaps[0][1]
    best_n = snaps[0][2]
    for (tt, rr, nn) in snaps:
        if tt <= target_t:
            best = rr
            best_n = nn
    return best, best_n


def central_n_target(case):
    """Player count at the target in the central scenario (reporting)."""
    hist = History(case)
    horizon_h = float(case.get("horizon_minutes") or 0) / 60.0
    _, n = simulate(hist, horizon_h, 1.0, 1.0, 1.0)
    return n, hist.n_last


SCENARIOS = [
    (1.0, 1.0, 1.0),    # central
    (0.7, 1.0, 1.0),    # refresh sooner
    (1.4, 1.0, 1.0),    # refresh later
    (1.0, 0.75, 1.0),   # slower player growth
    (1.0, 1.30, 1.0),   # faster player growth
    (1.0, 1.0, 0.6),    # slower progression
    (1.0, 1.0, 1.7),    # faster progression
    (0.7, 1.30, 1.7),   # everything fast
    (1.4, 0.75, 0.6),   # everything slow
    (0.7, 0.75, 1.0),   # early refresh, slow growth
    (1.4, 1.30, 1.0),   # late refresh, fast growth
]


def wmedian(pairs):
    """Weighted median of (value, weight) pairs."""
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return median([v for v, _ in pairs])
    acc = 0.0
    for v, w in sorted(pairs):
        acc += w
        if acc >= 0.5 * tot:
            return v
    return sorted(pairs)[-1][0]


def wmean(pairs):
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return mean([v for v, _ in pairs])
    return sum(v * w for v, w in pairs) / tot


def forecast(case):
    """Return the list of [q05, q25, q50, q75, q95] (percent) per trophy."""
    hist = History(case)
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_h = horizon_min / 60.0
    days = max(horizon_h / 24.0, 0.0)

    scen_rates = []
    scen_w = []
    for (gm, gr, hm) in SCENARIOS:
        rates, _n = simulate(hist, horizon_h, gm, gr, hm)
        scen_rates.append(rates)
        scen_w.append(1.0)
    # Short horizon (<= 6h): mix in a "no refresh" branch. The next
    # refresh is scheduled, not random, so its weight is the probability
    # that the scheduled refresh lands after the target (normal schedule
    # uncertainty, sigma = 1h). At long horizon a refresh is certain and
    # the branch vanishes.
    p_nr = 0.0
    if horizon_h <= 6.0:
        t0 = hist.t_last if hist.t_last is not None else hist.release
        target_t = t0.timestamp() + horizon_h * 3600.0
        if hist.jumps:
            gap = hist.gap_obs if hist.gap_obs else 10.0
            t_exp = hist.n_refresh_times[-1].timestamp() + gap * 3600.0
            if t_exp <= t0.timestamp():
                t_exp = t0.timestamp() + gap * 3600.0
        else:
            t_exp = hist.release.timestamp() + 4.5 * 3600.0
        sigma = 3600.0
        z = (target_t - t_exp) / (sigma * math.sqrt(2.0))
        p_nr = 0.5 * math.erfc(z)
        p_nr = clamp(p_nr, 0.02, 0.9)
        if p_nr > 0.02:
            scen_rates.append(list(hist.r_last))
            scen_w.append(p_nr / (1.0 - p_nr) * len(SCENARIOS))
    n = hist.n
    out = []
    for i in range(n):
        pairs = [(logit(sr[i]), w) for sr, w in zip(scen_rates, scen_w)]
        med_lg = wmedian(pairs)
        mu = wmean(pairs)
        disp = math.sqrt(wmean([( (x - mu) ** 2, w) for x, w in pairs]))
        s = math.sqrt(disp * disp + (0.5 * math.sqrt(days)) ** 2 + 0.25 ** 2)
        s = clamp(s, 0.35, 1.5)
        cls = hist.classes[i]
        if cls == "first":
            # the opening trophy: pinned near 100%. The drop by day 7 is
            # only a sync glitch, so the long-horizon median stays high
            # (~99.4%) with an honest lower band (q05 ~98%).
            med = 0.9999 if days < 1.0 else 0.994
            s2 = clamp(0.30 + 0.35 * math.sqrt(days), 0.30, 1.30)
            if days >= 1.0:
                # widen just enough that q05 lands near 98%
                s2 = max(s2, (logit(0.994) - logit(0.98)) / 1.6448536269514722)
            qs = [inv_logit(logit(med) + z * s2) for z in Z]
        else:
            qs = [inv_logit(med_lg + z * s) for z in Z]
        cap = hist.cap[i]
        qs = [min(q, cap) for q in qs]
        qs = [clamp(q, 0.0001, 0.9999) for q in qs]
        qs = sorted(qs)
        out.append([round(q * 100.0, 4) for q in qs])
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        with open(os.path.join(in_dir, fn), "r", encoding="utf-8") as f:
            case = json.load(f)
        qs = forecast(case)
        trophies = case.get("trophies") or []
        rows = []
        for i, t in enumerate(trophies):
            rows.append({"id": int(t.get("id", i + 1)), "q": qs[i]})
        payload = {"trophies": rows}
        text = json.dumps(payload, separators=(",", ":"), sort_keys=False)
        out_path = os.path.join(out_dir, fn[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
