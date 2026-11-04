// V7 analysis: gaps, paired contrasts and mechanism counters on the complete sets.
//
//   node analysis/v7/analyse_v7.js [outdir]          (run from the repository root)
//
// Reads the V7 runs in analysis/v7/data/ and the archived V0/V5/V6 runs in
// data/final50k/merged-quality-records.jsonl.gz (seeds 1-10). Writes three CSVs to
// outdir (default analysis/v7/expected/). No solver is run.
//
// Gap = 100 * (mean makespan over the 10 seeds - reference) / reference, per instance.
// Reference: optimum for J30, critical-path lower bound for J60/J120 (same as the paper).
// Contrasts: paired Wilcoxon signed-rank (normal approximation, zeros dropped),
// bootstrap 95 % CI of the mean difference, Holm correction within each family:
//   primary        V7sb - V6                     (3 sets x 4 budgets = 12)
//   ablation_vs_V6 V7s, V7b, V7sbk1 - V6         (36)
//   ablation_vs_V7sb V7s, V7b, V7sbk1 - V7sb     (36)
const fs = require("fs"), path = require("path"), zlib = require("zlib");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = path.resolve(process.argv[2] || path.join(__dirname, "expected"));
const BUDGETS = [1000, 5000, 25000, 50000];
const SETS = ["j30", "j60", "j120"];
const V7 = ["V7sb", "V7sbk1", "V7b", "V7s"];
const ARCHIVE = ["V0", "V5", "V6"];
const ARMS = [...ARCHIVE, ...V7];

// --- reference values ---------------------------------------------------------------
const opt = {};
fs.readFileSync(path.join(ROOT, "solvers/br-deh/experiments/j30_optimal.csv"), "utf8")
  .split(/\r?\n/).slice(1).filter(Boolean).forEach(l => { const [i, m] = l.split(","); opt[i.trim()] = +m; });
function cpm(p) {   // critical-path length of a PSPLIB .sm file
  const L = fs.readFileSync(p, "utf8").split(/\r?\n/);
  const n = +L[5].trim().split(/\s+/).pop(); const succ = [], dur = [];
  for (let i = 0; i < n; i++) succ.push(L[18 + i].trim().split(/\s+/).slice(3).map(x => +x - 1));
  for (let i = 0; i < n; i++) dur.push(+L[18 + n + 4 + i].trim().split(/\s+/)[2]);
  const es = Array(n).fill(0);
  for (let i = 0; i < n; i++) for (const j of succ[i]) es[j] = Math.max(es[j], es[i] + dur[i]);
  return Math.max(...es.map((e, i) => e + dur[i]));
}
const refC = {};
const ref = i => refC[i] ?? (refC[i] = i.startsWith("j30/") ? opt[i.split("/")[1]] : cpm(path.join(ROOT, "solvers/br-deh/inputs", i)));

// --- load runs ----------------------------------------------------------------------
const csv = f => zlib.gunzipSync(fs.readFileSync(f)).toString().split(/\r?\n/).filter(Boolean);
const traj = {};          // "inst|seed|arm" -> [[schedules, makespan], ...]
const runs = [];
for (const arm of V7) {
  for (const l of csv(path.join(__dirname, "data", `${arm}_traj.csv.gz`)).slice(1)) {
    const [i, s, a, sc, mk] = l.split(","); (traj[`${i}|${s}|${a}`] ||= []).push([+sc, +mk]);
  }
  const ls = csv(path.join(__dirname, "data", `${arm}_runs.csv.gz`)); const h = ls[0].split(",");
  for (const l of ls.slice(1)) { const v = l.split(","); runs.push(Object.fromEntries(h.map((k, j) => [k, v[j]]))); }
}
const insts = [...new Set(runs.map(r => r.instance))];
const want = new Set(insts);
for (const l of csv(path.join(ROOT, "data/final50k/merged-quality-records.jsonl.gz"))) {
  const r = JSON.parse(l);
  if (!ARCHIVE.includes(r.variant) || !want.has(r.instance) || r.seed_index > 10) continue;
  traj[`${r.instance}|${r.seed_index}|${r.variant}`] = r.trajectory.map(t => [t[1], t[2]]);
}

// --- checks -------------------------------------------------------------------------
for (const arm of ARMS) for (const i of insts) for (let s = 1; s <= 10; s++)
  if (!traj[`${i}|${s}|${arm}`]) throw new Error(`missing run ${arm} ${i} seed ${s}`);
const infeasible = runs.filter(r => r.feasible !== "true").length;
if (infeasible) throw new Error(`${infeasible} infeasible V7 runs`);

// --- statistics ---------------------------------------------------------------------
const at = (t, B) => { let m = Infinity; for (const r of t) if (r[0] <= B && r[1] < m) m = r[1]; return m; };
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
function wilcoxon(d) {
  const z = d.filter(x => x !== 0).map(x => ({ a: Math.abs(x), s: Math.sign(x) })).sort((p, q) => p.a - q.a);
  const n = z.length; if (!n) return 1;
  let i = 0; const rk = Array(n);
  while (i < n) { let j = i; while (j + 1 < n && z[j + 1].a === z[i].a) j++; for (let k = i; k <= j; k++) rk[k] = (i + j) / 2 + 1; i = j + 1; }
  let wp = 0; for (let k = 0; k < n; k++) if (z[k].s > 0) wp += rk[k];
  const mu = n * (n + 1) / 4, sd = Math.sqrt(n * (n + 1) * (2 * n + 1) / 24);
  const x = Math.abs((wp - mu) / sd) / Math.SQRT2, t = 1 / (1 + 0.5 * x);   // erfc, Numerical Recipes
  return t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
}
function boot(d, reps = 10000) {   // percentile bootstrap; fixed-seed mulberry32 so the CI is reproducible
  let s = 12345;
  const rnd = () => { s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const m = []; for (let r = 0; r < reps; r++) { let t = 0; for (let k = 0; k < d.length; k++) t += d[Math.floor(rnd() * d.length)]; m.push(t / d.length); }
  m.sort((a, b) => a - b); return [m[Math.floor(0.025 * reps)], m[Math.floor(0.975 * reps)]];
}
function holm(ps) {
  const o = ps.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0]); const out = Array(ps.length); let prev = 0;
  o.forEach(([p, i], r) => { prev = Math.max(prev, Math.min(1, (ps.length - r) * p)); out[i] = prev; }); return out;
}

// per-instance gaps: G[set][budget][arm] = [gap per instance]
const G = {}, setInsts = {};
for (const set of SETS) {
  setInsts[set] = insts.filter(i => i.startsWith(set + "/")); G[set] = {};
  for (const B of BUDGETS) {
    G[set][B] = {};
    for (const arm of ARMS) G[set][B][arm] = setInsts[set].map(i => {
      const v = []; for (let s = 1; s <= 10; s++) v.push(at(traj[`${i}|${s}|${arm}`], B));
      return 100 * (mean(v) - ref(i)) / ref(i);
    });
  }
}

const f3 = x => x.toFixed(3), fp = p => p.toExponential(2);
fs.mkdirSync(OUT, { recursive: true });

// gaps
const gapRows = ["set,n_instances,budget," + ARMS.join(",")];
for (const set of SETS) for (const B of BUDGETS)
  gapRows.push([set, setInsts[set].length, B, ...ARMS.map(a => f3(mean(G[set][B][a])))].join(","));
fs.writeFileSync(path.join(OUT, "v7_gaps.csv"), gapRows.join("\n") + "\n");

// contrasts
const families = {
  primary: [["V7sb", "V6"]],
  ablation_vs_V6: [["V7s", "V6"], ["V7b", "V6"], ["V7sbk1", "V6"]],
  ablation_vs_V7sb: [["V7s", "V7sb"], ["V7b", "V7sb"], ["V7sbk1", "V7sb"]],
};
const cRows = ["family,set,budget,arm,reference,diff_pp,ci95_lo,ci95_hi,better,tie,worse,p,p_holm"];
for (const [fam, pairs] of Object.entries(families)) {
  const rows = [];
  for (const set of SETS) for (const B of BUDGETS) for (const [a, b] of pairs) {
    const d = G[set][B][a].map((x, k) => x - G[set][B][b][k]);
    const w = d.filter(x => x < 0).length, l = d.filter(x => x > 0).length;
    rows.push({ fam, set, B, a, b, d: mean(d), ci: boot(d), w, t: d.length - w - l, l, p: wilcoxon(d) });
  }
  holm(rows.map(r => r.p)).forEach((ph, k) => rows[k].ph = ph);
  for (const r of rows) cRows.push([r.fam, r.set, r.B, r.a, r.b, f3(r.d), f3(r.ci[0]), f3(r.ci[1]), r.w, r.t, r.l, fp(r.p), fp(r.ph)].join(","));
}
fs.writeFileSync(path.join(OUT, "v7_contrasts.csv"), cRows.join("\n") + "\n");

// mechanism counters (V7 arms only; weights w = 0.0, 0.1, ..., 1.0)
const mRows = ["set,arm,ils_start_mean,ils_iters_mean,ils_records_mean,cpu_mean_s," +
  [...Array(11).keys()].map(j => `uses_w${j}`).join(",") + "," + [...Array(11).keys()].map(j => `impr_w${j}`).join(",")];
for (const set of SETS) for (const arm of V7) {
  const rr = runs.filter(r => r.arm === arm && r.instance.startsWith(set + "/"));
  const uses = Array(11).fill(0), impr = Array(11).fill(0);
  for (const r of rr) { r.uses.split(" ").forEach((x, j) => uses[j] += +x); r.improvements.split(" ").forEach((x, j) => impr[j] += +x); }
  mRows.push([set, arm, mean(rr.map(r => +r.ils_start)).toFixed(0), mean(rr.map(r => +r.ils_iters)).toFixed(0),
    mean(rr.map(r => +r.ils_records)).toFixed(2), mean(rr.map(r => +r.cpu_s)).toFixed(3), ...uses, ...impr].join(","));
}
fs.writeFileSync(path.join(OUT, "v7_mechanism.csv"), mRows.join("\n") + "\n");

console.log(`${insts.length} instances, ${runs.length} V7 runs (all feasible). Written to ${OUT}`);
