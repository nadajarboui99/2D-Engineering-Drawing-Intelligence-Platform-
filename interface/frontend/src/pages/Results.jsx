import { useState, useEffect } from "react";
import { api } from "../api/client";
import { Panel, Badge, Tabs, InfoBox, Btn } from "../components/ui";
import { RunImageCell, runImages, ImgThumb } from "../components/runViews";

const STAGE_METRIC = { detection: "map50", ocr: "coverage", vlm: "fill_rate" };
const STAGE_METRIC_LABEL = { detection: "mAP@0.5", ocr: "Coverage", vlm: "Fill rate" };

// Input-format / approach axis (separate from task). Reads run.extra.mode.
const APPROACH_OPTS = {
  ocr: [["", "All formats"], ["full", "Whole image"], ["crop", "Cropped · detector"], ["gtcrop", "Cropped · ground-truth"]],
  vlm: [["", "All modes"], ["whole_image", "Image only"], ["whole_image_ocr", "Image + page OCR"], ["cropped_ocr", "Image + crop OCR"]],
};
// Task = which classes. tables | dimensions | both (both = the whole page).
// Whole-image runs are stored as "all" but are conceptually "both", so they fold in.
const TASK_OPTS = [["", "All tasks"], ["tables", "Tables"], ["dimensions", "Dimensions"], ["both", "Tables + Dimensions"]];
const taskLabel = t => (t === "all" || t === "both") ? "Tables + Dimensions"
  : (TASK_OPTS.find(([v]) => v === t) || [t, t])[1];
const approachOf    = r => r?.extra?.mode || "";
const approachLabel = (stage, mode) => (APPROACH_OPTS[stage] || []).find(([v]) => v === mode)?.[1] || (mode || "—");

const fmt = v => v == null ? "—" : (typeof v !== "number" ? String(v) : (Number.isInteger(v) ? String(v) : v.toFixed(3)));
const runTime = ts => new Date(ts).toLocaleString(undefined, { month:"short", day:"numeric", hour:"2-digit", minute:"2-digit" });

export default function ResultsPage() {
  const [stage, setStage]       = useState("detection");
  const [taskF, setTaskF]       = useState("");
  const [approachF, setApproachF] = useState("");
  const [allRuns, setAllRuns]   = useState([]);
  const [summary, setSummary]   = useState(null);

  const [selected, setSelected] = useState(() => new Set());
  const [view, setView]         = useState(null);   // {type:"single"|"compare", ids:[]}
  const [snaps, setSnaps]       = useState({});      // id -> snapshot ({} if none)
  const [conf, setConf]         = useState(0.25);
  const [sortKey, setSortKey]   = useState("date");  // "date" | a metric name
  const [sortDir, setSortDir]   = useState("desc");  // desc = latest / highest first

  useEffect(() => { load(); setTaskF(""); setApproachF(""); setSelected(new Set()); setView(null); }, [stage]);
  // Changing filters changes what's comparable → drop any stale ticks.
  useEffect(() => { setSelected(new Set()); }, [taskF, approachF]);

  async function load() {
    try {
      let runs = await api.getAllResults(stage);
      // Detection has legacy TRAINING-metric runs logged too; the Results page
      // is about EVALUATION on your annotated set, so keep only annotated runs.
      if (stage === "detection") runs = runs.filter(r => r.extra?.on === "annotated");
      setAllRuns(runs);
      setSummary(await api.getSummary());
    } catch {}
  }
  async function remove(id) {
    await api.deleteRun(id);
    setSelected(s => { const n = new Set(s); n.delete(id); return n; });
    setView(null); load();
  }
  function toggle(id) { setSelected(s => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; }); }

  async function fetchSnaps(ids) {
    const missing = ids.filter(id => !(id in snaps));
    if (!missing.length) return;
    const got = await Promise.all(missing.map(id => api.getRunSnapshot(id).catch(() => ({}))));
    setSnaps(prev => { const m = { ...prev }; missing.forEach((id, i) => m[id] = got[i] || {}); return m; });
  }
  async function openSingle(id) { await fetchSnaps([id]); setView({ type: "single", ids: [id] }); }
  async function runCompare()   { const ids = [...selected]; await fetchSnaps(ids); setView({ type: "compare", ids }); }

  const runById = id => allRuns.find(r => r.id === id);
  const taskMatch = r => !taskF || r.task === taskF || (taskF === "both" && r.task === "all");
  const filtered = allRuns.filter(r => taskMatch(r) && (!approachF || approachOf(r) === approachF));
  const metricKeys = [...new Set(filtered.flatMap(r => Object.keys(r.metrics || {})))];
  const sortVal = r => sortKey === "date"
    ? new Date(r.timestamp).getTime()
    : (typeof r.metrics?.[sortKey] === "number" ? r.metrics[sortKey] : (sortDir === "desc" ? -Infinity : Infinity));
  const sorted = [...filtered].sort((a,b) => sortDir === "desc" ? sortVal(b) - sortVal(a) : sortVal(a) - sortVal(b));
  const hasApproach = !!APPROACH_OPTS[stage];
  // Runs are comparable only once the filters pin them to one kind:
  // detection needs a specific task; OCR/VLM need a specific input-format AND task.
  const comparable = hasApproach ? (!!taskF && !!approachF) : !!taskF;

  return (
    <div>
      <InfoBox>
        Every run is logged here. Use the filters to narrow to a <strong>comparable</strong> set, tick runs and <strong>Compare</strong>, or <strong>View</strong> one on its own.
        Only runs made after this feature was added carry the visual snapshot; older ones still compare on metrics.
      </InfoBox>

      {summary && (
        <Panel title="Best model per stage">
          <div className="grid grid-cols-3 gap-3">
            {["detection","ocr","vlm"].map(s => {
              const best = summary[s]?.best;
              return (
                <div key={s} className="border border-gray-100 rounded-xl p-4">
                  <p className="text-xs text-gray-400 uppercase tracking-wide mb-2">{s} · {STAGE_METRIC_LABEL[s]}</p>
                  {best ? (
                    <div className="flex items-center justify-between">
                      <div><p className="text-sm font-medium text-gray-700 truncate">{best.model}</p><Badge variant="gray">{best.task}</Badge></div>
                      <p className="text-sm text-emerald-600 font-mono">{fmt(best.metrics?.[STAGE_METRIC[s]])}</p>
                    </div>
                  ) : <span className="text-xs text-gray-300">no runs yet</span>}
                  <p className="text-xs text-gray-300 mt-2">{summary[s]?.count || 0} runs · tasks: {(summary[s]?.tasks||[]).join(", ") || "—"}</p>
                </div>
              );
            })}
          </div>
        </Panel>
      )}

      <Tabs tabs={[{id:"detection",label:"Detection"},{id:"ocr",label:"OCR"},{id:"vlm",label:"VLM"}]} active={stage} onChange={setStage} />

      <div className="flex items-center gap-3 mb-4 flex-wrap">
        {hasApproach && (
          <>
            <span className="text-xs text-gray-400">Input format</span>
            <select value={approachF} onChange={e => setApproachF(e.target.value)}
              className="text-sm border border-gray-200 rounded-lg px-3 py-1.5 bg-white text-gray-700 focus:outline-none focus:ring-1 focus:ring-gray-300">
              {APPROACH_OPTS[stage].map(([v,l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </>
        )}
        <span className="text-xs text-gray-400">Task</span>
        <select value={taskF} onChange={e => setTaskF(e.target.value)}
          className="text-sm border border-gray-200 rounded-lg px-3 py-1.5 bg-white text-gray-700 focus:outline-none focus:ring-1 focus:ring-gray-300">
          {TASK_OPTS.map(([v,l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        <span className="text-xs text-gray-400 ml-2">Sort by</span>
        <select value={sortKey} onChange={e => setSortKey(e.target.value)}
          className="text-sm border border-gray-200 rounded-lg px-3 py-1.5 bg-white text-gray-700 focus:outline-none focus:ring-1 focus:ring-gray-300">
          <option value="date">Date</option>
          {metricKeys.map(k => <option key={k} value={k}>{k}</option>)}
        </select>
        <button onClick={() => setSortDir(d => d === "desc" ? "asc" : "desc")}
          title={sortDir === "desc" ? "Descending (highest / latest first)" : "Ascending (lowest / oldest first)"}
          className="text-sm border border-gray-200 rounded-lg px-2.5 py-1.5 bg-white text-gray-600 hover:bg-gray-50">
          {sortDir === "desc" ? "▼" : "▲"}
        </button>
        <div className="flex-1" />
        {comparable && selected.size > 0 && (
          <>
            <span className="text-xs text-gray-400">{selected.size} selected</span>
            <Btn small onClick={runCompare} disabled={selected.size < 2}>⇄ Compare</Btn>
            <Btn small onClick={() => setSelected(new Set())}>clear</Btn>
          </>
        )}
      </div>

      <Panel title={`${stage} runs`} badge={sorted.length ? `${sorted.length} runs` : undefined}>
        <p className="text-xs text-gray-400 mb-2">
          {comparable
            ? "These runs are comparable — tick the ones you want and hit Compare."
            : `Set ${hasApproach ? "an input format and a task" : "a task"} above to make runs comparable — then checkboxes appear to pick which to compare. You can still View any run on its own.`}
        </p>
        {sorted.length === 0 && <p className="text-sm text-gray-300 text-center py-8">No runs match these filters.</p>}
        {sorted.map((r, i) => (
          <div key={r.id} className="flex items-center gap-3 py-3 border-b border-gray-50 last:border-0">
            {comparable && <input type="checkbox" checked={selected.has(r.id)} onChange={() => toggle(r.id)} className="accent-gray-900" />}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-700 truncate">{r.model}</p>
              <div className="flex gap-1.5 mt-0.5">
                <Badge variant="gray">{taskLabel(r.task)}</Badge>
                {hasApproach && <Badge variant="blue">{approachLabel(stage, approachOf(r))}</Badge>}
              </div>
            </div>
            <div className="flex gap-3 flex-wrap justify-end max-w-[40%]">
              {Object.entries(r.metrics).slice(0, 5).map(([k,v]) => (
                <div key={k} className="text-right"><p className="text-[10px] text-gray-400">{k}</p><p className="text-sm font-mono text-gray-700">{fmt(v)}</p></div>
              ))}
            </div>
            <span className="text-xs text-gray-300 w-20 text-right shrink-0">{runTime(r.timestamp)}</span>
            <button onClick={() => openSingle(r.id)} className="text-xs font-medium text-gray-600 hover:text-gray-900 shrink-0">View</button>
            <button onClick={() => remove(r.id)} className="text-xs text-gray-300 hover:text-red-400 shrink-0">remove</button>
          </div>
        ))}
      </Panel>

      {/* View / Compare open as a modal overlay so it's always visible. */}
      {view && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-start justify-center p-6 overflow-y-auto" onClick={() => setView(null)}>
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-6xl my-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 sticky top-0 bg-white rounded-t-2xl">
              <p className="text-sm font-medium text-gray-800">{view.type === "single" ? "Run details" : "Comparison"}</p>
              <div className="flex items-center gap-4">
                {stage === "detection" && (
                  <span className="flex items-center gap-2 text-xs text-gray-500">
                    conf ≥ {conf.toFixed(2)}
                    <input type="range" min="0.05" max="0.9" step="0.05" value={conf} onChange={e => setConf(parseFloat(e.target.value))} className="w-28 accent-gray-900" />
                  </span>
                )}
                <button onClick={() => setView(null)} className="text-gray-400 hover:text-gray-800 text-xl leading-none">✕</button>
              </div>
            </div>
            <div className="p-6">
              {view.type === "single" && (() => {
                const r = runById(view.ids[0]); const snap = snaps[view.ids[0]];
                if (!r) return <InfoBox>Run not found.</InfoBox>;
                const imgs = snap?.view ? runImages(snap) : [];
                return (
                  <>
                    <p className="font-medium text-gray-800">{r.model}</p>
                    <p className="text-xs text-gray-400 mb-4">{stage} · {taskLabel(r.task)}{hasApproach ? " · " + approachLabel(stage, approachOf(r)) : ""} · {runTime(r.timestamp)}</p>
                    <div className="flex gap-4 flex-wrap mb-4">
                      {Object.entries(r.metrics).map(([k,v]) => (
                        <div key={k}><p className="text-xs text-gray-400">{k}</p><p className="text-sm font-mono text-gray-800">{fmt(v)}</p></div>
                      ))}
                    </div>
                    {imgs.length ? (
                      <div className="grid grid-cols-2 gap-4">
                        {imgs.map(img => (
                          <div key={img} className="border border-gray-100 rounded-xl p-3">
                            <p className="font-mono text-xs text-gray-500 mb-2 truncate">{img}</p>
                            {stage !== "detection" && <div className="mb-2"><ImgThumb image={img} /></div>}
                            <RunImageCell snap={snap} image={img} conf={conf} />
                          </div>
                        ))}
                      </div>
                    ) : <InfoBox>Metrics shown above. This run has <strong>no visual snapshot</strong> (it predates run-history) — re-run it to capture the images/overlay.</InfoBox>}
                  </>
                );
              })()}

              {view.type === "compare" && (() => {
                const runs = view.ids.map(runById).filter(Boolean);
                if (runs.length < 2) return <InfoBox>Select at least two runs to compare.</InfoBox>;
                const labels = [];
                runs.forEach(r => Object.keys(r.metrics).forEach(k => { if (!labels.includes(k)) labels.push(k); }));
                const imgs = [];
                view.ids.filter(id => snaps[id]?.view).forEach(id => runImages(snaps[id]).forEach(im => { if (!imgs.includes(im)) imgs.push(im); }));
                return (
                  <>
                    <div className="overflow-x-auto">
                      <table className="text-sm border-collapse">
                        <thead>
                          <tr>
                            <th className="text-left text-xs text-gray-400 font-medium p-2 sticky left-0 bg-white">Metric</th>
                            {runs.map(r => (
                              <th key={r.id} className="text-left text-xs font-medium p-2 min-w-[170px]">
                                <p className="text-gray-700 truncate">{r.model}</p>
                                <p className="text-gray-300 font-normal">{taskLabel(r.task)}{hasApproach ? " · " + approachLabel(stage, approachOf(r)) : ""}</p>
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {labels.map(lbl => (
                            <tr key={lbl} className="border-t border-gray-50">
                              <td className="text-gray-500 p-2 sticky left-0 bg-white">{lbl}</td>
                              {runs.map(r => <td key={r.id} className="p-2 font-mono text-gray-800">{fmt(r.metrics[lbl])}</td>)}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {imgs.length ? (
                      <>
                        <p className="text-xs text-gray-400 mt-4 mb-2">Per-image (only runs with a saved snapshot)</p>
                        <div className="space-y-4">
                          {imgs.map(img => (
                            <div key={img} className="border border-gray-100 rounded-xl p-3">
                              <p className="font-mono text-xs text-gray-500 mb-2 truncate">{img}</p>
                              <div className="grid gap-3" style={{ gridTemplateColumns: `150px repeat(${runs.length}, minmax(0,1fr))` }}>
                                <div><ImgThumb image={img} /></div>
                                {runs.map(r => (
                                  <div key={r.id}>
                                    <p className="text-[11px] text-gray-400 mb-1 truncate">{r.model}</p>
                                    {snaps[r.id]?.view ? <RunImageCell snap={snaps[r.id]} image={img} conf={conf} /> : <p className="text-xs text-gray-300">no snapshot</p>}
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </>
                    ) : <p className="text-xs text-gray-400 mt-4">Metrics compared above. None of these runs has a visual snapshot yet — re-run them to compare images/overlays too.</p>}
                  </>
                );
              })()}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
