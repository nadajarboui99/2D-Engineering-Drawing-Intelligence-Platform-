import { useState, useEffect } from "react";
import { api } from "../api/client";
import { Panel, Badge, Tabs, InfoBox } from "../components/ui";

const STAGE_METRIC_LABEL = {
  detection: "mAP@0.5",
  ocr:       "Coverage",
  vlm:       "Fill rate",
};

function getMetricValue(run, stage) {
  if (!run || !run.metrics) return 0;
  const key = stage === "detection" ? "map50" : stage === "ocr" ? "coverage" : "fill_rate";
  return run.metrics[key] ?? 0;
}

export default function ResultsPage() {
  const [stage, setStage]     = useState("detection");
  const [task, setTask]       = useState("all");
  const [allRuns, setAllRuns] = useState([]);   // all runs for the stage (any task)
  const [summary, setSummary] = useState(null);

  useEffect(() => { load(); }, [stage]);
  useEffect(() => { setTask("all"); }, [stage]);

  async function load() {
    try {
      setAllRuns(await api.getAllResults(stage));
      setSummary(await api.getSummary());
    } catch {}
  }

  async function remove(id) {
    await api.deleteRun(id);
    load();
  }

  // Tasks present for this stage → chips: All + each distinct task ("all"/"both"/tables/dimensions).
  const taskOptions = ["all", ...Array.from(new Set(allRuns.map(r => r.task))).sort()];
  const runs   = task === "all" ? allRuns : allRuns.filter(r => r.task === task);
  const sorted = [...runs].sort((a,b) => getMetricValue(b,stage) - getMetricValue(a,stage));

  return (
    <div>
      <InfoBox>
        Every evaluation run from every stage is logged here automatically. Use this page to compare models within a stage,
        and to identify the best detection, OCR, and VLM model — which you can then combine in the Full Pipeline page.
      </InfoBox>

      {/* Summary — best of each stage (across all tasks) */}
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
                      <div>
                        <p className="text-sm font-medium text-gray-700 truncate">{best.model}</p>
                        <Badge variant="gray">{best.task}</Badge>
                      </div>
                      <p className="text-sm text-emerald-600 font-mono">{getMetricValue(best,s).toFixed(3)}</p>
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

      <div className="flex gap-2 mb-4 flex-wrap">
        {taskOptions.map(t => (
          <button key={t} onClick={() => setTask(t)}
            className={`px-3 py-1 text-xs rounded-full border transition
              ${task === t ? "bg-gray-900 text-white border-transparent" : "border-gray-200 text-gray-500 hover:bg-gray-50"}`}>
            {t === "all" ? "All tasks" : t}
          </button>
        ))}
      </div>

      <Panel title={`${stage} runs · ${task === "all" ? "all tasks" : task}`} badge={sorted.length ? `${sorted.length} runs` : undefined}>
        {sorted.length === 0 && (
          <p className="text-sm text-gray-300 text-center py-8">No runs yet for this stage and task.</p>
        )}
        {sorted.map((r, i) => (
          <div key={r.id} className={`flex items-center gap-4 py-3 border-b border-gray-50 last:border-0 ${i === 0 ? "bg-emerald-50/40 -mx-5 px-5" : ""}`}>
            {i === 0 && <Badge variant="green">best</Badge>}
            <span className="text-sm font-medium text-gray-700 flex-1 truncate">{r.model}</span>
            <div className="flex gap-4">
              {Object.entries(r.metrics).map(([k,v]) => (
                <div key={k} className="text-right">
                  <p className="text-xs text-gray-400">{k}</p>
                  <p className="text-sm font-mono text-gray-700">{typeof v === "number" ? v.toFixed(3) : v}</p>
                </div>
              ))}
            </div>
            <span className="text-xs text-gray-300 w-24 text-right shrink-0">
              {new Date(r.timestamp).toLocaleString(undefined, { month:"short", day:"numeric", hour:"2-digit", minute:"2-digit" })}
            </span>
            <button onClick={() => remove(r.id)} className="text-xs text-gray-300 hover:text-red-400 shrink-0">remove</button>
          </div>
        ))}
      </Panel>
    </div>
  );
}