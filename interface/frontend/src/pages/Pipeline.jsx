import { useState } from "react";
import { api, pollJob } from "../api/client";
import { Panel, FormRow, Select, Btn, JobLog, InfoBox } from "../components/ui";

const INPUT_MODES = [
  {
    value: "cropped_ocr",
    label: "Crops + OCR → VLM",
    hint: "Full pipeline: detection finds regions → OCR reads text → VLM extracts features",
    stages: ["detection","ocr","vlm"],
  },
  {
    value: "whole_image_ocr",
    label: "Image + OCR → VLM",
    hint: "OCR runs on the full image → VLM uses image + text. No detection step.",
    stages: ["ocr","vlm"],
  },
  {
    value: "whole_image",
    label: "Whole image → VLM only",
    hint: "Image goes directly to the VLM. No detection, no OCR.",
    stages: ["vlm"],
  },
];

const MODEL_OPTS = {
  detection: ["yolov11","rtdetr"],
  ocr:       ["easyocr","tesseract"],
  vlm:       ["claude","gpt4o"],
};

export default function PipelinePage() {
  const [task, setTask]         = useState("tables");
  const [inputMode, setInputMode] = useState("cropped_ocr");
  const [models, setModels]     = useState({ detection:"yolov11", ocr:"easyocr", vlm:"claude" });
  const [logs, setLogs]         = useState([]);
  const [running, setRunning]   = useState(false);

  const selected = INPUT_MODES.find(m => m.value === inputMode);

  async function run() {
    setRunning(true); setLogs([]);
    try {
      const { job_id } = await api.runPipeline({
        stages:          selected.stages,
        task,
        detection_model: models.detection,
        ocr_model:       models.ocr,
        vlm_model:       models.vlm,
        input_mode:      inputMode,
      });
      await pollJob(job_id, setLogs);
    } catch (e) {
      setLogs(l => [...l, { ts: new Date().toISOString(), msg: `Error: ${e.message}` }]);
    } finally { setRunning(false); }
  }

  return (
    <div>
      <InfoBox>
        The full pipeline runs all required stages end-to-end. Choose your input mode first — it determines which stages run automatically.
      </InfoBox>

      {/* Step 1 — Task */}
      <Panel title="1 · Task">
       
          <Select value={task} onChange={setTask}
            options={[{value:"tables",label:"Tables"},{value:"dimensions",label:"Dimensions"},{value:"both",label:"Both"}]} />
        
      </Panel>

      {/* Step 2 — Input mode drives stages */}
      <Panel title="2 · Input mode">
        <div className="space-y-3">
          {INPUT_MODES.map(m => (
            <label key={m.value}
              onClick={() => setInputMode(m.value)}
              className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition
                ${inputMode === m.value ? "border-gray-900 bg-gray-50" : "border-gray-100 hover:border-gray-200"}`}>
              <input type="radio" name="input-mode" value={m.value} checked={inputMode === m.value} onChange={() => setInputMode(m.value)} className="mt-1" />
              <div>
                <p className="text-sm font-medium text-gray-800">{m.label}</p>
                <p className="text-xs text-gray-400 mt-0.5">{m.hint}</p>
                <div className="flex gap-1.5 mt-2">
                  {m.stages.map(s => (
                    <span key={s} className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 font-medium">{s}</span>
                  ))}
                </div>
              </div>
            </label>
          ))}
        </div>
      </Panel>

      {/* Step 3 — Model per stage (only stages that apply) */}
      <Panel title="3 · Models">
        {selected.stages.includes("detection") && (
          <FormRow label="Detection">
            <Select value={models.detection} onChange={v => setModels(m=>({...m,detection:v}))} options={MODEL_OPTS.detection} />
          </FormRow>
        )}
        {selected.stages.includes("ocr") && (
          <FormRow label="OCR">
            <Select value={models.ocr} onChange={v => setModels(m=>({...m,ocr:v}))} options={MODEL_OPTS.ocr} />
          </FormRow>
        )}
        {selected.stages.includes("vlm") && (
          <FormRow label="VLM">
            <Select value={models.vlm} onChange={v => setModels(m=>({...m,vlm:v}))} options={MODEL_OPTS.vlm} />
          </FormRow>
        )}
      </Panel>

      {/* Run */}
      <Btn primary onClick={run} loading={running} disabled={running}>
        ▶ Run {selected.stages.join(" → ")}
      </Btn>

      {(running || logs.length > 0) && (
        <div className="mt-4"><Panel title="Live logs"><JobLog logs={logs} /></Panel></div>
      )}
    </div>
  );
}