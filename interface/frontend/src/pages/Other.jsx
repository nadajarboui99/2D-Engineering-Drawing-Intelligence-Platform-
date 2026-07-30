import { useState, useEffect } from "react";
import { api, pollJob } from "../api/client";
import { Panel, FormRow, Select, Input, Btn, JobLog, InfoBox, RadioGroup, Badge } from "../components/ui";

// full pipeline
export function PipelinePage() {
  const [cfg, setCfg] = useState({
    stages: ["detection","ocr","vlm"],
    task: "tables",
    detection_model: "yolov11", detection_size: "n",
    ocr_model: "easyocr",
    vlm_model: "claude",
    input_mode: "cropped_ocr",
    epochs: 100, batch_size: 16, imgsz: 640, device: "mps",
  });
  const [logs, setLogs]       = useState([]);
  const [running, setRunning] = useState(false);

  function toggleStage(s) {
    setCfg(c => ({ ...c, stages: c.stages.includes(s) ? c.stages.filter(x => x !== s) : [...c.stages, s] }));
  }

  async function run() {
    setRunning(true); setLogs([]);
    try {
      const { job_id } = await api.runPipeline(cfg);
      await pollJob(job_id, setLogs);
    } catch (e) {
      setLogs(l => [...l, { ts: new Date().toISOString(), msg: `Error: ${e.message}` }]);
    } finally { setRunning(false); }
  }

  return (
    <div>
      <InfoBox>
        The full pipeline runs all selected stages in sequence on a dataset from scratch.
        Use this when you want to evaluate the complete system end-to-end on new data,
        rather than running each stage individually.
      </InfoBox>

      <Panel title="Dataset & task">
        <FormRow label="Task">
          <Select value={cfg.task} onChange={v => setCfg(c=>({...c,task:v}))}
            options={[{value:"tables",label:"Tables"},{value:"dimensions",label:"Dimensions"},{value:"both",label:"Both"}]} />
        </FormRow>
      </Panel>

      <Panel title="Stages to run">
        {[
          { id:"detection", label:"Detection",        desc:"Find tables and dimensions in images" },
          { id:"ocr",       label:"OCR",              desc:"Extract text from detected regions" },
          { id:"vlm",       label:"VLM extraction",   desc:"Extract structured features" },
        ].map(s => (
          <div key={s.id} className="flex items-start gap-3 py-3 border-b border-gray-50 last:border-0">
            <input type="checkbox" checked={cfg.stages.includes(s.id)} onChange={() => toggleStage(s.id)} className="mt-1" />
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-700">{s.label}</p>
              <p className="text-xs text-gray-400">{s.desc}</p>
            </div>
            {s.id === "detection" && cfg.stages.includes(s.id) && (
              <Select value={cfg.detection_model} onChange={v => setCfg(c=>({...c,detection_model:v}))} options={["yolov11","rtdetr"]} />
            )}
            {s.id === "ocr" && cfg.stages.includes(s.id) && (
              <Select value={cfg.ocr_model} onChange={v => setCfg(c=>({...c,ocr_model:v}))} options={["easyocr","tesseract"]} />
            )}
            {s.id === "vlm" && cfg.stages.includes(s.id) && (
              <Select value={cfg.vlm_model} onChange={v => setCfg(c=>({...c,vlm_model:v}))} options={["claude","gpt4o"]} />
            )}
          </div>
        ))}
      </Panel>

      {cfg.stages.includes("vlm") && (
        <Panel title="VLM input mode">
          <RadioGroup name="pipe-mode" value={cfg.input_mode} onChange={v => setCfg(c=>({...c,input_mode:v}))} options={[
            { value:"cropped_ocr",     label:"Crops + OCR",  hint:"Detection crops → OCR → VLM (most complete)" },
            { value:"whole_image_ocr", label:"Image + OCR",  hint:"Full image + OCR text → VLM" },
            { value:"whole_image",     label:"Whole image",  hint:"Full image only → VLM" },
          ]} />
        </Panel>
      )}

      <div className="flex gap-2">
        <Btn primary onClick={run} loading={running} disabled={running || cfg.stages.length === 0}>
          ▶ Run pipeline
        </Btn>
      </div>

      {(running || logs.length > 0) && (
        <div className="mt-4"><Panel title="Live logs"><JobLog logs={logs} /></Panel></div>
      )}
    </div>
  );
}

// features
export function FeaturesPage() {
  const [features, setFeatures] = useState([]);
  const [form, setForm]         = useState({ name:"", type:"number", unit:"mm", description:"" });

  useEffect(() => { api.getFeatures().then(setFeatures).catch(() => {}); }, []);

  async function add() {
    if (!form.name || !form.description) return;
    await api.addFeature(form);
    setFeatures(await api.getFeatures());
    setForm({ name:"", type:"number", unit:"mm", description:"" });
  }

  async function del(name) {
    await api.deleteFeature(name);
    setFeatures(await api.getFeatures());
  }

  return (
    <div>
      <InfoBox>
        Features defined here are automatically included in every VLM prompt. Adding or removing a feature takes effect on the next VLM run — no code changes needed.
      </InfoBox>
      <Panel title="Feature schema"
        action={<span className="text-xs text-gray-400">{features.length} features</span>}>
        {features.map(f => (
          <div key={f.name} className="flex items-center gap-3 py-2.5 border-b border-gray-50 last:border-0">
            <span className="font-medium text-sm text-gray-800 w-32 shrink-0">{f.name}</span>
            <span className="text-xs text-gray-400 w-24 shrink-0">{f.type}{f.unit ? ` · ${f.unit}` : ""}</span>
            <span className="text-xs text-gray-500 flex-1">{f.description}</span>
            <button onClick={() => del(f.name)} className="text-xs text-gray-300 hover:text-red-400 transition shrink-0">remove</button>
          </div>
        ))}
      </Panel>
      <Panel title="Add feature">
        <FormRow label="Name"><Input value={form.name} onChange={v => setForm(f=>({...f,name:v}))} placeholder="thickness" /></FormRow>
        <FormRow label="Type"><Select value={form.type} onChange={v => setForm(f=>({...f,type:v}))} options={["number","integer","string"]} /></FormRow>
        <FormRow label="Unit"><Input value={form.unit} onChange={v => setForm(f=>({...f,unit:v}))} placeholder="mm (leave empty if not applicable)" /></FormRow>
        <FormRow label="Description"><Input value={form.description} onChange={v => setForm(f=>({...f,description:v}))} placeholder="Thickness of the sheet metal part" /></FormRow>
        <Btn onClick={add} disabled={!form.name || !form.description}>+ Add feature</Btn>
      </Panel>
    </div>
  );
}

// add model
export function ModelsPage() {
  const [tab, setTab]           = useState("detection");
  const [form, setForm]         = useState({ name:"", install_cmd:"", module_class:"", module_path:"" });
  const [installing, setInstalling] = useState(false);
  const [logs, setLogs]         = useState([]);

  async function install() {
    if (!form.install_cmd || !form.name) return;
    setInstalling(true); setLogs([]);
    try {
      const { job_id } = await api.installModel({ ...form, task: tab });
      await pollJob(job_id, setLogs);
    } catch (e) {
      setLogs(l => [...l, { ts: new Date().toISOString(), msg: `Error: ${e.message}` }]);
    } finally { setInstalling(false); }
  }

  return (
    <div>
      <InfoBox>
        Add a new model to any module. Provide the pip install command and the model will be installed in the current environment and registered automatically.
        After installation, the model will appear in the selector on the relevant page.
      </InfoBox>

      <div className="flex gap-2 mb-4">
        {["detection","ocr","vlm"].map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-1.5 text-sm rounded-lg border transition
              ${tab === t ? "bg-gray-900 text-white border-transparent" : "border-gray-200 text-gray-600 hover:bg-gray-50"}`}>
            {t.toUpperCase()}
          </button>
        ))}
      </div>

      <Panel title={`Add ${tab} model`}>
        <FormRow label="Model name" hint="How it will appear in the selector">
          <Input value={form.name} onChange={v => setForm(f=>({...f,name:v}))} placeholder="e.g. yolov12, trocr, gpt4o" />
        </FormRow>
        <FormRow label="Install command" hint="Run in the project virtualenv">
          <Input value={form.install_cmd} onChange={v => setForm(f=>({...f,install_cmd:v}))} placeholder="pip install ultralytics==9.0" />
        </FormRow>
        <FormRow label="Module path" hint="Path to the wrapper file you create in models/">
          <Input value={form.module_path} onChange={v => setForm(f=>({...f,module_path:v}))} placeholder="models/yolov12.py" />
        </FormRow>
        <FormRow label="Class name" hint="Class inside the wrapper that implements the interface">
          <Input value={form.module_class} onChange={v => setForm(f=>({...f,module_class:v}))} placeholder="YOLOv12Detector" />
        </FormRow>
        <Btn primary onClick={install} loading={installing} disabled={installing || !form.name || !form.install_cmd}>
          ⬇ Install & register
        </Btn>
      </Panel>

      {(installing || logs.length > 0) && (
        <Panel title="Installation logs"><JobLog logs={logs} /></Panel>
      )}

      <Panel title="How it works">
        <ol className="text-xs text-gray-500 space-y-2 list-decimal list-inside">
          <li>The backend runs your install command in a subprocess inside the project virtualenv.</li>
          <li>It registers the model name → module path → class name in a local registry file.</li>
          <li>The model appears in the selector on the relevant page immediately.</li>
          <li>You still need to create the wrapper file (e.g. <code className="bg-gray-100 px-1 rounded">models/yolov12.py</code>) that inherits from the base interface — this gives you full control over how the model is called.</li>
        </ol>
      </Panel>
    </div>
  );
}