const BASE = "http://localhost:8000";
export const API_BASE = BASE;

async function req(method, path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export const api = {
  // Detection
  getWeightsList:          (task)            => req("GET",  `/detection/weights/${task}`),
  getBestWeights:          (task)            => req("GET",  `/detection/best-weights/${task}`),
  runDetection:            (task, cfg)       => req("POST", `/detection/run/${task}`, cfg),
  evalDetection:           (task)            => req("POST", `/detection/eval/${task}`),
  evalDetectionWithWeights:(task, path, sz)  => req("POST", `/detection/eval/${task}`, { weights_path: path, imgsz: sz }),
  evalDetectionAnnotated:  (task, path, sz)  => req("POST", `/detection/eval-annotated/${task}`, { weights_path: path, imgsz: sz }),
  evalDetectionDetail:     (task, path, sz)  => req("POST", `/detection/detail/${task}`, { weights_path: path, imgsz: sz }),
  getDetectionAnnotated:   (task)            => req("GET",  `/detection/annotated-result/${task}`),
  getDetectionResults:     (task)            => req("GET",  `/detection/results/${task}`),

  // Weights management
  deleteWeights:    (id)   => req("DELETE", `/weights/${id}`),

  // OCR
  runOCR:        (cfg)            => req("POST", `/ocr/run`, cfg),
  getOCRResults: (task, m, mode)  => req("GET",  `/ocr/results/${task}/${m}?mode=${mode || "crop"}`),
  getOCRMetrics: (task, m, mode)  => req("GET",  `/ocr/metrics/${task}/${m}?mode=${mode || "crop"}`),
  getOCRDetail:  (m)              => req("GET",  `/ocr/detail?model=${m || "easyocr"}`),
  getOCRCrops:   (img, m, source) => req("GET",  `/ocr/crops-detail/${encodeURIComponent(img)}?model=${m || "easyocr"}&source=${source || "detector"}`),

  // VLM
  runVLM:        (cfg)       => req("POST", `/vlm/run`, cfg),
  getVLMResults: (task, mode)=> req("GET",  `/vlm/results${task?`?task=${task}`:""}${mode?`${task?"&":"?"}mode=${mode}`:""}`),
  getVLMMetrics: (task, mode)=> req("GET",  `/vlm/metrics?task=${task || "tables"}${mode?`&mode=${mode}`:""}`),
  getVLMCompare: (task)      => req("GET",  `/vlm/compare?task=${task || "tables"}`),
  getVLMDetail:  ()          => req("GET",  `/vlm/detail`),

  // Pipeline
  runPipeline:   (cfg)       => req("POST", `/pipeline/run`, cfg),

  // Annotation
  listAnnotations:  ()      => req("GET",    `/annotation/list`),
  getMaster:        (stem)  => req("GET",    `/annotation/master/${encodeURIComponent(stem)}`),
  deleteAnnotation: (stem)  => req("DELETE", `/annotation/${encodeURIComponent(stem)}`),

  // Features
  getFeatures:   ()          => req("GET",  `/features/`),
  addFeature:    (d)         => req("POST", `/features/`, d),
  deleteFeature: (n)         => req("DELETE", `/features/${n}`),

  // Prompts
  getPrompt:     (mode)      => req("GET",  `/prompts/${mode}`),
  updatePrompt:  (d)         => req("POST", `/prompts/`, d),
  resetPrompt:   (mode)      => req("POST", `/prompts/reset/${mode}`),

  // API keys
  getProviders:  ()          => req("GET",  `/keys/providers`),
  setKey:        (d)         => req("POST", `/keys/set`, d),
  getAllVLMModels:()          => req("GET",  `/keys/models`),

  // Models registry
  getAvailableModels: (task) => req("GET",  `/models/available/${task}`),
  installModel:       (d)    => req("POST", `/models/install`, d),

  // Results
  getAllResults:  (stage, task) => req("GET", `/results/all${stage?`?stage=${stage}`:""}${task?`${stage?"&":"?"}task=${task}`:""}`),
  getBestResult:  (stage, task) => req("GET", `/results/best/${stage}/${task}`),
  getSummary:     ()            => req("GET", `/results/summary`),
  deleteRun:      (id)          => req("DELETE", `/results/${id}`),

  // Jobs
  getJobs:  ()   => req("GET", `/jobs/`),
  getJob:   (id) => req("GET", `/jobs/${id}`),
};

export async function saveAnnotation(payload, file) {
  const form = new FormData();
  form.append("payload", JSON.stringify(payload));
  if (file) form.append("image", file);
  const res = await fetch(`${BASE}/annotation/save`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function pollJob(jobId, onLog, intervalMs = 1500) {
  return new Promise((resolve, reject) => {
    const timer = setInterval(async () => {
      try {
        const job = await api.getJob(jobId);
        onLog(job.logs);
        if (job.status === "done") { clearInterval(timer); resolve(job.result); }
        else if (job.status === "failed") { clearInterval(timer); reject(new Error(job.error)); }
      } catch (e) { clearInterval(timer); reject(e); }
    }, intervalMs);
  });
}