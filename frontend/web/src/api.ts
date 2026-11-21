export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface HighlightRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FieldEvidence {
  field: string;
  value: unknown;
  page: number | null;
  quote: string | null;
  rects?: HighlightRect[];
}

export interface ExtractionMeta {
  doc_type: string;
  confidence: number;
  valid: boolean;
  attempts: number;
  mode: string;
  model: string;
  elapsed_seconds: number;
  text_length: number;
  page_count?: number;
  pages?: string[];
  page_images?: string[];
  usage: TokenUsage;
  evidence?: FieldEvidence[];
  validation_errors: string[];
  validation_warnings: string[];
}

export interface ExtractionResponse {
  filename: string;
  data: Record<string, unknown>;
  meta: ExtractionMeta;
}

const DIRECT_BACKEND = "http://127.0.0.1:8000";

async function fetchWithFallback(path: string, init?: RequestInit): Promise<Response> {
  try {
    const res = await fetch(path, init);
    if (res.status !== 404) return res;
  } catch {
    /* Vite proxy unreachable -> fall back to direct FastAPI URL */
  }
  return fetch(`${DIRECT_BACKEND}${path}`, init);
}

export async function extractDocument(file: File): Promise<ExtractionResponse> {
  const buildBody = () => {
    const fd = new FormData();
    fd.append("file", file);
    return fd;
  };

  let res: Response;
  try {
    res = await fetch("/api/extract/document", { method: "POST", body: buildBody() });
    if (res.status === 404) {
      res = await fetch(`${DIRECT_BACKEND}/api/extract/document`, { method: "POST", body: buildBody() });
    }
  } catch {
    res = await fetch(`${DIRECT_BACKEND}/api/extract/document`, { method: "POST", body: buildBody() });
  }

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetchWithFallback("/health");
    if (!res.ok) return false;
    const j = await res.json();
    return j?.status === "ok";
  } catch {
    return false;
  }
}
