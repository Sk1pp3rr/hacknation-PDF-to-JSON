export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface FieldEvidence {
  field: string;
  value: unknown;
  page: number | null;
  quote: string | null;
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

export async function extractDocument(file: File): Promise<ExtractionResponse> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/extract/document", { method: "POST", body });
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
    const res = await fetch("/health");
    return res.ok;
  } catch {
    return false;
  }
}
