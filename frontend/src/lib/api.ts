export const API_URL: string = (import.meta.env.VITE_API_URL as string | undefined) ?? "/api";

export interface ChatResponse {
  query: string;
  response: string;
}

export interface IndexedDocument {
  id: string | null;
  title: string | null;
  year: number | null;
  venue: string | null;
  authors: string[];
}

export type IngestionState = "pending" | "processing" | "completed" | "failed";
export type IngestionStatusMap = Record<string, IngestionState>;

export interface PaperMetadata {
  paper_title?: string;
  authors?: string[];
  doi?: string;
  venue?: string;
  year?: number;
  abstract?: string;
  repository_link?: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
  ) {
    super(message);
  }
}

async function parseError(res: Response): Promise<ApiError> {
  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
  } catch {
    /* body is not JSON */
  }
  return new ApiError(`Error ${res.status}: ${detail}`, res.status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError("Could not reach the API. Is the FastAPI server running?");
  }
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<T>;
}

export function sendChat(query: string, signal?: AbortSignal): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });
}

export function getDocuments(): Promise<IndexedDocument[]> {
  return request<IndexedDocument[]>("/documents");
}

export function getIngestStatus(): Promise<IngestionStatusMap> {
  return request<IngestionStatusMap>("/ingest/status");
}

/** Uploads a paper with XHR so the caller can track upload progress. */
export function uploadPaper(
  file: File,
  metadata: PaperMetadata,
  onProgress?: (fraction: number) => void,
): Promise<{ status: string; filename: string[] }> {
  const form = new FormData();
  form.append("file", file, file.name);
  for (const [key, value] of Object.entries(metadata)) {
    if (value === undefined || value === "" || value === null) continue;
    if (Array.isArray(value)) value.forEach((v) => form.append(key, v));
    else form.append(key, String(value));
  }

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/ingest/paper`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: any = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else {
        const detail = typeof body?.detail === "string" ? body.detail : xhr.statusText;
        reject(new ApiError(`Error ${xhr.status}: ${detail}`, xhr.status));
      }
    };
    xhr.onerror = () => reject(new ApiError("Could not reach the API."));
    xhr.send(form);
  });
}
