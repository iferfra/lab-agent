import clsx from "clsx";
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  FileText,
  Loader2,
  UploadCloud,
  X,
  XCircle,
} from "lucide-react";
import { useEffect, useRef, useState, type DragEvent, type FormEvent, type KeyboardEvent } from "react";
import {
  getIngestStatus,
  uploadPaper,
  type IngestionState,
  type IngestionStatusMap,
  type PaperMetadata,
} from "../lib/api";

const MAX_MB = 50;

const STATUS_STYLE: Record<IngestionState, { label: string; icon: typeof Clock; cls: string }> = {
  pending: { label: "Queued", icon: Clock, cls: "bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300" },
  processing: {
    label: "Processing",
    icon: Loader2,
    cls: "bg-brand-50 text-brand-700 dark:bg-brand-500/10 dark:text-brand-300",
  },
  completed: {
    label: "Indexed",
    icon: CheckCircle2,
    cls: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300",
  },
  failed: { label: "Error", icon: XCircle, cls: "bg-red-50 text-red-700 dark:bg-red-500/10 dark:text-red-300" },
};

function formatSize(bytes: number) {
  return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function AuthorsInput({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  const [draft, setDraft] = useState("");
  const commit = () => {
    const names = draft
      .split(/[,;]/)
      .map((s) => s.trim())
      .filter((s) => s && !value.includes(s));
    if (names.length) onChange([...value, ...names]);
    setDraft("");
  };
  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      commit();
    } else if (e.key === "Backspace" && !draft && value.length) {
      onChange(value.slice(0, -1));
    }
  };
  return (
    <div className="input flex flex-wrap items-center gap-1.5 py-2 focus-within:border-brand-400 focus-within:ring-4 focus-within:ring-brand-100 dark:focus-within:ring-brand-900/40">
      {value.map((a) => (
        <span
          key={a}
          className="inline-flex items-center gap-1 rounded-lg bg-slate-100 py-0.5 pl-2 pr-1 text-sm dark:bg-slate-800"
        >
          {a}
          <button
            type="button"
            onClick={() => onChange(value.filter((x) => x !== a))}
            className="rounded p-0.5 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
            aria-label={`Remove ${a}`}
          >
            <X className="h-3 w-3" />
          </button>
        </span>
      ))}
      <input
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={onKeyDown}
        onBlur={commit}
        placeholder={value.length ? "" : "Type a name and press Enter"}
        className="min-w-[160px] flex-1 bg-transparent py-0.5 outline-none placeholder:text-slate-400"
      />
    </div>
  );
}

const EMPTY_FORM = { paper_title: "", doi: "", venue: "", year: "", abstract: "", repository_link: "" };

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [authors, setAuthors] = useState<string[]>([]);
  const [progress, setProgress] = useState<number | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [status, setStatus] = useState<IngestionStatusMap>({});
  const [uploaded, setUploaded] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);

  const hasActive = Object.values(status).some((s) => s === "pending" || s === "processing");

  useEffect(() => {
    let cancelled = false;
    const tick = () =>
      getIngestStatus()
        .then((s) => !cancelled && setStatus(s))
        .catch(() => {});
    tick();
    const id = setInterval(tick, hasActive ? 2500 : 10000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [hasActive]);

  function pick(f: File | undefined | null) {
    setFileError(null);
    setNotice(null);
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".pdf") || (f.type && f.type !== "application/pdf")) {
      setFileError("Only PDF files are accepted.");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setFileError(`The file is larger than ${MAX_MB} MB.`);
      return;
    }
    setFile(f);
    if (!form.paper_title) {
      const guess = f.name.replace(/\.pdf$/i, "").replace(/[_]+/g, " ").trim();
      setForm((p) => ({ ...p, paper_title: guess }));
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    pick(e.dataTransfer.files?.[0]);
  }

  function reset() {
    setFile(null);
    setForm(EMPTY_FORM);
    setAuthors([]);
    setProgress(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setSubmitError(null);
    setNotice(null);
    setProgress(0);

    const metadata: PaperMetadata = {
      paper_title: form.paper_title.trim() || undefined,
      authors: authors.length ? authors : undefined,
      doi: form.doi.trim() || undefined,
      venue: form.venue.trim() || undefined,
      year: form.year ? Number(form.year) : undefined,
      abstract: form.abstract.trim() || undefined,
      repository_link: form.repository_link.trim() || undefined,
    };

    try {
      const name = file.name;
      const before = (await getIngestStatus().catch(() => status))[name];
      await uploadPaper(file, metadata, setProgress);
      const latest = await getIngestStatus().catch(() => status);
      setStatus(latest);
      setUploaded((u) => [name, ...u.filter((n) => n !== name)]);
      // The API skips PDFs whose content hash already exists without queuing them,
      // so an unchanged final status means nothing was queued.
      const after = latest[name];
      const duplicate = after === undefined || (after === before && (after === "completed" || after === "failed"));
      setNotice(
        duplicate
          ? `"${name}" is already in the system (same content), so it was not indexed again.`
          : `"${name}" was added to the ingestion queue.`,
      );
      reset();
    } catch (err) {
      setSubmitError((err as Error).message);
      setProgress(null);
    }
  }

  const set = (k: keyof typeof EMPTY_FORM) => (e: { target: { value: string } }) =>
    setForm((p) => ({ ...p, [k]: e.target.value }));
  const uploading = progress !== null;
  const entries = Object.entries(status).sort(([a], [b]) => {
    const ia = uploaded.indexOf(a);
    const ib = uploaded.indexOf(b);
    return (ia === -1 ? Infinity : ia) - (ib === -1 ? Infinity : ib);
  });

  return (
    <div className="scrollbar-thin h-full overflow-y-auto">
      <div className="mx-auto grid max-w-6xl gap-6 px-5 py-8 lg:grid-cols-[1fr_320px]">
        <form onSubmit={onSubmit} className="space-y-6">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Upload paper</h1>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              Drop a PDF and fill in its metadata. Empty fields are extracted from the document automatically.
            </p>
          </div>

          {/* Dropzone */}
          {!file ? (
            <div
              role="button"
              tabIndex={0}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
              onDragEnter={(e) => {
                e.preventDefault();
                dragDepth.current++;
                setDragging(true);
              }}
              onDragLeave={() => {
                if (--dragDepth.current <= 0) setDragging(false);
              }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={onDrop}
              className={clsx(
                "flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-14 text-center transition",
                dragging
                  ? "scale-[1.01] border-brand-500 bg-brand-50 dark:bg-brand-500/10"
                  : "border-slate-300 bg-white hover:border-brand-400 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:hover:border-brand-600 dark:hover:bg-slate-900/60",
              )}
            >
              <div
                className={clsx(
                  "grid h-14 w-14 place-items-center rounded-2xl transition",
                  dragging ? "bg-brand-600 text-white" : "bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-300",
                )}
              >
                <UploadCloud className="h-7 w-7" />
              </div>
              <p className="mt-4 font-medium">{dragging ? "Drop the PDF here" : "Drag your paper here"}</p>
              <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                or <span className="font-medium text-brand-600 dark:text-brand-400">click to select it</span> · PDF
                up to {MAX_MB} MB
              </p>
            </div>
          ) : (
            <div className="card flex items-center gap-4 p-4">
              <div className="grid h-12 w-12 shrink-0 place-items-center rounded-xl bg-red-50 text-red-600 dark:bg-red-500/10 dark:text-red-400">
                <FileText className="h-6 w-6" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium">{file.name}</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">{formatSize(file.size)}</p>
                {uploading && (
                  <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                    <div className="h-full bg-brand-600 transition-all" style={{ width: `${(progress ?? 0) * 100}%` }} />
                  </div>
                )}
              </div>
              {!uploading && (
                <button type="button" onClick={() => setFile(null)} className="btn-ghost p-2" aria-label="Remove file">
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            hidden
            onChange={(e) => pick(e.target.files?.[0])}
          />
          {fileError && (
            <p className="flex items-center gap-1.5 text-sm text-red-600 dark:text-red-400">
              <AlertCircle className="h-4 w-4" /> {fileError}
            </p>
          )}

          {/* Metadata */}
          <fieldset disabled={uploading} className="card space-y-5 p-6">
            <div className="flex items-baseline justify-between">
              <h2 className="font-semibold">Metadata</h2>
              <span className="text-xs text-slate-400">All optional</span>
            </div>

            <div>
              <label className="label" htmlFor="title">Title</label>
              <input id="title" className="input" value={form.paper_title} onChange={set("paper_title")} placeholder="Paper title" />
            </div>

            <div>
              <label className="label">Authors</label>
              <AuthorsInput value={authors} onChange={setAuthors} />
            </div>

            <div className="grid gap-5 sm:grid-cols-[1fr_1fr_120px]">
              <div>
                <label className="label" htmlFor="venue">Venue</label>
                <input id="venue" className="input" value={form.venue} onChange={set("venue")} placeholder="CVPR, NeurIPS…" />
              </div>
              <div>
                <label className="label" htmlFor="doi">DOI</label>
                <input id="doi" className="input" value={form.doi} onChange={set("doi")} placeholder="10.1109/…" />
              </div>
              <div>
                <label className="label" htmlFor="year">Year</label>
                <input
                  id="year"
                  type="number"
                  min={1900}
                  max={2100}
                  className="input"
                  value={form.year}
                  onChange={set("year")}
                  placeholder="2026"
                />
              </div>
            </div>

            <div>
              <label className="label" htmlFor="repo">Repository</label>
              <input
                id="repo"
                type="url"
                className="input"
                value={form.repository_link}
                onChange={set("repository_link")}
                placeholder="https://github.com/…"
              />
            </div>

            <div>
              <label className="label" htmlFor="abstract">Abstract</label>
              <textarea
                id="abstract"
                rows={5}
                className="input resize-y"
                value={form.abstract}
                onChange={set("abstract")}
                placeholder="Paste the paper abstract here"
              />
            </div>
          </fieldset>

          {submitError && (
            <div className="flex items-start gap-2 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" /> {submitError}
            </div>
          )}
          {notice && (
            <div className="flex items-start gap-2 rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-700 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-300">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" /> {notice}
            </div>
          )}

          <div className="flex justify-end gap-2">
            <button type="button" onClick={reset} disabled={uploading} className="btn-ghost">
              Clear
            </button>
            <button type="submit" disabled={!file || uploading} className="btn-primary px-5">
              {uploading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {progress! < 1 ? `Uploading ${Math.round(progress! * 100)}%` : "Queuing…"}
                </>
              ) : (
                <>
                  <UploadCloud className="h-4 w-4" /> Upload and index
                </>
              )}
            </button>
          </div>
        </form>

        {/* Ingestion status */}
        <aside className="lg:pt-[80px]">
          <div className="card p-5 lg:sticky lg:top-6">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold">Ingestion status</h2>
              {hasActive && <Loader2 className="h-4 w-4 animate-spin text-brand-500" />}
            </div>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Markdown conversion, extraction and graph loading.
            </p>
            {entries.length === 0 ? (
              <p className="mt-6 text-center text-sm text-slate-400">No ingestions since the server started.</p>
            ) : (
              <ul className="mt-4 space-y-2">
                {entries.map(([name, s]) => {
                  const st = STATUS_STYLE[s] ?? STATUS_STYLE.pending;
                  const Icon = st.icon;
                  return (
                    <li key={name} className="flex items-center gap-2.5 rounded-xl border border-slate-100 p-2.5 dark:border-slate-800">
                      <FileText className="h-4 w-4 shrink-0 text-slate-400" />
                      <span className="min-w-0 flex-1 truncate text-sm" title={name}>
                        {name}
                      </span>
                      <span className={clsx("inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium", st.cls)}>
                        <Icon className={clsx("h-3 w-3", s === "processing" && "animate-spin")} />
                        {st.label}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
