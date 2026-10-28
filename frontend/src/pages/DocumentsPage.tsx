import clsx from "clsx";
import {
  AlertCircle,
  ArrowDownUp,
  BookOpen,
  CalendarDays,
  FileUp,
  LayoutGrid,
  List,
  MessageSquare,
  RefreshCw,
  Search,
  Users,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { getDocuments, type IndexedDocument } from "../lib/api";

type SortKey = "title" | "year-desc" | "year-asc";

function Stat({ icon: Icon, label, value }: { icon: typeof BookOpen; label: string; value: string | number }) {
  return (
    <div className="card flex items-center gap-3 px-4 py-3">
      <div className="grid h-9 w-9 place-items-center rounded-lg bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-300">
        <Icon className="h-[18px] w-[18px]" />
      </div>
      <div>
        <div className="text-lg font-semibold leading-none">{value}</div>
        <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{label}</div>
      </div>
    </div>
  );
}

function Authors({ authors, max = 4 }: { authors: string[]; max?: number }) {
  if (!authors.length) return <span className="text-slate-400">Autores desconocidos</span>;
  const shown = authors.slice(0, max);
  return (
    <span>
      {shown.join(", ")}
      {authors.length > max && <span className="text-slate-400"> +{authors.length - max} más</span>}
    </span>
  );
}

export default function DocumentsPage() {
  const [docs, setDocs] = useState<IndexedDocument[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortKey>("title");
  const [view, setView] = useState<"grid" | "list">("grid");
  const navigate = useNavigate();

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setDocs(await getDocuments());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const filtered = useMemo(() => {
    if (!docs) return [];
    const q = query.trim().toLowerCase();
    const list = docs.filter((d) =>
      !q
        ? true
        : [d.title, d.venue, String(d.year ?? ""), ...(d.authors ?? [])].some((v) => v?.toLowerCase().includes(q)),
    );
    return list.sort((a, b) => {
      if (sort === "title") return (a.title ?? "").localeCompare(b.title ?? "");
      const diff = (a.year ?? 0) - (b.year ?? 0);
      return sort === "year-asc" ? diff : -diff;
    });
  }, [docs, query, sort]);

  const stats = useMemo(() => {
    const authors = new Set<string>();
    const years: number[] = [];
    docs?.forEach((d) => {
      d.authors?.forEach((a) => authors.add(a));
      if (d.year) years.push(d.year);
    });
    const range = years.length ? `${Math.min(...years)}–${Math.max(...years)}` : "—";
    return { authors: authors.size, range };
  }, [docs]);

  const askAbout = (d: IndexedDocument) =>
    navigate("/chat", { state: { prefill: `Resume el paper "${d.title}": objetivo, método y resultados principales.` } });

  return (
    <div className="scrollbar-thin h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl px-5 py-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Documentos indexados</h1>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              Papers presentes en el grafo de conocimiento de Neo4j.
            </p>
          </div>
          <div className="flex gap-2">
            <button onClick={load} disabled={loading} className="btn-outline">
              <RefreshCw className={clsx("h-4 w-4", loading && "animate-spin")} /> Actualizar
            </button>
            <Link to="/upload" className="btn-primary">
              <FileUp className="h-4 w-4" /> Subir paper
            </Link>
          </div>
        </div>

        <div className="mt-6 grid gap-3 sm:grid-cols-3">
          <Stat icon={BookOpen} label="Papers" value={docs?.length ?? "—"} />
          <Stat icon={Users} label="Autores únicos" value={docs ? stats.authors : "—"} />
          <Stat icon={CalendarDays} label="Rango de años" value={docs ? stats.range : "—"} />
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-2">
          <div className="relative min-w-[220px] flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Buscar por título, autor, venue o año…"
              className="input pl-9"
            />
          </div>
          <div className="relative">
            <ArrowDownUp className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)} className="input w-auto pl-9 pr-8">
              <option value="title">Título (A–Z)</option>
              <option value="year-desc">Más recientes</option>
              <option value="year-asc">Más antiguos</option>
            </select>
          </div>
          <div className="flex rounded-xl border border-slate-200 bg-white p-1 dark:border-slate-700 dark:bg-slate-900">
            {(["grid", "list"] as const).map((v) => {
              const Icon = v === "grid" ? LayoutGrid : List;
              return (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  className={clsx(
                    "rounded-lg p-1.5",
                    view === v ? "bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-white" : "text-slate-400",
                  )}
                  aria-label={v === "grid" ? "Vista cuadrícula" : "Vista lista"}
                >
                  <Icon className="h-4 w-4" />
                </button>
              );
            })}
          </div>
        </div>

        {error && (
          <div className="mt-6 flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              <p className="font-medium">No se pudieron cargar los documentos</p>
              <p className="mt-0.5 opacity-80">{error}</p>
            </div>
          </div>
        )}

        {!docs && loading && (
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="card h-36 animate-pulse bg-slate-100 dark:bg-slate-800/50" />
            ))}
          </div>
        )}

        {docs && filtered.length === 0 && (
          <div className="card mt-6 flex flex-col items-center px-6 py-14 text-center">
            <BookOpen className="h-10 w-10 text-slate-300 dark:text-slate-600" />
            <p className="mt-3 font-medium">{docs.length ? "Ningún paper coincide con la búsqueda" : "Aún no hay papers indexados"}</p>
            {!docs.length && (
              <Link to="/upload" className="btn-primary mt-4">
                <FileUp className="h-4 w-4" /> Subir el primero
              </Link>
            )}
          </div>
        )}

        {filtered.length > 0 && view === "grid" && (
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            {filtered.map((d, i) => (
              <article
                key={d.id ?? i}
                className="card group flex flex-col p-5 transition hover:border-brand-200 hover:shadow-md dark:hover:border-brand-900"
              >
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  {d.year && (
                    <span className="rounded-md bg-brand-50 px-2 py-0.5 font-medium text-brand-700 dark:bg-brand-500/10 dark:text-brand-300">
                      {d.year}
                    </span>
                  )}
                  {d.venue && (
                    <span className="rounded-md bg-slate-100 px-2 py-0.5 font-medium text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                      {d.venue}
                    </span>
                  )}
                </div>
                <h2 className="mt-3 font-semibold leading-snug">{d.title ?? "Sin título"}</h2>
                <p className="mt-2 flex items-start gap-1.5 text-sm text-slate-500 dark:text-slate-400">
                  <Users className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <Authors authors={d.authors ?? []} />
                </p>
                <div className="mt-auto flex items-center justify-between pt-4">
                  <code className="truncate text-[11px] text-slate-400">{d.id}</code>
                  <button
                    onClick={() => askAbout(d)}
                    className="btn-ghost px-2.5 py-1.5 text-xs text-brand-600 dark:text-brand-400"
                  >
                    <MessageSquare className="h-3.5 w-3.5" /> Preguntar
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}

        {filtered.length > 0 && view === "list" && (
          <div className="card mt-6 overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-slate-200 text-xs uppercase tracking-wider text-slate-500 dark:border-slate-800">
                <tr>
                  <th className="px-4 py-3 font-medium">Título</th>
                  <th className="px-4 py-3 font-medium">Autores</th>
                  <th className="px-4 py-3 font-medium">Venue</th>
                  <th className="px-4 py-3 font-medium">Año</th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {filtered.map((d, i) => (
                  <tr key={d.id ?? i} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                    <td className="max-w-md px-4 py-3 font-medium">{d.title ?? "Sin título"}</td>
                    <td className="px-4 py-3 text-slate-500 dark:text-slate-400">
                      <Authors authors={d.authors ?? []} max={2} />
                    </td>
                    <td className="px-4 py-3 text-slate-500 dark:text-slate-400">{d.venue ?? "—"}</td>
                    <td className="px-4 py-3 tabular-nums text-slate-500 dark:text-slate-400">{d.year ?? "—"}</td>
                    <td className="px-4 py-3 text-right">
                      <button onClick={() => askAbout(d)} className="btn-ghost p-1.5" aria-label="Preguntar">
                        <MessageSquare className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
