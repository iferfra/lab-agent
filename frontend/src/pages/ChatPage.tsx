import clsx from "clsx";
import { ArrowUp, Plus, RotateCcw, Sparkles, Square, Trash2, User } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Markdown, { CopyButton } from "../components/Markdown";
import { sendChat } from "../lib/api";
import { newConversation, uid, useConversations, type ChatMessage, type Conversation } from "../lib/conversations";

const SUGGESTIONS = [
  "¿Qué papers hay indexados y de qué tratan?",
  "Resume las contribuciones principales de cada paper",
  "¿Qué métricas y datasets se utilizan en los experimentos?",
  "¿Qué autores han colaborado entre sí?",
];

function formatTitle(text: string) {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > 48 ? t.slice(0, 48) + "…" : t;
}

function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-3 text-sm text-slate-500 dark:text-slate-400">
      <div className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-2 w-2 animate-blink rounded-full bg-brand-500"
            style={{ animationDelay: `${i * 0.2}s` }}
          />
        ))}
      </div>
      Consultando el grafo de conocimiento…
    </div>
  );
}

function AssistantAvatar() {
  return (
    <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-sm">
      <Sparkles className="h-4 w-4" />
    </div>
  );
}

function MessageRow({ msg, onRetry }: { msg: ChatMessage; onRetry?: () => void }) {
  if (msg.role === "user") {
    return (
      <div className="flex animate-fadeIn justify-end gap-3">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-tr-md bg-brand-600 px-4 py-2.5 text-[15px] leading-relaxed text-white shadow-sm">
          {msg.content}
        </div>
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-slate-200 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
          <User className="h-4 w-4" />
        </div>
      </div>
    );
  }
  return (
    <div className="group flex animate-fadeIn gap-3">
      <AssistantAvatar />
      <div className="min-w-0 flex-1">
        {msg.error ? (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300">
            <p>{msg.content}</p>
            {onRetry && (
              <button onClick={onRetry} className="mt-2 inline-flex items-center gap-1.5 font-medium hover:underline">
                <RotateCcw className="h-3.5 w-3.5" /> Reintentar
              </button>
            )}
          </div>
        ) : (
          <>
            <Markdown content={msg.content} />
            <div className="mt-2 flex items-center gap-3 text-xs text-slate-400 opacity-0 transition group-hover:opacity-100">
              <CopyButton
                getText={() => msg.content}
                className="flex items-center gap-1 rounded-md px-1.5 py-1 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-800 dark:hover:text-slate-200"
              />
              {msg.durationMs !== undefined && <span>{(msg.durationMs / 1000).toFixed(1)} s</span>}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function ChatPage() {
  const [conversations, setConversations] = useConversations();
  const [activeId, setActiveId] = useState(() => conversations[0].id);
  const [input, setInput] = useState("");
  const [pendingId, setPendingId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const location = useLocation();
  const navigate = useNavigate();

  const active = conversations.find((c) => c.id === activeId) ?? conversations[0];
  const loading = pendingId === active.id;
  const sorted = useMemo(
    () => [...conversations].sort((a, b) => b.updatedAt - a.updatedAt),
    [conversations],
  );

  const update = (id: string, fn: (c: Conversation) => Conversation) =>
    setConversations((cs) => cs.map((c) => (c.id === id ? fn(c) : c)));

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [active.messages.length, loading]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [input]);

  // Prefilled question coming from the documents page
  useEffect(() => {
    const prefill = (location.state as { prefill?: string } | null)?.prefill;
    if (prefill) {
      setInput(prefill);
      textareaRef.current?.focus();
      navigate(location.pathname, { replace: true, state: null });
    }
  }, [location.state, location.pathname, navigate]);

  async function ask(question: string, convId = active.id) {
    const text = question.trim();
    if (!text || pendingId) return;

    const userMsg: ChatMessage = { id: uid(), role: "user", content: text, createdAt: Date.now() };
    update(convId, (c) => ({
      ...c,
      title: c.messages.length ? c.title : formatTitle(text),
      messages: [...c.messages.filter((m) => !m.error), userMsg],
      updatedAt: Date.now(),
    }));
    setInput("");
    setPendingId(convId);

    const controller = new AbortController();
    abortRef.current = controller;
    const start = performance.now();
    let reply: ChatMessage;
    try {
      const res = await sendChat(text, controller.signal);
      reply = {
        id: uid(),
        role: "assistant",
        content: res.response || "_El agente no devolvió respuesta._",
        createdAt: Date.now(),
        durationMs: performance.now() - start,
      };
    } catch (e) {
      const aborted = (e as Error).name === "AbortError";
      reply = {
        id: uid(),
        role: "assistant",
        content: aborted ? "Consulta cancelada." : (e as Error).message,
        error: true,
        createdAt: Date.now(),
      };
    }
    update(convId, (c) => ({ ...c, messages: [...c.messages, reply], updatedAt: Date.now() }));
    setPendingId(null);
    abortRef.current = null;
  }

  function retry() {
    const lastUser = [...active.messages].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    update(active.id, (c) => ({ ...c, messages: c.messages.filter((m) => m.id !== lastUser.id && !m.error) }));
    ask(lastUser.content);
  }

  function startNew() {
    const empty = conversations.find((c) => c.messages.length === 0);
    if (empty) return setActiveId(empty.id);
    const conv = newConversation();
    setConversations((cs) => [conv, ...cs]);
    setActiveId(conv.id);
    textareaRef.current?.focus();
  }

  function remove(id: string) {
    setConversations((cs) => {
      const rest = cs.filter((c) => c.id !== id);
      const next = rest.length ? rest : [newConversation()];
      if (id === activeId) setActiveId(next[0].id);
      return next;
    });
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      ask(input);
    }
  }

  const lastMsg = active.messages[active.messages.length - 1];

  return (
    <div className="flex h-full">
      {/* Conversation history */}
      <aside className="hidden w-64 shrink-0 flex-col border-r border-slate-200 dark:border-slate-800 lg:flex">
        <div className="p-3">
          <button onClick={startNew} className="btn-outline w-full">
            <Plus className="h-4 w-4" /> Nueva conversación
          </button>
        </div>
        <div className="scrollbar-thin flex-1 space-y-0.5 overflow-y-auto px-3 pb-3">
          <div className="px-2 pb-1 pt-2 text-xs font-medium uppercase tracking-wider text-slate-400">Historial</div>
          {sorted.map((c) => (
            <div
              key={c.id}
              onClick={() => setActiveId(c.id)}
              className={clsx(
                "group flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-2 text-sm transition",
                c.id === active.id
                  ? "bg-slate-200/70 text-slate-900 dark:bg-slate-800 dark:text-slate-100"
                  : "text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800/60",
              )}
            >
              <span className="flex-1 truncate">{c.title}</span>
              {pendingId === c.id && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-brand-500" />}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  remove(c.id);
                }}
                className="rounded p-1 text-slate-400 opacity-0 hover:text-red-500 group-hover:opacity-100"
                aria-label="Eliminar conversación"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* Chat */}
      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-3 dark:border-slate-800">
          <h1 className="truncate text-sm font-medium text-slate-700 dark:text-slate-200">{active.title}</h1>
          <button onClick={startNew} className="btn-ghost px-2.5 py-1.5 lg:hidden" aria-label="Nueva conversación">
            <Plus className="h-4 w-4" />
          </button>
        </div>

        <div ref={scrollRef} className="scrollbar-thin flex-1 overflow-y-auto">
          {active.messages.length === 0 ? (
            <div className="mx-auto flex h-full max-w-2xl flex-col items-center justify-center px-6 py-10 text-center">
              <div className="mb-5 grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-lg shadow-brand-600/30">
                <Sparkles className="h-7 w-7" />
              </div>
              <h2 className="text-2xl font-semibold tracking-tight">¿Qué quieres saber de tus papers?</h2>
              <p className="mt-2 max-w-md text-sm text-slate-500 dark:text-slate-400">
                Asistente académico sobre un grafo de conocimiento en Neo4j. Pregunta por papers, autores, métodos,
                datasets o métricas.
              </p>
              <div className="mt-8 grid w-full gap-2.5 sm:grid-cols-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => ask(s)}
                    className="card px-4 py-3 text-left text-sm text-slate-600 transition hover:border-brand-300 hover:shadow-md dark:text-slate-300 dark:hover:border-brand-700"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-7 px-5 py-8">
              {active.messages.map((m) => (
                <MessageRow
                  key={m.id}
                  msg={m}
                  onRetry={m.error && m === lastMsg && !pendingId ? retry : undefined}
                />
              ))}
              {loading && (
                <div className="flex gap-3">
                  <AssistantAvatar />
                  <div className="pt-1.5">
                    <ThinkingIndicator />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="px-4 pb-4 pt-2">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              ask(input);
            }}
            className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-lg shadow-slate-200/50 transition focus-within:border-brand-300 dark:border-slate-700 dark:bg-slate-900 dark:shadow-none dark:focus-within:border-brand-700"
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder="Escribe tu pregunta…"
              className="max-h-[200px] flex-1 resize-none bg-transparent px-3 py-2 text-[15px] outline-none placeholder:text-slate-400"
            />
            {loading ? (
              <button
                type="button"
                onClick={() => abortRef.current?.abort()}
                className="grid h-10 w-10 place-items-center rounded-xl bg-slate-800 text-white hover:bg-slate-700 dark:bg-slate-700"
                aria-label="Cancelar"
              >
                <Square className="h-4 w-4 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim() || !!pendingId}
                className="grid h-10 w-10 place-items-center rounded-xl bg-brand-600 text-white transition hover:bg-brand-700 disabled:bg-slate-200 disabled:text-slate-400 dark:disabled:bg-slate-800"
                aria-label="Enviar"
              >
                <ArrowUp className="h-5 w-5" />
              </button>
            )}
          </form>
          <p className="mt-2 text-center text-xs text-slate-400">
            Enter para enviar · Shift + Enter para nueva línea
          </p>
        </div>
      </section>
    </div>
  );
}
