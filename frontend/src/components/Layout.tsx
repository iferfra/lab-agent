import clsx from "clsx";
import { FileUp, Library, Menu, MessageSquare, Moon, Sun, X } from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

const NAV = [
  { to: "/chat", label: "Chat", icon: MessageSquare, hint: "Pregunta al grafo" },
  { to: "/documents", label: "Documentos", icon: Library, hint: "Papers indexados" },
  { to: "/upload", label: "Subir paper", icon: FileUp, hint: "Ingesta de PDFs" },
];

function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("theme", dark ? "dark" : "light");
    } catch {
      /* storage unavailable */
    }
  }, [dark]);
  return [dark, setDark] as const;
}

export function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 shadow-md shadow-brand-600/30">
        <svg viewBox="0 0 32 32" className="h-5 w-5">
          <g stroke="#fff" strokeWidth="2.2" fill="none">
            <path d="M9 10l14 3M9 10l5 13M23 13l-9 10" />
          </g>
          <g fill="#fff">
            <circle cx="9" cy="10" r="3.2" />
            <circle cx="23" cy="13" r="3.2" />
            <circle cx="14" cy="23" r="3.2" />
          </g>
        </svg>
      </div>
      <div className="leading-tight">
        <div className="text-[15px] font-semibold tracking-tight">Lab Agent</div>
        <div className="text-xs text-slate-500 dark:text-slate-400">GraphRAG · Neo4j</div>
      </div>
    </div>
  );
}

export default function Layout() {
  const [dark, setDark] = useTheme();
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  useEffect(() => setMobileOpen(false), [location.pathname]);

  const sidebar = (
    <div className="flex h-full flex-col gap-6 p-4">
      <div className="px-2 pt-1">
        <Logo />
      </div>
      <nav className="flex flex-col gap-1">
        {NAV.map(({ to, label, icon: Icon, hint }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              clsx(
                "group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition",
                isActive
                  ? "bg-brand-50 text-brand-700 dark:bg-brand-500/10 dark:text-brand-300"
                  : "text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800/70",
              )
            }
          >
            <Icon className="h-[18px] w-[18px] shrink-0" />
            <div className="flex flex-col">
              <span className="font-medium">{label}</span>
              <span className="text-xs text-slate-400 dark:text-slate-500">{hint}</span>
            </div>
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto flex items-center justify-between rounded-xl border border-slate-200 px-3 py-2 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
        <span>Tema {dark ? "oscuro" : "claro"}</span>
        <button
          onClick={() => setDark(!dark)}
          className="rounded-lg p-1.5 hover:bg-slate-100 dark:hover:bg-slate-800"
          aria-label="Cambiar tema"
        >
          {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </div>
    </div>
  );

  return (
    <div className="flex h-full">
      <aside className="hidden w-64 shrink-0 border-r border-slate-200 bg-white/70 backdrop-blur dark:border-slate-800 dark:bg-slate-900/60 md:block">
        {sidebar}
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div className="absolute inset-0 bg-slate-900/40" onClick={() => setMobileOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-72 bg-white shadow-xl dark:bg-slate-900">
            <button className="btn-ghost absolute right-2 top-3 p-2" onClick={() => setMobileOpen(false)}>
              <X className="h-5 w-5" />
            </button>
            {sidebar}
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-slate-200 bg-white px-4 py-3 dark:border-slate-800 dark:bg-slate-900 md:hidden">
          <button className="btn-ghost p-2" onClick={() => setMobileOpen(true)} aria-label="Abrir menú">
            <Menu className="h-5 w-5" />
          </button>
          <Logo />
        </header>
        <main className="min-h-0 flex-1">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
