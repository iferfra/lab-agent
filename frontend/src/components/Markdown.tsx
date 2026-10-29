import { Check, Copy } from "lucide-react";
import { memo, useRef, useState, type ComponentPropsWithoutRef } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

export function CopyButton({ getText, className }: { getText: () => string; className?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(getText());
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          /* clipboard unavailable */
        }
      }}
      className={className}
      aria-label="Copy"
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      <span>{copied ? "Copied" : "Copy"}</span>
    </button>
  );
}

function CodeBlock({ children, ...props }: ComponentPropsWithoutRef<"pre">) {
  const ref = useRef<HTMLPreElement>(null);
  const child = Array.isArray(children) ? children[0] : children;
  const className: string = (child as any)?.props?.className ?? "";
  const lang = /language-(\w+)/.exec(className)?.[1] ?? "code";

  return (
    <div className="not-prose my-4 overflow-hidden rounded-xl border border-slate-800 bg-[#0d1117] text-[13px]">
      <div className="flex items-center justify-between border-b border-slate-800 px-4 py-1.5 text-xs text-slate-400">
        <span className="font-mono">{lang}</span>
        <CopyButton
          getText={() => ref.current?.innerText ?? ""}
          className="flex items-center gap-1 rounded-md px-2 py-1 hover:bg-slate-800 hover:text-slate-200"
        />
      </div>
      <pre ref={ref} {...props} className="markdown overflow-x-auto p-4 font-mono leading-relaxed text-slate-100">
        {children}
      </pre>
    </div>
  );
}

function Markdown({ content }: { content: string }) {
  return (
    <div
      className="markdown prose prose-slate max-w-none text-[15px] leading-relaxed dark:prose-invert
        prose-headings:font-semibold prose-headings:tracking-tight prose-h1:text-xl prose-h2:text-lg prose-h3:text-base
        prose-a:text-brand-600 prose-a:no-underline hover:prose-a:underline dark:prose-a:text-brand-400
        prose-p:my-2.5 prose-li:my-0.5 prose-ul:my-2.5 prose-ol:my-2.5
        prose-blockquote:border-l-brand-400 prose-blockquote:font-normal prose-blockquote:not-italic
        prose-th:bg-slate-50 prose-th:px-3 prose-th:py-2 prose-td:px-3 prose-td:py-2 dark:prose-th:bg-slate-800/60"
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }], rehypeKatex]}
        components={{
          pre: CodeBlock,
          a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
          table: ({ node: _node, ...props }) => (
            <div className="my-4 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700">
              <table {...props} />
            </div>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

export default memo(Markdown);
