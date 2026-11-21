import { useCallback, useEffect, useRef, useState } from "react";
import { checkHealth, extractDocument, type ExtractionResponse, type FieldEvidence } from "./api";
import { useTheme } from "./useTheme";

type ItemStatus = "queued" | "processing" | "done" | "error";

interface QueueItem {
  id: string;
  file: File;
  status: ItemStatus;
  result?: ExtractionResponse;
  error?: string;
}

const STATUS_LABEL: Record<ItemStatus, string> = {
  queued: "W kolejce",
  processing: "Przetwarzanie…",
  done: "Gotowe",
  error: "Błąd",
};

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
    </svg>
  );
}

function downloadJson(name: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name.replace(/\.pdf$/i, "") + ".json";
  a.click();
  URL.revokeObjectURL(url);
}

function formatJsonKey(fieldPath: string): string {
  const leaf = fieldPath.split(".").pop() ?? fieldPath;
  return leaf.replace(/\[\d+\]$/, "");
}

function HighlightedPageText({ text, quote }: { text: string; quote: string | null }) {
  const markRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (markRef.current) {
      markRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [quote, text]);

  if (!quote || !quote.trim()) {
    return <pre className="pdf-page-text">{text}</pre>;
  }

  const escapedTokens = quote
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((tok) => tok.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));

  if (escapedTokens.length === 0) {
    return <pre className="pdf-page-text">{text}</pre>;
  }

  const pattern = new RegExp(`(${escapedTokens.join("\\s+")})`, "i");
  const match = pattern.exec(text);
  if (!match || match.index === undefined) {
    return <pre className="pdf-page-text">{text}</pre>;
  }

  const before = text.slice(0, match.index);
  const hit = text.slice(match.index, match.index + match[0].length);
  const after = text.slice(match.index + match[0].length);

  return (
    <pre className="pdf-page-text">
      {before}
      <mark ref={markRef} className="pdf-highlight">
        {hit}
      </mark>
      {after}
    </pre>
  );
}

function QuotesWithPdfView({
  evidence,
  pages,
  pageImages,
}: {
  evidence: FieldEvidence[];
  pages: string[];
  pageImages: string[];
}) {
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [viewMode, setViewMode] = useState<"pdf" | "text">("pdf");
  const [zoom, setZoom] = useState<number>(100);
  const activeRectRef = useRef<HTMLDivElement | null>(null);
  const activePageRef = useRef<HTMLDivElement | null>(null);

  const activeEv: FieldEvidence | undefined = evidence[selectedIdx] ?? evidence[0];
  const activePage = activeEv?.page ?? 1;
  const fullDocumentText = (pages.length > 0 ? pages : ["(Brak warstwy tekstowej PDF)"]).join("\n\n");

  useEffect(() => {
    const target = activeRectRef.current ?? activePageRef.current;
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [selectedIdx, viewMode, activePage]);

  return (
    <div className="quotes-split">
      <div className="quotes-fields">
        <div className="quotes-pane-header">
          <span>Cytaty i odnośniki ({evidence.length})</span>
          <span className="quotes-pane-hint">Kliknij pole, aby podświetlić w dokumencie</span>
        </div>
        {evidence.length === 0 ? (
          <div className="empty">Brak wyekstrahowanych pól do zacytowania.</div>
        ) : (
          <div className="evidence-list">
            {evidence.map((ev, i) => {
              const isSelected = i === selectedIdx;
              return (
                <div
                  key={i}
                  className={`evidence-card clickable ${isSelected ? "active" : ""}`}
                  onClick={() => setSelectedIdx(i)}
                >
                  <div className="evidence-header">
                    <span className="evidence-field" title={ev.field}>"{formatJsonKey(ev.field)}"</span>
                    <span className={`ref-badge ${isSelected ? "active" : ""}`}>
                      {isSelected ? "Podświetlono" : "Pokaż"}
                    </span>
                  </div>
                  <div className="evidence-value">{String(ev.value)}</div>
                  <blockquote className="evidence-quote">
                    {ev.quote ? `„${ev.quote}”` : "Nie znaleziono bezpośredniego fragmentu w tekście."}
                  </blockquote>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="quotes-viewer">
        <div className="viewer-toolbar">
          <div className="viewer-active-info">
            {activeEv ? (
              <>
                <span className="viewer-active-field" title={activeEv.field}>"{formatJsonKey(activeEv.field)}":</span>{" "}
                <strong className="viewer-active-val">{String(activeEv.value)}</strong>
              </>
            ) : (
              <span>Eksplorator PDF</span>
            )}
          </div>
          <div className="viewer-controls">
            {viewMode === "pdf" && (
              <div className="zoom-controls">
                <button
                  type="button"
                  className="zoom-btn"
                  onClick={() => setZoom((z) => Math.max(70, z - 15))}
                  title="Pomniejsz"
                >
                  −
                </button>
                <span className="zoom-label">{zoom}%</span>
                <button
                  type="button"
                  className="zoom-btn"
                  onClick={() => setZoom((z) => Math.min(175, z + 15))}
                  title="Powiększ"
                >
                  +
                </button>
              </div>
            )}
            <div className="tabs">
              <button
                type="button"
                className={viewMode === "pdf" ? "tab active" : "tab"}
                onClick={() => setViewMode("pdf")}
              >
                Eksplorator PDF
              </button>
              <button
                type="button"
                className={viewMode === "text" ? "tab active" : "tab"}
                onClick={() => setViewMode("text")}
              >
                Tekst
              </button>
            </div>
          </div>
        </div>

        {activeEv?.quote && (
          <div className="viewer-quote-banner">
            <span className="viewer-quote-label">Podświetlony fragment:</span>{" "}
            <em>„{activeEv.quote}”</em>
          </div>
        )}

        {viewMode === "pdf" && pageImages.length > 0 ? (
          <div className="pdf-explorer-scroll">
            <div className="pdf-document-sheet" style={{ width: `${zoom}%` }}>
              {pageImages.map((imgSrc, pageIdx) => {
                const pNum = pageIdx + 1;
                const isTargetPage = pNum === activePage;
                const pageEvidences = evidence
                  .map((ev, evIdx) => ({ ev, evIdx }))
                  .filter(({ ev }) => (ev.page ?? 1) === pNum && (ev.rects?.length ?? 0) > 0);

                return (
                  <div
                    key={pNum}
                    ref={isTargetPage ? activePageRef : null}
                    className="pdf-canvas-stage"
                  >
                    <img src={imgSrc} alt="Dokument PDF" className="pdf-page-img" />
                    {pageEvidences.map(({ ev, evIdx }) => {
                      const isActive = evIdx === selectedIdx;
                      return (ev.rects ?? []).map((rect, rIdx) => (
                        <div
                          key={`${evIdx}-${rIdx}`}
                          ref={isActive && rIdx === 0 ? activeRectRef : null}
                          className={`pdf-rect-overlay ${isActive ? "active" : "inactive"}`}
                          style={{
                            left: `${rect.x}%`,
                            top: `${rect.y}%`,
                            width: `${rect.w}%`,
                            height: `${rect.h}%`,
                          }}
                          title={`"${formatJsonKey(ev.field)}": ${String(ev.value)}`}
                          onClick={() => setSelectedIdx(evIdx)}
                        />
                      ));
                    })}
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="pdf-text-pages">
            <div className="pdf-page-box target-page">
              <HighlightedPageText
                text={fullDocumentText}
                quote={activeEv?.quote ?? null}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ResultView({ item }: { item: QueueItem }) {
  const [tab, setTab] = useState<"quotes" | "json" | "meta">("quotes");
  const [copied, setCopied] = useState(false);

  if (item.status === "queued") return <div className="empty">Plik czeka w kolejce.</div>;
  if (item.status === "processing")
    return (
      <div className="empty">
        <div className="spinner" />
        Trwa ekstrakcja danych…
      </div>
    );
  if (item.status === "error") return <div className="alert alert-error">{item.error}</div>;

  const { data, meta } = item.result!;
  const json = JSON.stringify(data, null, 2);
  const isNegative = Object.keys(data).length === 0;
  const evidence = meta.evidence ?? [];
  const pages = meta.pages ?? [];
  const pageImages = meta.page_images ?? [];

  const copy = async () => {
    await navigator.clipboard.writeText(json);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="result">
      <div className="stats">
        <div className="stat">
          <span className="stat-label">Klasa</span>
          <span className="stat-value">{meta.doc_type}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Pewność</span>
          <span className="stat-value">{(meta.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="stat">
          <span className="stat-label">Walidacja</span>
          <span className={`badge ${meta.valid ? "badge-ok" : "badge-warn"}`}>{meta.valid ? "OK" : "Błędy"}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Czas</span>
          <span className="stat-value">{meta.elapsed_seconds.toFixed(1)} s</span>
        </div>
        <div className="stat">
          <span className="stat-label">Tokeny</span>
          <span className="stat-value">{meta.usage.total_tokens}</span>
        </div>
      </div>

      {isNegative && (
        <div className="alert alert-info">Dokument rozpoznany jako spoza zakresu — nie wyekstrahowano danych.</div>
      )}
      {meta.validation_errors.length > 0 && (
        <div className="alert alert-error">
          <strong>Błędy walidacji</strong>
          <ul>{meta.validation_errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
        </div>
      )}
      {meta.validation_warnings.length > 0 && (
        <div className="alert alert-warn">
          <strong>Ostrzeżenia</strong>
          <ul>{meta.validation_warnings.map((e, i) => <li key={i}>{e}</li>)}</ul>
        </div>
      )}

      <div className="toolbar">
        <div className="tabs">
          <button className={tab === "quotes" ? "tab active" : "tab"} onClick={() => setTab("quotes")}>
            Cytaty i Eksplorator PDF ({evidence.length})
          </button>
          <button className={tab === "json" ? "tab active" : "tab"} onClick={() => setTab("json")}>JSON UZP</button>
          <button className={tab === "meta" ? "tab active" : "tab"} onClick={() => setTab("meta")}>Metadane</button>
        </div>
        <div className="actions">
          <button className="btn" onClick={copy}>{copied ? "Skopiowano" : "Kopiuj JSON"}</button>
          <button className="btn" onClick={() => downloadJson(item.file.name, data)}>Pobierz JSON</button>
        </div>
      </div>
      {tab === "quotes" ? (
        <QuotesWithPdfView evidence={evidence} pages={pages} pageImages={pageImages} />
      ) : (
        <pre className="code">{tab === "json" ? json : JSON.stringify(meta, null, 2)}</pre>
      )}
    </div>
  );
}

export default function App() {
  const { theme, toggle } = useTheme();
  const [items, setItems] = useState<QueueItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [concurrency, setConcurrency] = useState<number>(2);
  const inputRef = useRef<HTMLInputElement>(null);
  const activeIds = useRef<Set<string>>(new Set());

  useEffect(() => {
    checkHealth().then(setOnline);
  }, []);

  const update = (id: string, patch: Partial<QueueItem>) =>
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...patch } : it)));

  useEffect(() => {
    const slots = Math.max(0, concurrency - activeIds.current.size);
    if (slots === 0) return;
    const queued = items.filter((it) => it.status === "queued" && !activeIds.current.has(it.id)).slice(0, slots);
    if (!queued.length) return;

    for (const next of queued) {
      activeIds.current.add(next.id);
      update(next.id, { status: "processing" });
      extractDocument(next.file)
        .then((result) => update(next.id, { status: "done", result }))
        .catch((err: Error) => update(next.id, { status: "error", error: err.message }))
        .finally(() => {
          activeIds.current.delete(next.id);
          setItems((prev) => [...prev]);
        });
    }
  }, [items, concurrency]);

  const addFiles = useCallback((files: FileList | File[]) => {
    const pdfs = Array.from(files).filter((f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name));
    if (!pdfs.length) return;
    const newItems = pdfs.map((file) => ({ id: crypto.randomUUID(), file, status: "queued" as const }));
    setItems((prev) => [...prev, ...newItems]);
    setSelectedId((cur) => cur ?? newItems[0].id);
  }, []);

  const selected = items.find((it) => it.id === selectedId) ?? null;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="logo">{"{ }"}</div>
          <div>
            <h1>Odwołania2json</h1>
            <p className="subtitle">Ekstrakcja danych ze spraw odwoławczych KIO · PDF → JSON UZP</p>
          </div>
        </div>
        <div className="header-right">
          <label className="concurrency-select" title="Liczba dokumentów przetwarzanych równolegle">
            <span>Równolegle:</span>
            <select value={concurrency} onChange={(e) => setConcurrency(Number(e.target.value))}>
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={4}>4</option>
              <option value={8}>8</option>
            </select>
          </label>
          <span className={`status-dot ${online ? "on" : online === false ? "off" : ""}`} />
          <span className="status-text">{online ? "API online" : online === false ? "API offline" : "…"}</span>
          <button className="icon-btn" onClick={toggle} aria-label="Przełącz motyw" title={theme === "dark" ? "Tryb dzienny" : "Tryb nocny"}>
            {theme === "dark" ? <SunIcon /> : <MoonIcon />}
          </button>
        </div>
      </header>

      <main className="main">
        <aside className="sidebar">
          <div
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              addFiles(e.dataTransfer.files);
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="application/pdf,.pdf"
              multiple
              hidden
              onChange={(e) => {
                if (e.target.files) addFiles(e.target.files);
                e.target.value = "";
              }}
            />
            <div className="dropzone-icon">↑</div>
            <div className="dropzone-title">Upuść pliki PDF</div>
            <div className="dropzone-hint">lub kliknij, aby wybrać</div>
          </div>

          {items.length > 0 && (
            <>
              <div className="list-header">
                <span>Dokumenty ({items.length})</span>
                <button className="link" onClick={() => { setItems([]); setSelectedId(null); }} disabled={items.some((i) => i.status === "processing")}>
                  Wyczyść
                </button>
              </div>
              <ul className="file-list">
                {items.map((it) => (
                  <li key={it.id} className={`file-item ${it.id === selectedId ? "selected" : ""}`} onClick={() => setSelectedId(it.id)}>
                    <span className="file-name" title={it.file.name}>{it.file.name}</span>
                    <span className={`pill pill-${it.status}`}>
                      {it.status === "done" ? it.result?.meta.doc_type : STATUS_LABEL[it.status]}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </aside>

        <section className="content">
          {selected ? (
            <>
              <h2 className="content-title">{selected.file.name}</h2>
              <ResultView key={selected.id} item={selected} />
            </>
          ) : (
            <div className="empty big">
              Wgraj dokument (odwołanie, pełnomocnictwo lub potwierdzenie przelewu), aby zobaczyć wynik ekstrakcji.
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
