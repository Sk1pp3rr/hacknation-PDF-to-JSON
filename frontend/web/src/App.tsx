import { useCallback, useEffect, useRef, useState } from "react";
import { checkHealth, extractDocument, type ExtractionResponse, type FieldEvidence } from "./api";
import { useTheme } from "./useTheme";

type ItemStatus = "queued" | "processing" | "done" | "error";

interface QueueItem {
  id: string;
  file: File;
  status: ItemStatus;
  result?: ExtractionResponse;
  originalData?: Record<string, unknown>;
  error?: string;
}

const CZYNNOSC_OPIS_BY_ID: Record<number, string> = {
  1: "Decyzja o odrzuceniu oferty",
  2: "Ogłoszenie wyboru oferty",
  3: "Decyzja o wykluczeniu wykonawcy",
  4: "Publikacja specyfikacji warunków zamówienia (SWZ)",
  5: "Modyfikacja treści SWZ",
  6: "Odpowiedzi/wyjaśnienia do SWZ",
  7: "Zaniechanie odrzucenia oferty",
  8: "Zaniechanie wykluczenia wykonawcy",
  9: "Decyzja o unieważnieniu postępowania",
  10: "Zaniechanie unieważnienia postępowania",
  11: "Opis przedmiotu zamówienia (OPZ)",
  12: "Projektowane postanowienia umowy",
};

const FRIENDLY_LABELS: Record<string, string> = {
  OdwolanieKIO: "Dane odwołania (KIO)",
  Odwolujacy: "Odwołujący",
  Zamawiajacy: "Zamawiający",
  ZamowieniePubliczne: "Zamówienie publiczne",
  przelew_bankowy: "Potwierdzenie przelewu",
  mocodawca: "Mocodawca",
  pelnomocnicy: "Pełnomocnicy",
  pelnomocnik: "Pełnomocnicy",
  podpisujacy: "Podpisujący",
  data: "Data pisma (YYYY-MM-DD)",
  czynnosc_dzien: "Data zaskarżonej czynności (YYYY-MM-DD)",
  czynnosc_powod_id: "Powód odwołania (1–12)",
  czynnosc_opis: "Opis zaskarżonej czynności",
  miejscowosc: "Miejscowość",
  nazwa: "Nazwa podmiotu / kancelarii",
  imie_nazwisko: "Imię i nazwisko",
  krs: "Numer KRS",
  nip: "Numer NIP",
  regon: "Numer REGON",
  pesel: "Numer PESEL",
  mail: "Adres e-mail",
  typ: "Typ podmiotu",
  "przez-pelnomocnika": "Działa przez pełnomocnika",
  prowadzacy_postepowanie: "Prowadzący postępowanie",
  numer_bzp: "Numer BZP",
  numer_ted: "Numer TED",
  numer_postepowania: "Numer postępowania",
  tytul: "Tytuł",
  data_operacji: "Data operacji (YYYY-MM-DD)",
  kwota: "Kwota (PLN)",
  z_rachunku: "Z rachunku (IBAN / NRB)",
  na_rachunek: "Na rachunek (IBAN / NRB)",
  numer_transakcji: "Numer transakcji",
};

function friendlyLabel(key: string): string {
  return FRIENDLY_LABELS[key] ?? key.replace(/[_-]+/g, " ");
}

function emptyFromTemplate(sample: unknown): unknown {
  if (Array.isArray(sample)) return [];
  if (sample && typeof sample === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(sample as Record<string, unknown>)) {
      out[k] = emptyFromTemplate(v);
    }
    return out;
  }
  return null;
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

function formatUzpJson(data: unknown): string {
  return JSON.stringify(data, null, 2).replace(
    /("kwota"\s*:\s*)(-?\d+(?:\.\d+)?)/g,
    (_match, prefix: string, num: string) => `${prefix}${Number(num).toFixed(2)}`,
  );
}

function downloadJson(name: string, data: unknown) {
  const blob = new Blob([formatUzpJson(data)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name.replace(/\.pdf$/i, "") + ".json";
  a.click();
  URL.revokeObjectURL(url);
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function downloadAllJsonZip(doneItems: QueueItem[]) {
  const enc = new TextEncoder();
  const used = new Map<string, number>();
  const localParts: Uint8Array[] = [];
  const centralParts: Uint8Array[] = [];
  let offset = 0;

  for (const item of doneItems) {
    const base = item.file.name.replace(/\.pdf$/i, "");
    const count = (used.get(base) ?? 0) + 1;
    used.set(base, count);
    const filename = count === 1 ? `${base}.json` : `${base}_${count}.json`;
    const nameBytes = enc.encode(filename);
    const dataBytes = enc.encode(formatUzpJson(item.result!.data) + "\n");
    const crc = crc32(dataBytes);

    const local = new Uint8Array(30 + nameBytes.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, 20, true);
    lv.setUint16(6, 0x0800, true);
    lv.setUint32(14, crc, true);
    lv.setUint32(18, dataBytes.length, true);
    lv.setUint32(22, dataBytes.length, true);
    lv.setUint16(26, nameBytes.length, true);
    local.set(nameBytes, 30);

    const central = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(central.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(8, 0x0800, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, dataBytes.length, true);
    cv.setUint32(24, dataBytes.length, true);
    cv.setUint16(28, nameBytes.length, true);
    cv.setUint32(42, offset, true);
    central.set(nameBytes, 46);

    localParts.push(local, dataBytes);
    centralParts.push(central);
    offset += local.length + dataBytes.length;
  }

  const centralSize = centralParts.reduce((acc, p) => acc + p.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, doneItems.length, true);
  ev.setUint16(10, doneItems.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);

  const blob = new Blob([...localParts, ...centralParts, eocd] as BlobPart[], { type: "application/zip" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "wyniki_json.zip";
  a.click();
  URL.revokeObjectURL(url);
}

function formatJsonKey(fieldPath: string): string {
  const leaf = fieldPath.split(".").pop() ?? fieldPath;
  return leaf.replace(/\[\d+\]$/, "");
}

function formatFieldValue(fieldPath: string, value: unknown): string {
  if (formatJsonKey(fieldPath) === "kwota" && typeof value === "number") {
    return value.toFixed(2);
  }
  return String(value);
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

function parseFieldPath(path: string): Array<string | number> {
  const tokens: Array<string | number> = [];
  for (const part of path.split(".")) {
    const re = /([^[\]]+)|\[(\d+)\]/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(part)) !== null) {
      if (m[1] !== undefined) tokens.push(m[1]);
      else if (m[2] !== undefined) tokens.push(Number(m[2]));
    }
  }
  return tokens;
}

function getValueAtPath(root: unknown, path: string): unknown {
  let cur: unknown = root;
  for (const key of parseFieldPath(path)) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string | number, unknown>)[key];
  }
  return cur;
}

function setValueAtPath(
  root: Record<string, unknown>,
  path: string,
  nextValue: unknown,
): Record<string, unknown> {
  const tokens = parseFieldPath(path);
  if (tokens.length === 0) return root;
  const clone = structuredClone(root);
  let cur: unknown = clone;
  for (let i = 0; i < tokens.length - 1; i++) {
    const k = tokens[i];
    if (cur == null || typeof cur !== "object") return root;
    cur = (cur as Record<string | number, unknown>)[k];
  }
  const last = tokens[tokens.length - 1];
  if (cur != null && typeof cur === "object") {
    (cur as Record<string | number, unknown>)[last] = nextValue;
  }
  return clone;
}

function QuotesWithPdfView({
  evidence,
  pages,
  pageImages,
  data,
  onUpdateData,
}: {
  evidence: FieldEvidence[];
  pages: string[];
  pageImages: string[];
  data: Record<string, unknown>;
  onUpdateData: (nextData: Record<string, unknown>) => void;
}) {
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [viewMode, setViewMode] = useState<"pdf" | "text">("pdf");
  const [zoom, setZoom] = useState<number>(100);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const activeRectRef = useRef<HTMLDivElement | null>(null);
  const activePageRef = useRef<HTMLDivElement | null>(null);
  const explorerRef = useRef<HTMLDivElement | null>(null);
  const dragStateRef = useRef<{
    x: number;
    y: number;
    scrollLeft: number;
    scrollTop: number;
    panX: number;
    panY: number;
    moved: boolean;
  } | null>(null);
  const suppressRectClickRef = useRef<boolean>(false);

  const activeEv: FieldEvidence | undefined = evidence[selectedIdx] ?? evidence[0];
  const activePage = activeEv?.page ?? 1;
  const fullDocumentText = (pages.length > 0 ? pages : ["(Brak warstwy tekstowej PDF)"]).join("\n\n");

  useEffect(() => {
    setPan({ x: 0, y: 0 });
    const target = activeRectRef.current ?? activePageRef.current;
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [selectedIdx, viewMode, activePage]);

  useEffect(() => {
    if (!isDragging) return;
    const handleMouseMove = (e: MouseEvent) => {
      const el = explorerRef.current;
      const drag = dragStateRef.current;
      if (!el || !drag) return;
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
        drag.moved = true;
      }
      const canScrollX = el.scrollWidth > el.clientWidth + 1;
      const canScrollY = el.scrollHeight > el.clientHeight + 1;
      if (canScrollX) {
        el.scrollLeft = drag.scrollLeft - dx;
      }
      if (canScrollY) {
        el.scrollTop = drag.scrollTop - dy;
      }
      if (!canScrollX || !canScrollY) {
        setPan({
          x: canScrollX ? drag.panX : Math.max(-320, Math.min(320, drag.panX + dx)),
          y: canScrollY ? drag.panY : Math.max(-320, Math.min(320, drag.panY + dy)),
        });
      }
    };
    const handleMouseUp = () => {
      if (dragStateRef.current?.moved) {
        suppressRectClickRef.current = true;
        window.setTimeout(() => {
          suppressRectClickRef.current = false;
        }, 0);
      }
      dragStateRef.current = null;
      setIsDragging(false);
    };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging]);

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
              const liveVal = getValueAtPath(data, ev.field);
              const effectiveVal = liveVal !== undefined ? liveVal : ev.value;
              const isEditableScalar =
                liveVal !== undefined &&
                (effectiveVal === null ||
                  typeof effectiveVal === "string" ||
                  typeof effectiveVal === "number");
              const isKwotaField = ev.field === "kwota" || ev.field.endsWith(".kwota");
              return (
                <div
                  key={i}
                  className={`evidence-card clickable ${isSelected ? "active" : ""}`}
                  onClick={() => {
                    setSelectedIdx(i);
                    if ((ev.rects?.length ?? 0) === 0 && pages.some((p) => p.trim().length > 0)) {
                      setViewMode("text");
                    }
                  }}
                >
                  <div className="evidence-header">
                    <span className="evidence-field" title={ev.field}>"{formatJsonKey(ev.field)}"</span>
                    <span className={`ref-badge ${isSelected ? "active" : ""}`}>
                      {isSelected ? "Podświetlono" : "Pokaż"}
                    </span>
                  </div>
                  <div className="evidence-value">{formatFieldValue(ev.field, effectiveVal)}</div>
                  {isSelected && isEditableScalar && (
                    <div className="evidence-inline-edit" onClick={(e) => e.stopPropagation()}>
                      {ev.field === "tytul" || ev.field.endsWith(".tytul") ? (
                        <textarea
                          className="editor-input editor-textarea"
                          rows={2}
                          value={effectiveVal == null ? "" : String(effectiveVal)}
                          placeholder="Wpisz lub popraw wartość…"
                          onChange={(e) =>
                            onUpdateData(
                              setValueAtPath(
                                data,
                                ev.field,
                                e.target.value === "" ? null : e.target.value,
                              ),
                            )
                          }
                        />
                      ) : (
                        <input
                          className="editor-input"
                          type={isKwotaField ? "number" : "text"}
                          step={isKwotaField ? "0.01" : undefined}
                          value={
                            isKwotaField && typeof effectiveVal === "number"
                              ? effectiveVal.toFixed(2)
                              : effectiveVal == null
                                ? ""
                                : String(effectiveVal)
                          }
                          placeholder="Wpisz lub popraw wartość…"
                          onChange={(e) => {
                            if (isKwotaField) {
                              const raw = e.target.value.replace(",", ".").trim();
                              if (!raw) {
                                onUpdateData(setValueAtPath(data, ev.field, null));
                                return;
                              }
                              const parsed = Number(raw);
                              if (!Number.isNaN(parsed)) {
                                onUpdateData(
                                  setValueAtPath(data, ev.field, Math.round(parsed * 100) / 100),
                                );
                              }
                              return;
                            }
                            onUpdateData(
                              setValueAtPath(
                                data,
                                ev.field,
                                e.target.value === "" ? null : e.target.value,
                              ),
                            );
                          }}
                        />
                      )}
                    </div>
                  )}
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
                <strong className="viewer-active-val">{formatFieldValue(activeEv.field, activeEv.value)}</strong>
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
                  onClick={() => setZoom((z) => Math.max(70, z - 20))}
                  title="Pomniejsz"
                >
                  −
                </button>
                <span className="zoom-label">{zoom}%</span>
                <button
                  type="button"
                  className="zoom-btn"
                  onClick={() => setZoom((z) => Math.min(250, z + 20))}
                  title="Powiększ"
                >
                  +
                </button>
                {(zoom !== 100 || pan.x !== 0 || pan.y !== 0) && (
                  <button
                    type="button"
                    className="zoom-btn zoom-reset-btn"
                    onClick={() => {
                      setZoom(100);
                      setPan({ x: 0, y: 0 });
                    }}
                    title="Resetuj powiększenie i pozycję"
                  >
                    ↺
                  </button>
                )}
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
          <div
            ref={explorerRef}
            className={`pdf-explorer-scroll ${isDragging ? "dragging" : ""}`}
            onMouseDown={(e) => {
              if (e.button !== 0 || !explorerRef.current) return;
              e.preventDefault();
              dragStateRef.current = {
                x: e.clientX,
                y: e.clientY,
                scrollLeft: explorerRef.current.scrollLeft,
                scrollTop: explorerRef.current.scrollTop,
                panX: pan.x,
                panY: pan.y,
                moved: false,
              };
              setIsDragging(true);
            }}
          >
            <div
              className="pdf-document-sheet"
              style={{
                width: `${zoom}%`,
                transform: `translate(${pan.x}px, ${pan.y}px)`,
              }}
            >
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
                    <img src={imgSrc} alt="Dokument PDF" className="pdf-page-img" draggable={false} />
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
                          title={`"${formatJsonKey(ev.field)}": ${formatFieldValue(ev.field, ev.value)}`}
                          onClick={() => {
                            if (suppressRectClickRef.current) return;
                            setSelectedIdx(evIdx);
                          }}
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

function JsonFormNode({
  keyName,
  value,
  onChange,
}: {
  keyName: string;
  value: unknown;
  onChange: (next: unknown) => void;
}) {
  if (Array.isArray(value)) {
    const template = value[0] ? emptyFromTemplate(value[0]) : { imie_nazwisko: null, nazwa: null, mail: null };
    return (
      <div className="editor-section">
        <div className="editor-section-head">
          <div>
            <strong>{friendlyLabel(keyName)}</strong>
            <span className="editor-key-tag">{keyName}</span>
          </div>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => onChange([...value, structuredClone(template)])}
          >
            + Dodaj pozycję
          </button>
        </div>
        {value.length === 0 ? (
          <div className="editor-empty">Brak wpisów — kliknij „+ Dodaj pozycję”, jeśli chcesz dodać.</div>
        ) : (
          <div className="editor-array-items">
            {value.map((entry, idx) => (
              <div key={idx} className="editor-card">
                <div className="editor-card-head">
                  <span>#{idx + 1}</span>
                  <button
                    type="button"
                    className="link editor-remove"
                    onClick={() => onChange(value.filter((_, i) => i !== idx))}
                  >
                    Usuń
                  </button>
                </div>
                {entry && typeof entry === "object" && !Array.isArray(entry) ? (
                  <div className="editor-fields">
                    {Object.entries(entry as Record<string, unknown>).map(([childKey, childVal]) => (
                      <JsonFormNode
                        key={childKey}
                        keyName={childKey}
                        value={childVal}
                        onChange={(nextChild) => {
                          const updated = [...value];
                          updated[idx] = { ...(entry as Record<string, unknown>), [childKey]: nextChild };
                          onChange(updated);
                        }}
                      />
                    ))}
                  </div>
                ) : (
                  <input
                    className="editor-input"
                    value={entry == null ? "" : String(entry)}
                    placeholder="Brak (null)"
                    onChange={(e) => {
                      const updated = [...value];
                      updated[idx] = e.target.value.trim() === "" ? null : e.target.value;
                      onChange(updated);
                    }}
                  />
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (value && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return (
      <div className="editor-section">
        <div className="editor-section-head">
          <div>
            <strong>{friendlyLabel(keyName)}</strong>
            <span className="editor-key-tag">{keyName}</span>
          </div>
        </div>
        <div className="editor-fields">
          {Object.entries(obj).map(([childKey, childVal]) => (
            <JsonFormNode
              key={childKey}
              keyName={childKey}
              value={childVal}
              onChange={(nextChild) => {
                const nextObj: Record<string, unknown> = { ...obj, [childKey]: nextChild };
                if (childKey === "czynnosc_powod_id" && typeof nextChild === "number" && "czynnosc_opis" in obj) {
                  nextObj.czynnosc_opis = CZYNNOSC_OPIS_BY_ID[nextChild] ?? obj.czynnosc_opis;
                }
                onChange(nextObj);
              }}
            />
          ))}
        </div>
      </div>
    );
  }

  if (keyName === "czynnosc_powod_id") {
    const numVal = typeof value === "number" ? value : Number(value) || 1;
    return (
      <label className="editor-row">
        <span className="editor-label">
          <span>{friendlyLabel(keyName)}</span>
          <span className="editor-key-tag">{keyName}</span>
        </span>
        <select
          className="editor-input"
          value={numVal}
          onChange={(e) => onChange(Number(e.target.value))}
        >
          {Object.entries(CZYNNOSC_OPIS_BY_ID).map(([id, desc]) => (
            <option key={id} value={id}>
              {id} — {desc}
            </option>
          ))}
        </select>
      </label>
    );
  }

  if (keyName === "przez-pelnomocnika") {
    const boolVal = value === 1 || value === true ? 1 : 0;
    return (
      <label className="editor-row">
        <span className="editor-label">
          <span>{friendlyLabel(keyName)}</span>
          <span className="editor-key-tag">{keyName}</span>
        </span>
        <select
          className="editor-input"
          value={boolVal}
          onChange={(e) => onChange(Number(e.target.value))}
        >
          <option value={1}>Tak (1)</option>
          <option value={0}>Nie (0)</option>
        </select>
      </label>
    );
  }

  if (keyName === "kwota") {
    const displayVal = typeof value === "number" ? value.toFixed(2) : value == null ? "" : String(value);
    return (
      <label className="editor-row">
        <span className="editor-label">
          <span>{friendlyLabel(keyName)}</span>
          <span className="editor-key-tag">{keyName}</span>
        </span>
        <input
          className="editor-input"
          type="number"
          step="0.01"
          value={displayVal}
          placeholder="0.00"
          onChange={(e) => {
            const raw = e.target.value.replace(",", ".").trim();
            if (!raw) {
              onChange(null);
              return;
            }
            const parsed = Number(raw);
            if (!Number.isNaN(parsed)) {
              onChange(Math.round(parsed * 100) / 100);
            }
          }}
        />
      </label>
    );
  }

  return (
    <label className="editor-row">
      <span className="editor-label">
        <span>{friendlyLabel(keyName)}</span>
        <span className="editor-key-tag">{keyName}</span>
      </span>
      {keyName === "tytul" ? (
        <textarea
          className="editor-input editor-textarea"
          rows={2}
          value={value == null ? "" : String(value)}
          placeholder="Brak (null)"
          onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
        />
      ) : (
        <input
          className="editor-input"
          type="text"
          value={value == null ? "" : String(value)}
          placeholder="Brak (null)"
          onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
        />
      )}
    </label>
  );
}

function FriendlyJsonEditor({
  data,
  originalData,
  onChange,
  onReset,
}: {
  data: Record<string, unknown>;
  originalData: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  onReset: () => void;
}) {
  const [rawMode, setRawMode] = useState(false);
  const [rawText, setRawText] = useState(() => formatUzpJson(data));
  const [rawError, setRawError] = useState<string | null>(null);
  const isModified = formatUzpJson(data) !== formatUzpJson(originalData);

  useEffect(() => {
    if (!rawMode) {
      setRawText(formatUzpJson(data));
      setRawError(null);
    }
  }, [data, rawMode]);

  const handleRawChange = (text: string) => {
    setRawText(text);
    try {
      const parsed = JSON.parse(text);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        setRawError("Główny JSON musi być obiektem { ... }.");
        return;
      }
      setRawError(null);
      onChange(parsed as Record<string, unknown>);
    } catch (err) {
      setRawError(`Błąd składni JSON: ${(err as Error).message}`);
    }
  };

  return (
    <div className="editor-shell">
      <div className="editor-banner">
        <div className="editor-banner-left">
          <span className={`badge ${isModified ? "badge-warn" : "badge-ok"}`}>
            {isModified ? "Zmodyfikowano ręcznie" : "Oryginał z AI"}
          </span>
          <span className="editor-hint">
            Popraw dowolne pole poniżej — przyciski „Kopiuj JSON” i „Pobierz JSON” zapiszą poprawioną wersję.
          </span>
        </div>
        <div className="editor-banner-actions">
          {isModified && (
            <button type="button" className="btn btn-sm" onClick={onReset}>
              Przywróć oryginał
            </button>
          )}
          <button type="button" className="btn btn-sm" onClick={() => setRawMode((v) => !v)}>
            {rawMode ? "Widok formularza" : "Zaawansowane: surowy JSON"}
          </button>
        </div>
      </div>

      {rawMode ? (
        <div className="editor-raw-wrap">
          {rawError && <div className="alert alert-error">{rawError}</div>}
          <textarea
            className="editor-raw"
            value={rawText}
            onChange={(e) => handleRawChange(e.target.value)}
            spellCheck={false}
          />
        </div>
      ) : Object.keys(data).length === 0 ? (
        <div className="empty">Brak pól do edycji (dokument sklasyfikowany jako spoza zakresu).</div>
      ) : (
        <div className="editor-root">
          {Object.entries(data).map(([topKey, topVal]) => (
            <JsonFormNode
              key={topKey}
              keyName={topKey}
              value={topVal}
              onChange={(nextVal) => onChange({ ...data, [topKey]: nextVal })}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ResultView({
  item,
  onUpdateData,
}: {
  item: QueueItem;
  onUpdateData: (nextData: Record<string, unknown>) => void;
}) {
  const [tab, setTab] = useState<"quotes" | "edit" | "json" | "meta">("quotes");
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
  const originalData = item.originalData ?? data;
  const json = formatUzpJson(data);
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

      {meta.ocr_error && (
        <div className="alert alert-warn">
          <strong>OCR niedostępny lokalnie:</strong> {meta.ocr_error}. Na serwerze Docker / Azure z paczką{" "}
          <code>tesseract-ocr-pol</code> skany są odczytywane automatycznie.
        </div>
      )}
      {isNegative && !meta.ocr_error && (
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
          <button className={tab === "edit" ? "tab active" : "tab"} onClick={() => setTab("edit")}>
            Edytuj pola
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
        <QuotesWithPdfView
          evidence={evidence}
          pages={pages}
          pageImages={pageImages}
          data={data}
          onUpdateData={onUpdateData}
        />
      ) : tab === "edit" ? (
        <FriendlyJsonEditor
          data={data}
          originalData={originalData}
          onChange={onUpdateData}
          onReset={() => onUpdateData(structuredClone(originalData))}
        />
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
        .then((result) => update(next.id, { status: "done", result, originalData: structuredClone(result.data) }))
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
              {items.some((i) => i.status === "done") && (
                <button
                  type="button"
                  className="btn btn-download-all"
                  onClick={() => downloadAllJsonZip(items.filter((i) => i.status === "done" && i.result))}
                >
                  Pobierz wszystkie JSON-y ({items.filter((i) => i.status === "done").length})
                </button>
              )}
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
              <ResultView
                key={selected.id}
                item={selected}
                onUpdateData={(nextData) => {
                  if (!selected.result) return;
                  update(selected.id, { result: { ...selected.result, data: nextData } });
                }}
              />
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
