"use client"

import { useEffect, useMemo, useRef, useState, type MouseEvent, type WheelEvent } from "react"
import { AlertTriangle, Download, History, Layers, Loader2, Maximize2, RefreshCcw, RotateCcw, ZoomIn, ZoomOut } from "lucide-react"

type MatchStatus = "matched" | "left_only" | "right_only" | string

interface MvpRow {
  tube_number: number
  length_in?: number
  length_in_display?: string
  length_ft_display?: string
  match_status?: MatchStatus
}

interface MvpMarker {
  tube_number: number
  tube_idx?: number
  match_status?: MatchStatus
  x?: number
  y?: number
  visual_x?: number
  visual_y?: number
  visual_source?: string
  calc_x?: number
  calc_y?: number
  ref_x?: number
  ref_y?: number
  display_color?: string
  label_fill?: string
  label_fill_opacity?: number
  label_text?: string
}

interface MvpImage {
  title?: string
  image_name?: string
  url?: string
  width?: number
  height?: number
  markers?: MvpMarker[]
  marker_count?: number
}

interface MvpState {
  generated_at?: string
  current_run?: {
    run_id?: string
    captured_at?: string
    is_latest?: boolean
  } | null
  is_latest_view?: boolean
  summary?: {
    pipe_count?: number
    matched?: number
    left_only?: number
    right_only?: number
    detection_source?: string
  }
  rows?: MvpRow[]
  images?: Record<string, MvpImage>
}

interface HistoryEntry {
  run_id: string
  artifact_name?: string
  run_url?: string
  captured_at?: string
  status?: string
  is_latest?: boolean
  can_open?: boolean
  summary?: {
    matched?: number
    left_only?: number
    right_only?: number
  }
}

interface HistoryState {
  entries?: HistoryEntry[]
  runs?: HistoryEntry[]
  latest_run_id?: string
}

interface PipeGroup {
  label: string
  face: string
  light: string
  dark: string
  cap: string
  textColor: string
}

interface Pipe {
  id: number
  group: "A" | "B" | "C" | "X"
  length: string
  widthPercent: number
  status: MatchStatus
}

const groups: Record<Pipe["group"], PipeGroup> = {
  A: {
    label: "A",
    face: "#f4f4f2",
    light: "#ffffff",
    dark: "#cdd0d4",
    cap: "#c4c8cc",
    textColor: "#3a3c3f",
  },
  B: {
    label: "B",
    face: "#e8cf7e",
    light: "#f3e2a8",
    dark: "#caa944",
    cap: "#bd9d3c",
    textColor: "#5a4715",
  },
  C: {
    label: "C",
    face: "#e2b4ae",
    light: "#f0d4cf",
    dark: "#c98a86",
    cap: "#bd7b76",
    textColor: "#5e3330",
  },
  X: {
    label: "X",
    face: "#2f343a",
    light: "#464d55",
    dark: "#171b20",
    cap: "#101318",
    textColor: "#f4f7fb",
  },
}

function groupForRow(row: MvpRow): Pipe["group"] {
  const length = Number(row.length_in)
  if (row.match_status && row.match_status !== "matched") return "X"
  if (!Number.isFinite(length)) return "X"
  if (length >= 493) return "A"
  if (length >= 491) return "B"
  return "C"
}

function formatDate(value?: string) {
  if (!value) return "No timestamp"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

function rowsToPipes(rows: MvpRow[] = []): Pipe[] {
  const finiteLengths = rows
    .map((row) => Number(row.length_in))
    .filter((value) => Number.isFinite(value))
  const maxLength = finiteLengths.length ? Math.max(...finiteLengths) : 1

  return rows
    .filter((row) => {
      const tubeNumber = Number(row.tube_number)
      return Number.isFinite(tubeNumber) && tubeNumber >= 1 && tubeNumber <= 10
    })
    .sort((a, b) => Number(a.tube_number) - Number(b.tube_number))
    .map((row) => {
      const length = Number(row.length_in)
      const widthPercent = Number.isFinite(length)
        ? Math.max(18, Math.min(100, (length / maxLength) * 100))
        : 42
      return {
        id: Number(row.tube_number),
        group: groupForRow(row),
        length: row.length_ft_display || row.length_in_display || "-",
        widthPercent,
        status: row.match_status || "",
      }
    })
}

function assetSrc(url?: string) {
  if (!url) return ""
  if (/^https?:\/\//i.test(url)) return url
  return `/api/mvp-asset?path=${encodeURIComponent(url)}`
}

function PipeRow({ pipe }: { pipe: Pipe }) {
  const group = groups[pipe.group]
  const isWarning = pipe.status && pipe.status !== "matched"

  return (
    <div className="flex items-center gap-3 rounded-md transition-opacity hover:opacity-90">
      <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-[#2a3138] text-lg font-medium text-[#f3f5f7]">
        {pipe.id}
      </div>

      <div className="flex min-h-[40px] flex-1 items-center">
        <div className="relative h-[34px]" style={{ width: `${pipe.widthPercent}%` }}>
          <div
            className="absolute inset-0 rounded-full"
            style={{
              background: `linear-gradient(to bottom, ${group.dark} 0%, ${group.face} 22%, ${group.light} 48%, ${group.face} 74%, ${group.dark} 100%)`,
            }}
          />
          <div
            className="absolute bottom-0 right-0 top-0 w-[13px] rounded-full"
            style={{
              background: `radial-gradient(ellipse at 60% 40%, ${group.light}, ${group.cap})`,
              border: `1px solid ${group.dark}`,
            }}
          />
          <div
            className="absolute bottom-0 left-0 top-0 w-[9px] rounded-full"
            style={{ background: group.dark }}
          />
          <div
            className="absolute inset-0 flex items-center justify-center text-sm font-medium"
            style={{ color: group.textColor }}
          >
            {pipe.length}
          </div>
        </div>
      </div>

      {isWarning ? (
        <span className="w-20 text-right text-xs font-medium uppercase text-[#9d5a1d]">
          {pipe.status}
        </span>
      ) : null}
    </div>
  )
}

function LegendItem({
  color,
  label,
  border,
}: {
  color: string
  label: string
  border?: string
}) {
  return (
    <span className="flex items-center gap-2 text-xs text-[#5f5e5a]">
      <span
        className="h-3 w-3 rounded-full"
        style={{
          background: color,
          border: border ? `1px solid ${border}` : undefined,
        }}
      />
      {label}
    </span>
  )
}

function StatusMessage({ error }: { error: string }) {
  return (
    <div className="flex items-start gap-3 bg-[#fff2df] px-5 py-4 text-sm text-[#6f3d10]">
      <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
      <div>
        <p className="font-semibold">MVP source is not available</p>
        <p className="mt-1 text-xs">{error}</p>
      </div>
    </div>
  )
}

function markerPoint(marker: MvpMarker) {
  const x = Number(marker.visual_x ?? marker.x ?? marker.calc_x)
  const y = Number(marker.visual_y ?? marker.y ?? marker.calc_y)
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null
  return { x, y }
}

function clampZoom(value: number) {
  return Math.max(1, Math.min(5, value))
}

function CameraPanel({ image, side }: { image?: MvpImage; side: "151" | "152" }) {
  const width = Number(image?.width || 0)
  const height = Number(image?.height || 0)
  const markers = Array.isArray(image?.markers) ? image.markers : []
  const title = side === "151" ? "RAW LEFT" : "RAW RIGHT"
  const [zoom, setZoom] = useState(1)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const dragRef = useRef({
    active: false,
    startX: 0,
    startY: 0,
    scrollLeft: 0,
    scrollTop: 0,
  })

  function updateZoom(nextZoom: number) {
    setZoom(clampZoom(nextZoom))
  }

  function handleWheel(event: WheelEvent<HTMLDivElement>) {
    event.preventDefault()
    updateZoom(zoom * (event.deltaY < 0 ? 1.12 : 1 / 1.12))
  }

  function handleMouseDown(event: MouseEvent<HTMLDivElement>) {
    if (event.button !== 0 && event.button !== 1) return
    const node = scrollRef.current
    if (!node) return
    event.preventDefault()
    dragRef.current = {
      active: true,
      startX: event.clientX,
      startY: event.clientY,
      scrollLeft: node.scrollLeft,
      scrollTop: node.scrollTop,
    }
  }

  function handleMouseMove(event: MouseEvent<HTMLDivElement>) {
    const node = scrollRef.current
    const drag = dragRef.current
    if (!node || !drag.active) return
    node.scrollLeft = drag.scrollLeft - (event.clientX - drag.startX)
    node.scrollTop = drag.scrollTop - (event.clientY - drag.startY)
  }

  function stopPan() {
    dragRef.current.active = false
  }

  return (
    <section className="overflow-hidden border border-[#323b44] bg-[#11161c]">
      <div className="flex items-center justify-between gap-3 bg-[#1f262c] px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold tracking-wide text-[#f3f5f7]">{title}</h2>
          <p className="mt-1 text-xs text-[#8a919a]">
            {image?.image_name || "No image"} | {markers.length} points
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-3 text-xs text-[#aab4be]">
            <span className="inline-flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-[#54d8ff]" />
              reference
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-[#ffd84e]" />
              pipe end
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button
              aria-label="Zoom out"
              className="flex h-8 w-8 items-center justify-center bg-[#11161c] text-[#c9d2db] disabled:opacity-35"
              disabled={zoom <= 1.01}
              onClick={() => updateZoom(zoom / 1.25)}
              title="Zoom out"
              type="button"
            >
              <ZoomOut className="h-4 w-4" />
            </button>
            <button
              aria-label="Reset zoom"
              className="flex h-8 min-w-12 items-center justify-center bg-[#11161c] px-2 text-xs font-semibold text-[#c9d2db]"
              onClick={() => updateZoom(1)}
              title="Reset zoom"
              type="button"
            >
              {Math.round(zoom * 100)}%
            </button>
            <button
              aria-label="Zoom in"
              className="flex h-8 w-8 items-center justify-center bg-[#11161c] text-[#c9d2db] disabled:opacity-35"
              disabled={zoom >= 4.99}
              onClick={() => updateZoom(zoom * 1.25)}
              title="Zoom in"
              type="button"
            >
              <ZoomIn className="h-4 w-4" />
            </button>
            <button
              aria-label="Fit image"
              className="flex h-8 w-8 items-center justify-center bg-[#11161c] text-[#c9d2db]"
              onClick={() => updateZoom(1)}
              title="Fit image"
              type="button"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {width > 0 && height > 0 && image?.url ? (
        <div
          className="relative max-h-[72vh] w-full cursor-grab overflow-auto bg-black active:cursor-grabbing"
          onMouseDown={handleMouseDown}
          onMouseLeave={stopPan}
          onMouseMove={handleMouseMove}
          onMouseUp={stopPan}
          onWheel={handleWheel}
          ref={scrollRef}
        >
          <div
            className="relative min-w-full"
            style={{
              aspectRatio: `${width} / ${height}`,
              width: `${zoom * 100}%`,
            }}
          >
            <img
              alt={title}
              className="absolute inset-0 h-full w-full object-fill"
              draggable={false}
              src={assetSrc(image.url)}
            />
            <svg
              className="absolute inset-0 h-full w-full"
              preserveAspectRatio="none"
              viewBox={`0 0 ${width} ${height}`}
            >
              {markers.map((marker) => {
                const calc = markerPoint(marker)
                const refX = Number(marker.ref_x)
                const refY = Number(marker.ref_y)
                const hasRef = Number.isFinite(refX) && Number.isFinite(refY)
                if (!calc) return null
                const labelX = Math.max(18, Math.min(width - 18, calc.x + 18))
                const labelY = Math.max(24, Math.min(height - 12, calc.y - 18))
                const fill = marker.label_fill || "#2a3138"
                const labelText = marker.label_text || "#ffffff"

                return (
                  <g key={`${side}-${marker.tube_number}`}>
                    {hasRef ? (
                      <line
                        stroke="#54d8ff"
                        strokeOpacity="0.38"
                        strokeWidth="2"
                        x1={refX}
                        x2={refX}
                        y1={refY}
                        y2={calc.y}
                      />
                    ) : null}
                    <circle
                      cx={calc.x}
                      cy={calc.y}
                      fill="#ffd84e"
                      r="9"
                      stroke="#302309"
                      strokeWidth="3"
                    />
                    <rect
                      fill={fill}
                      fillOpacity={marker.label_fill_opacity ?? 0.86}
                      height="28"
                      rx="4"
                      width={String(marker.tube_number > 9 ? 42 : 30)}
                      x={labelX}
                      y={labelY - 22}
                    />
                    <text
                      dominantBaseline="middle"
                      fill={labelText}
                      fontSize="20"
                      fontWeight="800"
                      textAnchor="middle"
                      x={labelX + (marker.tube_number > 9 ? 21 : 15)}
                      y={labelY - 8}
                    >
                      {marker.tube_number}
                    </text>
                  </g>
                )
              })}
            </svg>
          </div>
        </div>
      ) : (
        <div className="px-4 py-10 text-center text-sm text-[#8a919a]">
          No raw image available.
        </div>
      )}
    </section>
  )
}

export default function LODSortingTable() {
  const [state, setState] = useState<MvpState | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [activeRunId, setActiveRunId] = useState<string>("")
  const [activeArtifactName, setActiveArtifactName] = useState<string>("")
  const [activeTab, setActiveTab] = useState<"current" | "history" | "full">("current")
  const [error, setError] = useState("")
  const [actionMessage, setActionMessage] = useState("")
  const [captureRunning, setCaptureRunning] = useState(false)
  const [loading, setLoading] = useState(true)

  async function loadState(runId = activeRunId, artifactName = activeArtifactName) {
    setLoading(true)
    setError("")
    try {
      const params = new URLSearchParams()
      if (artifactName) {
        params.set("artifact", artifactName)
      } else if (runId) {
        params.set("run_id", runId)
      }
      const suffix = params.toString() ? `?${params}` : ""
      const response = await fetch(`/api/mvp-state${suffix}`, { cache: "no-store" })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.error || "Could not load the MVP state.")
      setState(payload)
      setActiveRunId(runId)
      setActiveArtifactName(artifactName)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the MVP state.")
    } finally {
      setLoading(false)
    }
  }

  async function loadHistory() {
    try {
      const response = await fetch("/api/mvp-history", { cache: "no-store" })
      const payload: HistoryState & { error?: string } = await response.json()
      if (!response.ok) throw new Error(payload.error || "Could not load history.")
      setHistory(payload.entries || payload.runs || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load history.")
    }
  }

  async function runCapture() {
    setCaptureRunning(true)
    setActionMessage("Downloading images and processing the pair...")
    setError("")
    try {
      const response = await fetch("/api/mvp-capture", {
        cache: "no-store",
        method: "POST",
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.error || "Could not complete the capture.")
      setActiveTab("current")
      await loadState("", "")
      await loadHistory()
      const matched = Number(payload.summary?.matched)
      const missing = Number(payload.summary?.left_only || 0) + Number(payload.summary?.right_only || 0)
      setActionMessage(
        Number.isFinite(matched)
          ? `Processed run ${payload.run_id || ""}: ${matched} matched, ${missing} missing sides.`
          : "Capture finished and the latest run is loaded.",
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not complete the capture.")
      setActionMessage("")
    } finally {
      setCaptureRunning(false)
    }
  }

  useEffect(() => {
    void loadState("")
    void loadHistory()
  }, [])

  const pipes = useMemo(() => rowsToPipes(state?.rows || []), [state])
  const summary = state?.summary || {}
  const currentRun = state?.current_run

  return (
    <div className="bg-[#1a1f24] p-5">
      <p className="mb-2 text-xs text-[#8a919a]">MVP 1 - live sorting table</p>

      <div className="overflow-hidden border border-[#c0c4c9]/50 bg-[#e4e6e9]">
        <div className="flex flex-wrap items-center justify-between gap-3 bg-[#2a3138] px-5 py-4">
          <div className="flex items-center gap-3">
            <Layers className="h-6 w-6 text-[#c9d2db]" />
            <div>
              <p className="text-lg font-medium tracking-wide text-[#f3f5f7]">
                LOD Sorting Table
              </p>
              <p className="text-xs text-[#aab4be]">
                {activeRunId ? `Historical run ${activeRunId}` : "Latest run"} |{" "}
                {formatDate(currentRun?.captured_at || state?.generated_at)}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              className="flex items-center gap-2 bg-[#1f262c] px-3 py-2 text-xs font-medium text-[#c9d2db] disabled:cursor-wait disabled:opacity-60"
              disabled={captureRunning}
              onClick={() => {
                void runCapture()
              }}
              type="button"
            >
              {captureRunning ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              Download and Process
            </button>
            <button
              className={`flex items-center gap-2 px-3 py-2 text-xs font-medium ${
                activeTab === "current"
                  ? "bg-[#c9d2db] text-[#1f262c]"
                  : "bg-[#1f262c] text-[#c9d2db]"
              }`}
              onClick={() => {
                setActiveTab("current")
              }}
              type="button"
            >
              <Layers className="h-4 w-4" />
              Current
            </button>
            <button
              className={`flex items-center gap-2 px-3 py-2 text-xs font-medium ${
                activeTab === "full"
                  ? "bg-[#c9d2db] text-[#1f262c]"
                  : "bg-[#1f262c] text-[#c9d2db]"
              }`}
              onClick={() => {
                setActiveTab("full")
              }}
              type="button"
            >
              <Maximize2 className="h-4 w-4" />
              Full View
            </button>
            <button
              className={`flex items-center gap-2 px-3 py-2 text-xs font-medium ${
                activeTab === "history"
                  ? "bg-[#c9d2db] text-[#1f262c]"
                  : "bg-[#1f262c] text-[#c9d2db]"
              }`}
              onClick={() => {
                setActiveTab("history")
                void loadHistory()
              }}
              type="button"
            >
              <History className="h-4 w-4" />
              History
            </button>
            <button
              className="flex items-center gap-2 bg-[#1f262c] px-3 py-2 text-xs font-medium text-[#c9d2db]"
              onClick={() => {
                void loadState(activeRunId, activeArtifactName)
                void loadHistory()
              }}
              type="button"
            >
              <RefreshCcw className="h-4 w-4" />
              Refresh
            </button>
            <div className="flex items-center gap-2 bg-[#1f262c] px-3 py-2">
              <span className="h-2 w-2 rounded-full bg-[#5dcaa5]" />
              <span className="text-xs text-[#c9d2db]">
                {activeTab === "full" ? "2 raw views" : `${pipes.length} pipes`}
              </span>
            </div>
          </div>
        </div>

        {error ? <StatusMessage error={error} /> : null}
        {actionMessage ? (
          <div className="bg-[#edf7f3] px-5 py-3 text-xs font-medium text-[#225f4c]">
            {actionMessage}
          </div>
        ) : null}

        {activeTab === "full" ? (
          <div className="grid gap-4 bg-[#0f1318] p-4 xl:grid-cols-2">
            <CameraPanel image={state?.images?.["151"]} side="151" />
            <CameraPanel image={state?.images?.["152"]} side="152" />
          </div>
        ) : activeTab === "history" ? (
          <div className="max-h-[560px] overflow-auto px-5 py-4">
            <div className="flex flex-col gap-2">
              {history.length ? (
                history.map((entry) => (
                  <button
                    className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 bg-[#f2f3f5] px-4 py-3 text-left hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
                    disabled={!entry.can_open}
                    key={entry.run_id}
                    onClick={() => {
                      setActiveTab("current")
                      void loadState(entry.run_id, entry.artifact_name || "")
                    }}
                    type="button"
                  >
                    <span>
                      <span className="block text-sm font-semibold text-[#2a3138]">
                        {entry.run_id}
                        {entry.is_latest ? " - latest" : ""}
                      </span>
                      <span className="mt-1 block text-xs text-[#68717a]">
                        {formatDate(entry.captured_at)} | {entry.status || "unknown"}
                      </span>
                    </span>
                    <span className="text-right text-xs font-medium text-[#5f5e5a]">
                      {entry.summary?.matched || 0} matched
                      <br />
                      {(entry.summary?.left_only || 0) + (entry.summary?.right_only || 0)} missing sides
                    </span>
                  </button>
                ))
              ) : (
                <div className="bg-[#f2f3f5] px-4 py-8 text-center text-sm text-[#68717a]">
                  No historical runs found.
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="max-h-[560px] overflow-auto px-5 py-4">
            {loading ? (
              <div className="bg-[#f2f3f5] px-4 py-8 text-center text-sm text-[#68717a]">
                Loading live MVP data...
              </div>
            ) : pipes.length ? (
              <div className="flex flex-col gap-2">
                {pipes.map((pipe) => (
                  <PipeRow key={pipe.id} pipe={pipe} />
                ))}
              </div>
            ) : (
              <div className="bg-[#f2f3f5] px-4 py-8 text-center text-sm text-[#68717a]">
                No pipes available.
              </div>
            )}
          </div>
        )}

        {activeTab !== "full" ? (
          <div className="flex flex-wrap items-center gap-5 border-t border-[#c0c4c9]/50 bg-[#d4d7db] px-5 py-3">
            <span className="text-xs font-medium text-[#5f5e5a]">Group:</span>
            <LegendItem color="#f4f4f2" label="A - long" border="#c4c8cc" />
            <LegendItem color="#e8cf7e" label="B - medium" />
            <LegendItem color="#d98b85" label="C - short" />
            <LegendItem color="#2f343a" label="X - missing side" />
            <span className="ml-auto text-xs text-[#5f5e5a]">
              {summary.matched || 0} matched | {summary.left_only || 0} left only |{" "}
              {summary.right_only || 0} right only
            </span>
          </div>
        ) : null}
      </div>
    </div>
  )
}
