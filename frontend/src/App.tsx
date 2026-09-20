import {
  Activity,
  AlertTriangle,
  AudioLines,
  Check,
  CheckCircle2,
  CircleHelp,
  Clock3,
  FileAudio,
  Gauge,
  Info,
  Loader2,
  Mic,
  RefreshCw,
  Server,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Upload,
  Waves,
  X,
} from "lucide-react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

const MIN_CAPTURE_SECONDS = 4
const MAX_CAPTURE_SECONDS = 10

type Phase = "idle" | "capturing" | "analyzing" | "file" | "result" | "error"

type HealthData = {
  ok: boolean
  model?: string
  device?: string
  labels?: Record<string, string>
  model_error?: string | null
  live_capture_seconds?: number
  model_window_seconds?: number
  target_sample_rate?: number
}

type AnalysisResult = {
  ok: boolean
  state: "real" | "synthetic" | "uncertain" | "insufficient" | "error"
  status: string
  detail: string
  trust_score: number | null
  fake_probability: number | null
  real_probability?: number | null
  confidence: number | null
  windows: number
  total_windows: number
  duration_seconds?: number
  inference_seconds?: number
  error?: string
}

type AudioContextConstructor = typeof AudioContext

function getAudioContext(): AudioContextConstructor {
  const browserWindow = window as typeof window & {
    webkitAudioContext?: AudioContextConstructor
  }
  return window.AudioContext || browserWindow.webkitAudioContext!
}

function formatTime(seconds: number) {
  return `00:${seconds.toFixed(1).padStart(4, "0")}`
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

function MetricRow({
  icon: Icon,
  label,
  value,
  hint,
}: {
  icon: typeof Activity
  label: string
  value: string
  hint: string
}) {
  return (
    <div className="metric-row grid grid-cols-[auto_1fr_auto] items-center gap-3 py-2.5 sm:py-3">
      <div className="grid size-9 place-items-center rounded-lg border bg-muted/45 text-muted-foreground">
        <Icon className="size-4" />
      </div>
      <div className="min-w-0">
        <div className="flex items-center gap-1.5 text-sm font-medium">
          {label}
          <Tooltip>
            <TooltipTrigger asChild>
              <button className="text-muted-foreground transition-colors hover:text-foreground" aria-label={`About ${label}`}>
                <CircleHelp className="size-3.5" />
              </button>
            </TooltipTrigger>
            <TooltipContent className="max-w-64">{hint}</TooltipContent>
          </Tooltip>
        </div>
      </div>
      <span className="font-mono text-sm font-semibold tabular-nums">{value}</span>
    </div>
  )
}

function App() {
  const [health, setHealth] = useState<HealthData | null>(null)
  const [checkingHealth, setCheckingHealth] = useState(true)
  const [phase, setPhase] = useState<Phase>("idle")
  const [elapsed, setElapsed] = useState(0)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [fileStatus, setFileStatus] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [mobileView, setMobileView] = useState<"analyze" | "result">("analyze")

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const gainRef = useRef<GainNode | null>(null)
  const chunksRef = useRef<Float32Array[]>([])
  const sampleCountRef = useRef(0)
  const sampleRateRef = useRef(0)
  const runningRef = useRef(false)
  const capturingRef = useRef(false)
  const animationRef = useRef<number | null>(null)
  const finishCaptureRef = useRef<() => void>(() => undefined)

  const busy = phase === "capturing" || phase === "analyzing" || phase === "file"
  const canStop = phase === "capturing" && elapsed >= MIN_CAPTURE_SECONDS

  const checkHealth = useCallback(async (quiet = false) => {
    if (!quiet) setCheckingHealth(true)
    try {
      const response = await fetch("/health", { cache: "no-store" })
      const data = (await response.json()) as HealthData
      setHealth(data)
      return data
    } catch {
      const unavailable: HealthData = {
        ok: false,
        model_error: "The FastAPI backend could not be reached.",
      }
      setHealth(unavailable)
      return unavailable
    } finally {
      if (!quiet) setCheckingHealth(false)
    }
  }, [])

  useEffect(() => {
    void checkHealth()
  }, [checkHealth])

  const cleanupCapture = useCallback(async () => {
    if (processorRef.current) {
      processorRef.current.onaudioprocess = null
      try { processorRef.current.disconnect() } catch { /* already disconnected */ }
    }
    for (const node of [sourceRef.current, analyserRef.current, gainRef.current]) {
      if (node) {
        try { node.disconnect() } catch { /* already disconnected */ }
      }
    }
    streamRef.current?.getTracks().forEach((track) => track.stop())
    if (audioContextRef.current) {
      try { await audioContextRef.current.close() } catch { /* already closed */ }
    }
    processorRef.current = null
    sourceRef.current = null
    analyserRef.current = null
    gainRef.current = null
    streamRef.current = null
    audioContextRef.current = null
  }, [])

  const finishCapture = useCallback(async () => {
    if (!runningRef.current) return
    runningRef.current = false
    capturingRef.current = false
    if (processorRef.current) processorRef.current.onaudioprocess = null

    const sampleRate = sampleRateRef.current
    const joined = new Float32Array(sampleCountRef.current)
    let offset = 0
    for (const chunk of chunksRef.current) {
      joined.set(chunk, offset)
      offset += chunk.length
    }

    const audio = joined.slice(0, Math.ceil(sampleRate * MAX_CAPTURE_SECONDS))
    const duration = audio.length / sampleRate
    setElapsed(Math.min(duration, MAX_CAPTURE_SECONDS))

    if (duration < MIN_CAPTURE_SECONDS) {
      await cleanupCapture()
      setError("Please record at least four seconds before submitting.")
      setPhase("error")
      return
    }

    setError(null)
    setPhase("analyzing")
    try {
      const response = await fetch("/analyze-live", {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "X-Audio-Sample-Rate": String(sampleRate),
        },
        body: audio.buffer,
      })
      const data = (await response.json()) as AnalysisResult
      if (!response.ok) throw new Error(data.error || data.detail || "Live analysis failed.")
      setResult(data)
      setMobileView("result")
      setPhase("result")
    } catch (captureError) {
      setError(captureError instanceof Error ? captureError.message : "Live analysis failed.")
      setPhase("error")
    } finally {
      await cleanupCapture()
      chunksRef.current = []
      sampleCountRef.current = 0
    }
  }, [cleanupCapture])

  finishCaptureRef.current = () => { void finishCapture() }

  const beginCapture = useCallback(async () => {
    if (runningRef.current) return
    const currentHealth = await checkHealth(true)
    if (!currentHealth.ok) {
      setError(currentHealth.model_error || "The detector is not ready.")
      setPhase("error")
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      })
      const AudioContextClass = getAudioContext()
      const context = new AudioContextClass({ latencyHint: "interactive", sampleRate: 16_000 })
      await context.resume()

      const source = context.createMediaStreamSource(stream)
      const analyser = context.createAnalyser()
      analyser.fftSize = 512
      analyser.smoothingTimeConstant = 0.78
      const processor = context.createScriptProcessor(4096, 1, 1)
      const gain = context.createGain()
      gain.gain.value = 0
      source.connect(analyser)
      source.connect(processor)
      processor.connect(gain)
      gain.connect(context.destination)

      streamRef.current = stream
      audioContextRef.current = context
      sourceRef.current = source
      analyserRef.current = analyser
      processorRef.current = processor
      gainRef.current = gain
      sampleRateRef.current = context.sampleRate
      chunksRef.current = []
      sampleCountRef.current = 0
      runningRef.current = true
      capturingRef.current = true
      setElapsed(0)
      setError(null)
      setResult(null)
      setPhase("capturing")

      const maxSamples = Math.ceil(context.sampleRate * MAX_CAPTURE_SECONDS)
      processor.onaudioprocess = (event) => {
        if (!capturingRef.current) return
        const input = event.inputBuffer.getChannelData(0)
        const copy = new Float32Array(input.length)
        copy.set(input)
        chunksRef.current.push(copy)
        sampleCountRef.current += copy.length
        setElapsed(Math.min(sampleCountRef.current / context.sampleRate, MAX_CAPTURE_SECONDS))
        if (sampleCountRef.current >= maxSamples) {
          capturingRef.current = false
          finishCaptureRef.current()
        }
      }
    } catch (captureError) {
      await cleanupCapture()
      runningRef.current = false
      capturingRef.current = false
      setError(captureError instanceof Error ? captureError.message : "Microphone access failed.")
      setPhase("error")
    }
  }, [checkHealth, cleanupCapture])

  useEffect(() => () => {
    runningRef.current = false
    capturingRef.current = false
    if (animationRef.current) cancelAnimationFrame(animationRef.current)
    void cleanupCapture()
  }, [cleanupCapture])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const draw = () => {
      const bounds = canvas.getBoundingClientRect()
      const ratio = window.devicePixelRatio || 1
      const width = Math.max(1, Math.floor(bounds.width * ratio))
      const height = Math.max(1, Math.floor(bounds.height * ratio))
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width
        canvas.height = height
      }
      const context = canvas.getContext("2d")
      if (!context) return
      context.setTransform(ratio, 0, 0, ratio, 0, 0)
      context.clearRect(0, 0, bounds.width, bounds.height)

      const analyser = analyserRef.current
      if (phase === "capturing" && analyser) {
        const samples = new Uint8Array(analyser.fftSize)
        analyser.getByteTimeDomainData(samples)
        context.strokeStyle = "rgba(129, 140, 248, 0.95)"
        context.lineWidth = 2
        context.beginPath()
        samples.forEach((sample, index) => {
          const x = (index / (samples.length - 1)) * bounds.width
          const y = (sample / 255) * bounds.height
          if (index === 0) context.moveTo(x, y)
          else context.lineTo(x, y)
        })
        context.stroke()
      } else {
        const bars = Math.max(36, Math.floor(bounds.width / 10))
        const gap = bounds.width / bars
        context.strokeStyle = "rgba(148, 163, 184, 0.26)"
        context.lineWidth = 2
        context.lineCap = "round"
        for (let index = 0; index < bars; index += 1) {
          const centerDistance = Math.abs(index - (bars - 1) / 2) / (bars / 2)
          const barHeight = Math.max(4, (1 - centerDistance * 0.45) * (5 + Math.sin(index * 1.7) ** 2 * 12))
          const x = index * gap + gap / 2
          context.beginPath()
          context.moveTo(x, bounds.height / 2 - barHeight / 2)
          context.lineTo(x, bounds.height / 2 + barHeight / 2)
          context.stroke()
        }
      }
      if (phase === "capturing") animationRef.current = requestAnimationFrame(draw)
    }

    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(canvas)
    return () => {
      observer.disconnect()
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [phase])

  const chooseFile = useCallback((file?: File) => {
    if (!file) return
    const supported = file.type.startsWith("audio/") || /\.(wav|mp3|m4a|aac|ogg|oga|webm|flac|opus)$/i.test(file.name)
    if (!supported) {
      setFileStatus("Choose a supported audio file.")
      return
    }
    setSelectedFile(file)
    setFileStatus(null)
    setError(null)
  }, [])

  const decodeFile = useCallback(async (file: File) => {
    const AudioContextClass = getAudioContext()
    const context = new AudioContextClass()
    try {
      const raw = await file.arrayBuffer()
      const decoded = await context.decodeAudioData(raw.slice(0))
      if (decoded.duration < MIN_CAPTURE_SECONDS) {
        throw new Error("The audio file must be at least four seconds long.")
      }
      const mono = new Float32Array(decoded.length)
      for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
        const channelData = new Float32Array(decoded.length)
        decoded.copyFromChannel(channelData, channel)
        for (let index = 0; index < channelData.length; index += 1) {
          mono[index] += channelData[index] / decoded.numberOfChannels
        }
      }
      if (decoded.sampleRate === 16_000) return { samples: mono, duration: decoded.duration }
      const length = Math.ceil((mono.length * 16_000) / decoded.sampleRate)
      const offline = new OfflineAudioContext(1, length, 16_000)
      const buffer = offline.createBuffer(1, mono.length, decoded.sampleRate)
      buffer.copyToChannel(mono, 0)
      const source = offline.createBufferSource()
      source.buffer = buffer
      source.connect(offline.destination)
      source.start()
      const output = await offline.startRendering()
      return { samples: output.getChannelData(0).slice(), duration: output.duration }
    } finally {
      try { await context.close() } catch { /* already closed */ }
    }
  }, [])

  const analyzeFile = useCallback(async () => {
    if (!selectedFile || busy) return
    const currentHealth = await checkHealth(true)
    if (!currentHealth.ok) {
      setError(currentHealth.model_error || "The detector is not ready.")
      setPhase("error")
      return
    }
    setPhase("file")
    setError(null)
    setResult(null)
    setFileStatus("Decoding the complete recording…")
    try {
      const decoded = await decodeFile(selectedFile)
      setFileStatus(`Analyzing ${decoded.duration.toFixed(1)} seconds…`)
      const response = await fetch("/analyze-file", {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "X-Audio-Sample-Rate": "16000",
        },
        body: decoded.samples.buffer,
      })
      const data = (await response.json()) as AnalysisResult
      if (!response.ok) throw new Error(data.error || data.detail || "File analysis failed.")
      setResult(data)
      setMobileView("result")
      setFileStatus(`Complete · ${data.windows} of ${data.total_windows} windows analyzed`)
      setPhase("result")
    } catch (fileError) {
      const message = fileError instanceof Error ? fileError.message : "File analysis failed."
      setFileStatus(message)
      setError(message)
      setPhase("error")
    }
  }, [busy, checkHealth, decodeFile, selectedFile])

  const clearFile = () => {
    setSelectedFile(null)
    setFileStatus(null)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const resultTheme = useMemo(() => {
    if (!result) return {
      icon: AudioLines,
      color: "text-muted-foreground",
      surface: "bg-muted/35",
      label: "Waiting for a sample",
      description: "Record your voice or upload an audio file to begin.",
    }
    if (result.state === "real") return {
      icon: ShieldCheck,
      color: "text-emerald-400",
      surface: "bg-emerald-500/10",
      label: result.status,
      description: result.detail,
    }
    if (result.state === "synthetic") return {
      icon: ShieldAlert,
      color: "text-rose-400",
      surface: "bg-rose-500/10",
      label: result.status,
      description: result.detail,
    }
    return {
      icon: AlertTriangle,
      color: "text-amber-400",
      surface: "bg-amber-500/10",
      label: result.status,
      description: result.detail,
    }
  }, [result])

  const ResultIcon = resultTheme.icon
  const trustScore = result?.trust_score ?? 0
  const fakeProbability = result?.fake_probability ?? null
  const confidence = result?.confidence ?? null
  const progressClass = result?.state === "real"
    ? "[&_[data-slot=progress-indicator]]:bg-emerald-400"
    : result?.state === "synthetic"
      ? "[&_[data-slot=progress-indicator]]:bg-rose-400"
      : "[&_[data-slot=progress-indicator]]:bg-amber-400"

  return (
    <TooltipProvider delayDuration={250}>
      <div className="signal-grid flex h-dvh min-h-0 flex-col overflow-hidden">
        <header className="shrink-0 border-b bg-background/75 backdrop-blur-xl">
          <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:h-16 sm:px-6 lg:px-8">
            <div className="flex items-center gap-3">
              <div className="grid size-9 place-items-center rounded-xl bg-primary text-primary-foreground shadow-[0_0_28px_oklch(0.68_0.19_262/0.25)]">
                <Waves className="size-5" />
              </div>
              <div>
                <div className="text-sm font-bold tracking-[0.16em]">VOXGUARD</div>
                <div className="hidden text-[11px] text-muted-foreground sm:block">Synthetic voice intelligence</div>
              </div>
            </div>

            <Sheet>
              <SheetTrigger asChild>
                <Button variant="outline" size="sm" className="gap-2 bg-card/60">
                  <span className={cn("size-2 rounded-full", checkingHealth ? "animate-pulse bg-amber-400" : health?.ok ? "bg-emerald-400" : "bg-rose-400")} />
                  <span className="hidden sm:inline">{checkingHealth ? "Checking system" : health?.ok ? "System ready" : "System unavailable"}</span>
                  <span className="sm:hidden">Status</span>
                </Button>
              </SheetTrigger>
              <SheetContent className="border-border/80 bg-background/95 p-0 backdrop-blur-xl">
                <SheetHeader className="border-b p-6">
                  <SheetTitle>System status</SheetTitle>
                  <SheetDescription>Detector and analysis-path diagnostics.</SheetDescription>
                </SheetHeader>
                <div className="space-y-3 p-6">
                  <div className="flex items-start gap-3 rounded-xl border bg-card/60 p-4">
                    <Server className={cn("mt-0.5 size-5", health?.ok ? "text-emerald-400" : "text-rose-400")} />
                    <div className="min-w-0">
                      <div className="font-medium">Detector backend</div>
                      <div className="mt-1 break-words text-sm text-muted-foreground">
                        {health?.ok ? `Ready on ${health.device || "CPU"}` : health?.model_error || "Not available"}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 rounded-xl border bg-card/60 p-4">
                    <Activity className="mt-0.5 size-5 text-primary" />
                    <div>
                      <div className="font-medium">Analysis path</div>
                      <div className="mt-1 text-sm text-muted-foreground">One recording, one secure HTTP request.</div>
                    </div>
                  </div>
                  <div className="rounded-xl border bg-card/60 p-4 text-sm">
                    <div className="mb-3 text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">Technical details</div>
                    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-muted-foreground">
                      <dt>Model</dt><dd className="truncate text-right text-foreground">{health?.model || "—"}</dd>
                      <dt>Device</dt><dd className="text-right text-foreground">{health?.device || "—"}</dd>
                      <dt>Sample rate</dt><dd className="text-right text-foreground">{health?.target_sample_rate ? `${health.target_sample_rate.toLocaleString()} Hz` : "—"}</dd>
                      <dt>Window</dt><dd className="text-right text-foreground">{health?.model_window_seconds ? `${health.model_window_seconds}s` : "—"}</dd>
                    </dl>
                  </div>
                  <Button variant="secondary" className="w-full" onClick={() => void checkHealth()} disabled={checkingHealth}>
                    <RefreshCw className={cn("size-4", checkingHealth && "animate-spin")} />
                    Check again
                  </Button>
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </header>

        <main className="mx-auto flex min-h-0 w-full max-w-7xl flex-1 flex-col px-4 py-3 sm:px-6 sm:py-4 lg:px-8 lg:py-5">
          <section className="viewport-intro mb-3 shrink-0 sm:mb-4 lg:mb-5">
            <Badge variant="outline" className="eyebrow mb-2 hidden border-primary/25 bg-primary/8 text-primary sm:inline-flex">
              <Sparkles className="size-3" />
              AI voice verification
            </Badge>
            <h1 className="text-balance text-2xl font-semibold tracking-[-0.04em] sm:text-3xl lg:text-4xl">
              Verify a voice before you trust it.
            </h1>
            <p className="mt-1.5 max-w-2xl text-pretty text-xs leading-5 text-muted-foreground sm:text-sm">
              Record a short sample or upload audio. VoxGuard analyzes the complete signal and returns one clear, explainable result.
            </p>
          </section>

          {error && (
            <Alert variant="destructive" className="fixed top-16 right-4 left-4 z-40 border-rose-500/30 bg-background/95 shadow-2xl backdrop-blur-xl sm:left-auto sm:max-w-md">
              <AlertTriangle />
              <AlertTitle>Analysis needs attention</AlertTitle>
              <AlertDescription className="flex items-center justify-between gap-4">
                <span>{error}</span>
                <button aria-label="Dismiss error" onClick={() => setError(null)} className="shrink-0 rounded-md p-1 hover:bg-rose-500/10">
                  <X className="size-4" />
                </button>
              </AlertDescription>
            </Alert>
          )}

          <div className="mb-3 grid h-10 shrink-0 grid-cols-2 rounded-lg bg-muted/70 p-1 lg:hidden">
            <Button
              variant={mobileView === "analyze" ? "secondary" : "ghost"}
              size="sm"
              className="h-8"
              onClick={() => setMobileView("analyze")}
            >
              <Mic className="size-4" /> Analyze
            </Button>
            <Button
              variant={mobileView === "result" ? "secondary" : "ghost"}
              size="sm"
              className="h-8"
              onClick={() => setMobileView("result")}
            >
              <Shield className="size-4" /> Result
              {result && <span className={cn("size-1.5 rounded-full", result.state === "real" ? "bg-emerald-400" : result.state === "synthetic" ? "bg-rose-400" : "bg-amber-400")} />}
            </Button>
          </div>

          <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,1.08fr)_minmax(360px,0.92fr)]">
            <Card className={cn("h-full min-h-0 overflow-hidden border-border/80 bg-card/70 py-0 shadow-2xl shadow-black/10 backdrop-blur-sm lg:flex", mobileView === "analyze" ? "flex" : "hidden")}>
              <CardHeader className="shrink-0 border-b px-4 py-3 sm:px-5 sm:py-4">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <CardTitle className="text-lg">Analyze audio</CardTitle>
                    <CardDescription className="panel-description mt-1">Choose how you want to provide the voice sample.</CardDescription>
                  </div>
                  {busy && (
                    <Badge variant="secondary" className="gap-1.5">
                      <Loader2 className="size-3 animate-spin" />
                      {phase === "capturing" ? "Recording" : "Analyzing"}
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="min-h-0 flex-1 p-3 sm:p-4">
                <Tabs defaultValue="record" className="h-full min-h-0 gap-3">
                  <TabsList className="grid h-10 w-full shrink-0 grid-cols-2 bg-muted/70 p-1">
                    <TabsTrigger value="record" disabled={phase === "analyzing" || phase === "file"}>
                      <Mic className="size-4" /> Record
                    </TabsTrigger>
                    <TabsTrigger value="upload" disabled={phase === "capturing" || phase === "analyzing" || phase === "file"}>
                      <Upload className="size-4" /> Upload
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="record" className="min-h-0">
                    <div className="capture-surface flex h-full min-h-0 flex-col items-center justify-center rounded-xl border bg-background/35 px-4 py-4 text-center sm:px-6">
                      <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Voice capture</div>
                      <h2 className="text-lg font-semibold sm:text-xl">
                        {phase === "capturing" ? (canStop ? "Stop when you’re ready" : "Keep speaking naturally") : phase === "analyzing" ? "Analyzing your recording" : "Record up to 10 seconds"}
                      </h2>
                      <p className="capture-help mt-1 max-w-md text-xs text-muted-foreground sm:text-sm">
                        {phase === "capturing" ? "The stop control unlocks after the four-second model minimum." : "One capture stays in your browser until it is submitted for analysis."}
                      </p>

                      <button
                        type="button"
                        className="record-button record-ring mt-5 grid size-28 shrink-0 place-items-center rounded-full border border-primary/50 bg-gradient-to-b from-primary/20 to-primary/8 text-primary transition-transform hover:scale-[1.02] focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-ring/40 disabled:pointer-events-none disabled:opacity-55 sm:size-32 xl:size-36"
                        data-active={phase === "capturing"}
                        onClick={() => { if (phase !== "capturing") void beginCapture() }}
                        disabled={busy}
                        aria-label="Start voice recording"
                      >
                        {phase === "analyzing" ? <Loader2 className="size-10 animate-spin" /> : <Mic className="size-10" strokeWidth={1.6} />}
                      </button>

                      <div className="mt-5 font-mono text-2xl font-medium tracking-[-0.04em] tabular-nums">
                        {formatTime(elapsed)} <span className="text-muted-foreground">/ 00:10.0</span>
                      </div>
                      <Progress value={(elapsed / MAX_CAPTURE_SECONDS) * 100} className="mt-3 h-1.5 max-w-sm" />
                      <canvas ref={canvasRef} className="capture-waveform mt-2 h-12 w-full max-w-lg" aria-label="Audio waveform" />

                      <div className="mt-3 flex w-full max-w-sm flex-col gap-2 sm:flex-row">
                        {phase === "capturing" ? (
                          <Button className="h-11 flex-1" onClick={() => void finishCapture()} disabled={!canStop}>
                            <Check className="size-4" /> Stop & analyze
                          </Button>
                        ) : (
                          <Button className="h-11 flex-1" onClick={() => void beginCapture()} disabled={busy || health?.ok === false}>
                            <Mic className="size-4" /> Start recording
                          </Button>
                        )}
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="upload" className="min-h-0">
                    <div className="flex h-full min-h-0 flex-col rounded-xl border bg-background/35 p-3 sm:p-4">
                      <div
                        role="button"
                        tabIndex={0}
                        onClick={() => fileInputRef.current?.click()}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault()
                            fileInputRef.current?.click()
                          }
                        }}
                        onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
                        onDragLeave={() => setDragging(false)}
                        onDrop={(event) => {
                          event.preventDefault()
                          setDragging(false)
                          chooseFile(event.dataTransfer.files[0])
                        }}
                        className={cn(
                          "grid min-h-0 flex-1 place-items-center rounded-xl border border-dashed p-5 text-center outline-none transition-colors focus-visible:ring-4 focus-visible:ring-ring/30",
                          dragging ? "border-primary bg-primary/8" : "border-border bg-card/35 hover:border-primary/45 hover:bg-card/60",
                        )}
                      >
                        <div>
                          <div className="mx-auto grid size-14 place-items-center rounded-2xl border bg-muted/50 text-muted-foreground">
                            <FileAudio className="size-6" />
                          </div>
                          <div className="mt-3 font-semibold">{selectedFile ? selectedFile.name : "Drop audio here"}</div>
                          <div className="mt-2 text-sm text-muted-foreground">
                            {selectedFile ? `${selectedFile.type || "Audio file"} · ${formatBytes(selectedFile.size)}` : "or click to choose a file"}
                          </div>
                          <div className="mt-3 text-xs text-muted-foreground">MP3, WAV, M4A, FLAC and more · minimum 4 seconds</div>
                        </div>
                      </div>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="audio/*"
                        className="hidden"
                        onChange={(event) => chooseFile(event.target.files?.[0])}
                      />
                      {fileStatus && <div className="mt-3 rounded-lg border bg-muted/35 px-3 py-2 text-xs text-muted-foreground">{fileStatus}</div>}
                      <div className="mt-3 flex gap-3">
                        <Button className="h-11 flex-1" onClick={() => void analyzeFile()} disabled={!selectedFile || busy || health?.ok === false}>
                          {phase === "file" ? <Loader2 className="size-4 animate-spin" /> : <Activity className="size-4" />}
                          Analyze file
                        </Button>
                        {selectedFile && (
                          <Button variant="outline" className="h-11" onClick={clearFile} disabled={busy}>Clear</Button>
                        )}
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>

            <Card className={cn("h-full min-h-0 overflow-hidden border-border/80 bg-card/70 py-0 shadow-2xl shadow-black/10 backdrop-blur-sm lg:flex", mobileView === "result" ? "flex" : "hidden")}>
              <CardHeader className="shrink-0 border-b px-4 py-3 sm:px-5 sm:py-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <CardTitle className="text-lg">Analysis result</CardTitle>
                    <CardDescription className="panel-description mt-1">Combined evidence across the recording.</CardDescription>
                  </div>
                  {result && (
                    <Badge variant="outline" className={cn("border-current/20", resultTheme.color)}>
                      {result.state === "real" ? "Human" : result.state === "synthetic" ? "Synthetic" : "Review"}
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="min-h-0 flex-1 p-3 sm:p-4">
                <div className={cn("result-summary rounded-xl border p-4", resultTheme.surface)}>
                  <div className={cn("grid size-10 place-items-center rounded-lg border bg-background/45", resultTheme.color)}>
                    <ResultIcon className="size-5" />
                  </div>
                  <h2 className="mt-3 text-lg font-semibold">{resultTheme.label}</h2>
                  <p className="mt-1 line-clamp-2 text-xs leading-5 text-muted-foreground sm:text-sm">{resultTheme.description}</p>
                </div>

                <div className="trust-panel mt-3 rounded-xl border bg-background/35 p-4">
                  <div className="flex items-end justify-between gap-4">
                    <div>
                      <div className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">Trust score</div>
                      <div className="mt-1 text-3xl font-semibold tracking-[-0.05em] tabular-nums sm:text-4xl">
                        {result?.trust_score == null ? "—" : `${Math.round(trustScore)}%`}
                      </div>
                    </div>
                    <Shield className={cn("mb-1 size-8", resultTheme.color)} strokeWidth={1.5} />
                  </div>
                  <Progress value={result?.trust_score == null ? 0 : trustScore} className={cn("mt-3 h-2", progressClass)} />
                  <p className="trust-help mt-2 text-[11px] text-muted-foreground">Higher scores indicate stronger human-voice evidence.</p>
                </div>

                <div className="mt-1 divide-y">
                  <MetricRow icon={Sparkles} label="Synthetic probability" value={fakeProbability == null ? "—" : `${fakeProbability.toFixed(1)}%`} hint="Average synthetic evidence across usable audio windows." />
                  <MetricRow icon={Gauge} label="Confidence" value={confidence == null ? "—" : `${confidence.toFixed(1)}%`} hint="A stability signal combining certainty and agreement between windows." />
                  <MetricRow icon={Clock3} label="Analysis coverage" value={result ? `${result.windows}/${result.total_windows}` : "—"} hint="The number of usable windows compared with all windows in the recording." />
                </div>

                <Separator className="my-3" />
                <div className="result-disclaimer flex gap-2 text-[11px] leading-4 text-muted-foreground">
                  <Info className="mt-0.5 size-4 shrink-0" />
                  <p>Model evidence is a useful verification signal, not definitive proof of authenticity. Recording quality can affect results.</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </TooltipProvider>
  )
}

export default App
