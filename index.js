import { fileURLToPath } from 'node:url'
import { spawn } from 'node:child_process'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'xray-report-generation'
export const inject = ['tools']

/**
 * @typedef {object} Config
 * @property {string} [skillsDir] Directory containing this package's
 *   skills/ subfolders. Defaults to the skills/ directory shipped inside
 *   this package — override only to point at a different checkout of these
 *   scripts.
 * @property {string} [pythonBin] Python interpreter to invoke each script
 *   with (each script re-execs itself into its own isolated venv on first
 *   run). Default "python3".
 * @property {number} [timeoutMs] Kill the model process if it hasn't
 *   finished after this many ms (model load + GPU inference can be slow).
 *   Default 30 minutes.
 */

const DEFAULT_TIMEOUT_MS = 30 * 60 * 1000

/** Directory this module lives in, so the shipped skills/ scripts resolve regardless of the caller's cwd. */
const PACKAGE_DIR = fileURLToPath(new URL('.', import.meta.url))

/** Script locations, relative to skillsDir. */
const SCRIPTS = {
  maira: 'xray_grounding/run_xray_grounding_maira.py',
  anatomy: 'medgemma_multimodal/run_xray_anatomy_localization.py',
  longitudinal: 'medgemma_multimodal/run_xray_medgemma_longitudinal.py',
}

/** Bytes of stdout/stderr retained for error diagnostics; older bytes are dropped. */
const MAX_CAPTURE_BYTES = 200_000

function appendCapped(buf, chunk) {
  const next = buf + chunk.toString('utf8')
  return next.length > MAX_CAPTURE_BYTES ? next.slice(next.length - MAX_CAPTURE_BYTES) : next
}

/**
 * Run one of the xray specialist python scripts and parse its stdout as
 * JSON. Every script prints progress to stderr and exactly one JSON object
 * to stdout (both success and script-reported error use this same
 * contract), so a killed process or stdout that never parses as JSON is
 * treated as an infrastructure failure (thrown); a well-formed
 * `{"status":"error",...}` object is returned as the canonical value, same
 * as the script's own contract.
 *
 * @param {{ pythonBin: string, scriptPath: string, args: string[], cwd: string, timeoutMs: number, signal: AbortSignal }} opts
 * @returns {Promise<unknown>}
 */
function runPythonScript(opts) {
  return new Promise((settle, reject) => {
    const child = spawn(opts.pythonBin, [opts.scriptPath, ...opts.args], { cwd: opts.cwd })
    let stdout = ''
    let stderr = ''
    let killedReason

    const timer = setTimeout(() => {
      killedReason = `timed out after ${opts.timeoutMs}ms`
      child.kill('SIGTERM')
    }, opts.timeoutMs)
    const onAbort = () => {
      killedReason = 'tool call aborted'
      child.kill('SIGTERM')
    }
    opts.signal.addEventListener('abort', onAbort, { once: true })
    const cleanup = () => {
      clearTimeout(timer)
      opts.signal.removeEventListener('abort', onAbort)
    }

    child.stdout.on('data', (chunk) => { stdout = appendCapped(stdout, chunk) })
    child.stderr.on('data', (chunk) => { stderr = appendCapped(stderr, chunk) })
    child.on('error', (error) => {
      cleanup()
      reject(error)
    })
    child.on('close', (code) => {
      cleanup()
      if (killedReason !== undefined) {
        reject(new Error(`${opts.scriptPath}: ${killedReason}${stderr ? `\n--- stderr tail ---\n${stderr.slice(-2000)}` : ''}`))
        return
      }
      // Most scripts print single-line JSON, but try the whole trimmed
      // stdout first regardless, falling back to just the last line for a
      // script whose stdout carries a trailing non-JSON line ahead of a
      // vendored library's own stdout chatter.
      const text = stdout.trim()
      const lastLine = text.split('\n').at(-1) ?? ''
      try {
        settle(JSON.parse(text))
        return
      } catch {
        // fall through to the last-line attempt below
      }
      try {
        settle(JSON.parse(lastLine))
        return
      } catch {
        // fall through to the diagnostic rejection below
      }
      reject(new Error(
        `${opts.scriptPath}: exited ${code} with non-JSON stdout`
        + `\n--- stdout tail ---\n${stdout.slice(-2000) || '(empty)'}`
        + `${stderr ? `\n--- stderr tail ---\n${stderr.slice(-2000)}` : ''}`,
      ))
    })
  })
}

/** Shared success/error output shape: every script's JSON has one of these two shapes on its final stdout line. */
function xrayOutputSchema(successStatus) {
  return {
    oneOf: [
      {
        type: 'object',
        additionalProperties: true,
        properties: {
          status: { type: 'string', const: successStatus, required: true },
        },
      },
      {
        type: 'object',
        additionalProperties: true,
        properties: {
          status: { type: 'string', const: 'error', required: true },
          error: { type: 'string', required: true },
        },
      },
    ],
  }
}

function renderXray(value, formatSuccess) {
  if (value.status === 'error') return `Error: ${value.error}`
  return formatSuccess(value)
}

function renderReport(value) {
  return renderXray(value, v => (typeof v.report_text === 'string' ? v.report_text : '(no report_text in output)'))
}

function renderAnatomy(value) {
  return renderXray(value, (v) => {
    const boxes = Array.isArray(v.boxes) ? v.boxes : []
    if (boxes.length === 0) return 'No anatomical structures located.'
    return boxes.map(b => `${b.label}: box_2d=${JSON.stringify(b.box_2d)}`).join('\n')
  })
}

function renderLongitudinal(value) {
  return renderXray(value, v => (typeof v.comparison_text === 'string' ? v.comparison_text : '(no comparison_text in output)'))
}

/**
 * @param {import('@deepseek-ai/cordis').Context} ctx
 * @param {Config} [config]
 */
export function apply(ctx, config = {}) {
  const skillsDir = config.skillsDir ?? `${PACKAGE_DIR}skills`
  const pythonBin = config.pythonBin ?? 'python3'
  const timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS

  const run = (relScript, args, signal) => {
    const scriptPath = `${skillsDir}/${relScript}`
    const cwd = scriptPath.slice(0, scriptPath.lastIndexOf('/'))
    return runPythonScript({ pythonBin, scriptPath, args, cwd, timeoutMs, signal })
  }

  ctx.tools.register(defineTool({
    name: 'xray_report_maira',
    description: 'Generate a chest X-ray radiology report with MAIRA-2. Supports plain report, grounded report (findings with bounding boxes), or locating a single phrase on the image. Loads a multi-GB model onto a GPU, so a single call can take minutes; the first call also downloads ~14GB of gated weights (requires HF_TOKEN).',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the frontal chest X-ray (PNG, JPG, or DICOM .dcm).' },
      lateral: { type: 'string', description: 'Absolute path to a lateral view from the same study (optional but recommended).' },
      prior: { type: 'string', description: 'Absolute path to a prior frontal X-ray (optional).' },
      prior_report: { type: 'string', description: 'Prior radiology report text (optional, used with prior).' },
      indication: { type: 'string', description: 'Clinical indication, e.g. "Dyspnea."' },
      technique: { type: 'string', description: 'Technique, e.g. "PA and lateral views."' },
      comparison: { type: 'string', description: 'Comparison, e.g. "None." or "Compared to 01/01/2024."' },
      mode: {
        type: 'string',
        enum: ['report', 'grounded_report', 'phrase_grounding'],
        description: 'report = plain narrative findings text. grounded_report (default) = findings with bounding boxes. phrase_grounding = locate a phrase on the image (requires phrase).',
      },
      phrase: { type: 'string', description: 'Phrase to locate on the image. Required when mode is "phrase_grounding".' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 2.' },
    },
    output: {
      schema: xrayOutputSchema('success'),
      render: (_args, value) => [{ type: 'text', text: renderReport(value) }],
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const mode = args.mode ?? 'grounded_report'
      if (mode === 'phrase_grounding' && args.phrase === undefined) {
        throw new Error('phrase is required when mode is "phrase_grounding"')
      }
      const cliArgs = ['--input', args.input]
      if (args.lateral !== undefined) cliArgs.push('--lateral', args.lateral)
      if (args.prior !== undefined) cliArgs.push('--prior', args.prior)
      if (args.prior_report !== undefined) cliArgs.push('--prior_report', args.prior_report)
      if (args.indication !== undefined) cliArgs.push('--indication', args.indication)
      if (args.technique !== undefined) cliArgs.push('--technique', args.technique)
      if (args.comparison !== undefined) cliArgs.push('--comparison', args.comparison)
      cliArgs.push('--mode', mode)
      if (args.phrase !== undefined) cliArgs.push('--phrase', args.phrase)
      cliArgs.push('--gpu', String(args.gpu ?? 2))
      return run(SCRIPTS.maira, cliArgs, exec.signal)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'xray_anatomy_localization',
    description: 'Localize anatomical structures in a chest X-ray (MedGemma 1.5-4b-it) as labeled bounding boxes only — no pathology assertions, no findings. Only call this when precise spatial localization is itself diagnostically relevant: verifying device/hardware position relative to an anatomic landmark, localizing an unusual mass/lesion/foreign body, or the user explicitly asks where something is. One of the slower tools (~2 minutes) — skip for routine screening.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the chest X-ray (PNG, JPG, or DICOM .dcm).' },
      anatomy: { type: 'array', items: { type: 'string' }, description: 'Specific anatomical structure names to localize (optional — omit to let the model choose).' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default -1 (CPU).' },
    },
    output: {
      schema: xrayOutputSchema('success'),
      render: (_args, value) => [{ type: 'text', text: renderAnatomy(value) }],
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', args.input, '--gpu', String(args.gpu ?? -1)]
      if (args.anatomy !== undefined && args.anatomy.length > 0) cliArgs.push('--anatomy', ...args.anatomy)
      return run(SCRIPTS.anatomy, cliArgs, exec.signal)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'xray_longitudinal_comparison',
    description: 'Compare a prior (earlier) chest X-ray to the current one for interval change using MedGemma 1.5\'s longitudinal comparison (Improved/Stable/Worsened across consolidation, edema, pleural effusion, pneumonia, pneumothorax). A genuine two-image visual comparison — only call this when the user provided or referenced an actual prior/earlier image, not for a single-image read.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the current (most recent) frontal chest X-ray.' },
      prior: { type: 'string', required: true, description: 'Absolute path to the prior (earlier) frontal chest X-ray for the SAME patient.' },
      indication: { type: 'string', description: 'Clinical indication, e.g. "Follow-up for pneumonia."' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 2.' },
    },
    output: {
      schema: xrayOutputSchema('success'),
      render: (_args, value) => [{ type: 'text', text: renderLongitudinal(value) }],
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', args.input, '--prior', args.prior, '--gpu', String(args.gpu ?? 2)]
      if (args.indication !== undefined) cliArgs.push('--indication', args.indication)
      return run(SCRIPTS.longitudinal, cliArgs, exec.signal)
    },
  }))
}
