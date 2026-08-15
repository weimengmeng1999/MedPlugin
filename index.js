import { fileURLToPath } from 'node:url'
import { spawn } from 'node:child_process'
import { readFile } from 'node:fs/promises'
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

/** Script locations, relative to skillsDir, organized by imaging modality. */
const SCRIPTS = {
  xrayMaira: 'xray/maira2_report.py',
  xrayAnatomy: 'xray/medgemma_anatomy_localization.py',
  xrayLongitudinal: 'xray/medgemma_longitudinal.py',
  xrayMedgemma: 'xray/medgemma_report.py',
  ctMedgemma: 'ct/medgemma_report.py',
  ctTotalseg: 'ct/totalseg_segmentation.py',
  mriTotalseg: 'mri/totalseg_segmentation.py',
}

/** Bytes of stdout/stderr retained for error diagnostics; older bytes are dropped. */
const MAX_CAPTURE_BYTES = 200_000

function appendCapped(buf, chunk) {
  const next = buf + chunk.toString('utf8')
  return next.length > MAX_CAPTURE_BYTES ? next.slice(next.length - MAX_CAPTURE_BYTES) : next
}

/**
 * Run one of the specialist python scripts under skills/ and parse its
 * stdout as JSON. Every script prints progress to stderr and exactly one
 * JSON object
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
function outputSchema(successStatus) {
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

function renderResult(value, formatSuccess) {
  if (value.status === 'error') return `Error: ${value.error}`
  return formatSuccess(value)
}

/** The image content block for `value.preview`, or none if no preview was attached. */
function previewBlocks(value) {
  return value.preview === undefined ? [] : [{ type: 'image', attachment: value.preview }]
}

function renderReport(value) {
  const text = renderResult(value, v => (typeof v.report_text === 'string' ? v.report_text : '(no report_text in output)'))
  return [{ type: 'text', text }, ...previewBlocks(value)]
}

function renderAnatomy(value) {
  const text = renderResult(value, (v) => {
    const boxes = Array.isArray(v.boxes) ? v.boxes : []
    if (boxes.length === 0) return 'No anatomical structures located.'
    return boxes.map(b => `${b.label}: box_2d=${JSON.stringify(b.box_2d)}`).join('\n')
  })
  return [{ type: 'text', text }, ...previewBlocks(value)]
}

function renderLongitudinal(value) {
  const text = renderResult(value, v => (typeof v.comparison_text === 'string' ? v.comparison_text : '(no comparison_text in output)'))
  return [{ type: 'text', text }, ...previewBlocks(value)]
}

function renderSegmentation(value) {
  const text = renderResult(value, (v) => {
    const structures = Array.isArray(v.structures_found) ? v.structures_found : []
    const lines = [`${v.n_structures ?? structures.length} structure(s) segmented, written to ${v.output_dir}. The segmentation masks are 3D NIfTI files, not directly viewable; the preview image (when attached) shows a rendered snapshot of where they landed, not the masks themselves.`]
    if (structures.length > 0) lines.push(structures.join(', '))
    return lines.join('\n')
  })
  return [{ type: 'text', text }, ...previewBlocks(value)]
}

/**
 * Soft-check whether the currently routed model declares image input.
 * Never throws: an unresolved provider/model/llm service is treated as not
 * capable, since a preview image is an enhancement, not the tool's primary
 * output.
 */
async function isImageCapableRoute(ctx, exec) {
  const routed = exec.agent?.session.requestHeader()?.config
  const provider = routed?.provider ?? exec.agent?.options.provider
  const model = routed?.model ?? exec.agent?.options.model
  const llm = ctx.get('llm')
  if (provider === undefined || model === undefined || llm === undefined) return false
  try {
    const active = await llm.resolveModelInfo(provider, model, exec.signal)
    return active.inputModalities !== undefined && active.inputModalities.includes('image')
  } catch {
    return false
  }
}

async function loadPreviewAttachment(attachments, path) {
  const data = await readFile(path)
  return attachments.saveImage({ data, mediaType: 'image/png', name: 'preview.png' })
}

/**
 * If `value.preview_image_path` names a file this process wrote and the
 * current route can carry an image, commit it through the attachment
 * service and attach the resulting reference as `value.preview`.
 * Best-effort: an unreadable temp file, a rejected save, no attachment
 * service, or a non-image-capable route all leave `value` unchanged rather
 * than failing a tool call whose primary result already succeeded.
 */
async function attachPreview(ctx, exec, value) {
  if (value.status !== 'success' || typeof value.preview_image_path !== 'string') return value
  const attachments = ctx.get('attachments')
  if (attachments === undefined) return value
  if (!(await isImageCapableRoute(ctx, exec))) return value
  const ref = await loadPreviewAttachment(attachments, value.preview_image_path).catch(() => undefined)
  if (ref !== undefined) {
    value.preview = { attachmentId: ref.attachmentId, mediaType: ref.mediaType, bytes: ref.bytes, width: ref.width, height: ref.height }
  }
  return value
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
      schema: outputSchema('success'),
      render: (_args, value) => renderReport(value),
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
      const value = await run(SCRIPTS.xrayMaira, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
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
      schema: outputSchema('success'),
      render: (_args, value) => renderAnatomy(value),
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', args.input, '--gpu', String(args.gpu ?? -1)]
      if (args.anatomy !== undefined && args.anatomy.length > 0) cliArgs.push('--anatomy', ...args.anatomy)
      const value = await run(SCRIPTS.xrayAnatomy, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
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
      schema: outputSchema('success'),
      render: (_args, value) => renderLongitudinal(value),
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', args.input, '--prior', args.prior, '--gpu', String(args.gpu ?? 2)]
      if (args.indication !== undefined) cliArgs.push('--indication', args.indication)
      const value = await run(SCRIPTS.xrayLongitudinal, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'xray_report_medgemma',
    description: 'Generate a chest X-ray radiology report with MedGemma 4B. Plain narrative findings text. Loads a multi-GB model onto a GPU, so a single call can take minutes.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the frontal chest X-ray (PNG, JPG, or DICOM .dcm).' },
      indication: { type: 'string', description: 'Clinical indication, e.g. "Shortness of breath."' },
      max_new_tokens: { type: 'integer', description: 'Max new tokens to generate. Default 512.' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 0.' },
    },
    output: {
      schema: outputSchema('success'),
      render: (_args, value) => renderReport(value),
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', args.input, '--gpu', String(args.gpu ?? 0)]
      if (args.indication !== undefined) cliArgs.push('--indication', args.indication)
      cliArgs.push('--max_new_tokens', String(args.max_new_tokens ?? 512))
      const value = await run(SCRIPTS.xrayMedgemma, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'ct_report_medgemma',
    description: 'Generate a CT radiology report candidate with MedGemma 4B. MedGemma is a 2D image-text model, so this converts the CT volume into a fixed axial-slice montage first — treat the result as a complementary candidate, not a substitute for a native 3D CT model. Loads a multi-GB model onto a GPU, so a single call can take minutes.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to a .nii/.nii.gz CT volume, or a directory containing one DICOM series.' },
      study_id: { type: 'string', description: 'Study identifier to echo back in the result (optional, for your own bookkeeping).' },
      indication: { type: 'string', description: 'Clinical indication, e.g. "Abdominal pain, rule out appendicitis."' },
      n_slices: { type: 'integer', description: 'Number of axial slices sampled into the montage. Default 16.' },
      max_new_tokens: { type: 'integer', description: 'Max new tokens to generate. Default 512.' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 0.' },
    },
    output: {
      schema: outputSchema('success'),
      render: (_args, value) => renderReport(value),
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', args.input, '--gpu', String(args.gpu ?? 0)]
      if (args.study_id !== undefined) cliArgs.push('--study_id', args.study_id)
      if (args.indication !== undefined) cliArgs.push('--indication', args.indication)
      cliArgs.push('--n_slices', String(args.n_slices ?? 16))
      cliArgs.push('--max_new_tokens', String(args.max_new_tokens ?? 512))
      const value = await run(SCRIPTS.ctMedgemma, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  const segmentationParams = {
    input: { type: 'string', required: true, description: 'Absolute path to a NIfTI volume (.nii/.nii.gz) or a directory containing one DICOM series.' },
    output: { type: 'string', description: 'Directory to write segmentation masks into (optional — defaults to a fresh temp directory).' },
    fast: { type: 'boolean', description: 'Use the faster, lower-resolution (3mm) model.' },
    ml: { type: 'boolean', description: 'Also write a single multilabel NIfTI file combining every structure.' },
    statistics: { type: 'boolean', description: 'Compute volume (mm3) and mean intensity per structure.' },
    preview: { type: 'boolean', description: 'Generate a PNG preview of the segmentation (default true; TotalSegmentator renders it, so it needs a headless-display-capable environment — set false if that fails).' },
    roi_subset: { type: 'array', items: { type: 'string' }, description: 'Only segment these specific structures, e.g. ["liver", "kidney_right"] (optional — omit to segment everything the task covers).' },
    gpu: { type: 'integer', description: 'GPU index (optional — auto-selected if omitted).' },
  }

  const segmentationCliArgs = (args) => {
    const cliArgs = ['--input', args.input]
    if (args.output !== undefined) cliArgs.push('--output', args.output)
    if (args.fast) cliArgs.push('--fast')
    if (args.ml) cliArgs.push('--ml')
    if (args.statistics) cliArgs.push('--statistics')
    if (args.preview ?? true) cliArgs.push('--preview')
    if (args.roi_subset !== undefined && args.roi_subset.length > 0) cliArgs.push('--roi_subset', ...args.roi_subset)
    if (args.gpu !== undefined) cliArgs.push('--gpu', String(args.gpu))
    return cliArgs
  }

  ctx.tools.register(defineTool({
    name: 'ct_segmentation_totalseg',
    description: 'Segment anatomical structures in a CT volume with TotalSegmentator. Writes one NIfTI mask file per structure to an output directory and returns their paths. Whole-body organ segmentation by default (task=total); set task=lung_vessels for pulmonary vessels and airways instead, or pass roi_subset to limit either task to specific structures. Can take several minutes on GPU, longer on CPU.',
    parameters: {
      ...segmentationParams,
      task: { type: 'string', enum: ['total', 'lung_vessels'], description: 'total (default) = whole-body organs. lung_vessels = pulmonary vessels and airways.' },
    },
    output: {
      schema: outputSchema('success'),
      render: (_args, value) => renderSegmentation(value),
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = segmentationCliArgs(args)
      if (args.task !== undefined) cliArgs.push('--task', args.task)
      const value = await run(SCRIPTS.ctTotalseg, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'mri_segmentation_totalseg',
    description: 'Segment anatomical structures in an MRI volume with TotalSegmentator (MR-specific model). Writes one NIfTI mask file per structure to an output directory and returns their paths. Can take several minutes on GPU, longer on CPU.',
    parameters: segmentationParams,
    output: {
      schema: outputSchema('success'),
      render: (_args, value) => renderSegmentation(value),
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const value = await run(SCRIPTS.mriTotalseg, segmentationCliArgs(args), exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))
}
