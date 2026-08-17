import { fileURLToPath } from 'node:url'
import { spawn } from 'node:child_process'
import { readFile, mkdir, copyFile } from 'node:fs/promises'
import { basename, join } from 'node:path'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { resolveImageInput } from './lib/attachment-input.js'
import { applyVisionRoutes } from './lib/vision-route.js'

export const name = 'medplugin'
export const inject = ['tools', 'llm']

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
 * @property {boolean} [wrapProviders] Register an image-capable twin route
 *   per live text provider ("<provider>-medplugin", e.g.
 *   "deepseek-official-medplugin") so pasted images are admitted on text-only
 *   routes and reach the model as attachment-id markers — select the twin in
 *   the model picker. Default true.
 * @property {string[]} [excludedProviders] Provider ids never to wrap
 *   (optional; routes ending in "-medplugin" or "-vision" are always
 *   excluded automatically).
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
  mriMedgemma: 'mri/medgemma_report.py',
  mriTotalseg: 'mri/totalseg_segmentation.py',
  xrayBiomedparse: 'xray/biomedparse_segmentation.py',
  ultrasoundBiomedparse: 'ultrasound/biomedparse_segmentation.py',
  ultrasoundBiomedclip: 'ultrasound/biomedclip_classify.py',
  retinalBiomedparse: 'retinal/biomedparse_segmentation.py',
  retinalMedgemma: 'retinal/medgemma_report.py',
  ctBiomedparse: 'ct/biomedparse_segmentation.py',
  mriBiomedparse: 'mri/biomedparse_segmentation.py',
}

/** Bytes of stdout/stderr retained for error diagnostics; older bytes are dropped. */
const MAX_CAPTURE_BYTES = 200_000

/**
 * Shared tail for image-input parameter descriptions: the argument may be a
 * filesystem path OR the durable attachment id (e.g. "sha256:...") of an
 * image pasted into the conversation — the same "path or pasted image"
 * contract the dsh-vision-router pixel tools use.
 */
const IMAGE_INPUT_NOTE = ' May also be the attachment id (e.g. "sha256:...") of an image pasted in this conversation instead of a path.'

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

/** Image content blocks for `value.previews`, or none if no preview was attached. */
function previewBlocks(value) {
  return Array.isArray(value.previews) ? value.previews.map(p => ({ type: 'image', attachment: p })) : []
}

/**
 * Append a note explaining how to see generated previews. When the previews
 * were attached inline (`value.previews` non-empty) the chat already shows
 * them as clickable image thumbnails, so we just point the user at that
 * (click a preview to enlarge it) instead of redundantly dumping the plain
 * file paths. When no preview could be attached inline (e.g. the route is
 * text-only) we list the workspace paths as the fallback way to open them,
 * and explain why images aren't shown inline.
 */
function withPreviewNote(text, value) {
  const parts = [text]
  const postedInline = Array.isArray(value.previews) && value.previews.length > 0
  if (!postedInline && value.preview_skipped_reason !== undefined) {
    const paths = value.preview_image_path
      ?? (Array.isArray(value.preview_image_paths) ? value.preview_image_paths.join(', ') : undefined)
    parts.push(`(A preview image was generated at ${paths} but not attached here: ${value.preview_skipped_reason}.)`)
  }
  if (value.preview_note !== undefined) parts.push(`(${value.preview_note})`)
  if (postedInline) {
    parts.push('Click a preview image to enlarge it.')
  } else if (Array.isArray(value.preview_file_links) && value.preview_file_links.length > 0) {
    parts.push('Preview image file(s):\n' + value.preview_file_links.map(p => `- \`${p}\``).join('\n'))
  }
  return parts.join('\n\n')
}

function renderReport(value) {
  const text = renderResult(value, (v) => {
    const report = typeof v.report_text === 'string' ? v.report_text : '(no report_text in output)'
    return typeof v.preprocessing_note === 'string' ? `${report}\n\n(${v.preprocessing_note})` : report
  })
  return [{ type: 'text', text: withPreviewNote(text, value) }, ...previewBlocks(value)]
}

function renderClassification(value) {
  const text = renderResult(value, (v) => {
    if (typeof v.prediction !== 'string') return '(no prediction in output)'
    const probs = v.probabilities && typeof v.probabilities === 'object' ? v.probabilities : {}
    const ranked = Object.entries(probs).sort((a, b) => b[1] - a[1])
    const lines = [`Prediction: ${v.prediction} (confidence ${v.confidence})`]
    if (ranked.length > 0) lines.push(ranked.map(([label, p]) => `${label}: ${p}`).join(', '))
    return lines.join('\n')
  })
  return [{ type: 'text', text: withPreviewNote(text, value) }, ...previewBlocks(value)]
}

function renderAnatomy(value) {
  const text = renderResult(value, (v) => {
    const boxes = Array.isArray(v.boxes) ? v.boxes : []
    if (boxes.length === 0) return 'No anatomical structures located.'
    return boxes.map(b => `${b.label}: box_2d=${JSON.stringify(b.box_2d)}`).join('\n')
  })
  return [{ type: 'text', text: withPreviewNote(text, value) }, ...previewBlocks(value)]
}

function renderLongitudinal(value) {
  const text = renderResult(value, v => (typeof v.comparison_text === 'string' ? v.comparison_text : '(no comparison_text in output)'))
  return [{ type: 'text', text: withPreviewNote(text, value) }, ...previewBlocks(value)]
}

function renderSegmentation(value) {
  const text = renderResult(value, (v) => {
    const structures = Array.isArray(v.structures_found) ? v.structures_found : []
    const lines = [`${v.n_structures ?? structures.length} structure(s) segmented, written to ${v.output_dir}. The segmentation masks are 3D NIfTI files, not directly viewable; the preview image (when attached) shows a rendered snapshot of where they landed, not the masks themselves.`]
    if (structures.length > 0) lines.push(structures.join(', '))
    return lines.join('\n')
  })
  return [{ type: 'text', text: withPreviewNote(text, value) }, ...previewBlocks(value)]
}

function renderBiomedparse2d(value) {
  const text = renderResult(value, (v) => {
    const outputs = Array.isArray(v.outputs) ? v.outputs : []
    if (outputs.length === 0) return 'No prompts segmented.'
    return outputs.map(o => `${o.prompt}: ${o.coverage_pct}% of image (score ${o.score})`).join('\n')
  })
  return [{ type: 'text', text: withPreviewNote(text, value) }, ...previewBlocks(value)]
}

function renderBiomedparse3d(value) {
  const text = renderResult(value, (v) => {
    const slices = Array.isArray(v.slices) ? v.slices : []
    const lines = [`${v.n_slices_processed ?? slices.length} of ${v.n_slices_total} slice(s) processed.`]
    // Summarize per prompt rather than per slice — all_slices mode can process hundreds of slices, and a line each would flood the conversation.
    const byPrompt = new Map()
    for (const slice of slices) {
      for (const p of (Array.isArray(slice.prompts) ? slice.prompts : [])) {
        const stat = byPrompt.get(p.prompt) ?? { maxCoverage: 0, nonzeroSlices: 0 }
        if (p.coverage > 0) stat.nonzeroSlices += 1
        stat.maxCoverage = Math.max(stat.maxCoverage, p.coverage)
        byPrompt.set(p.prompt, stat)
      }
    }
    for (const [prompt, stat] of byPrompt) {
      lines.push(`${prompt}: found on ${stat.nonzeroSlices} of ${slices.length} processed slice(s), max coverage ${(stat.maxCoverage * 100).toFixed(1)}%`)
    }
    const masks = v.nifti_masks !== undefined && typeof v.nifti_masks === 'object' ? Object.entries(v.nifti_masks) : []
    if (masks.length > 0) lines.push('3D NIfTI masks: ' + masks.map(([prompt, path]) => `${prompt} -> ${path}`).join(', '))
    return lines.join('\n')
  })
  return [{ type: 'text', text: withPreviewNote(text, value) }, ...previewBlocks(value)]
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
 * Copy the generated preview PNGs into a `previews/` folder in the session's
 * workspace and return their absolute paths, so the tool's text output can
 * name a clickable file. This mirrors how dsh-vision-router surfaces its
 * artifacts (a path string the user can open) — the web UI cannot render
 * plugin-supplied images inline on a text-only route, so a real file path is
 * the dependable way to actually see the overlay.
 * @returns {{ dir: string, links: string[] }} or null when the workspace is
 *   unavailable or no preview files exist.
 */
async function savePreviewsToWorkspace(exec, paths) {
  const session = exec && exec.agent && exec.agent.session
  const cwd = session && session.header && session.header.cwd
  if (typeof cwd !== 'string' || cwd === '') return null
  const outDir = join(cwd, 'medplugin', 'previews')
  await mkdir(outDir, { recursive: true })
  const links = []
  for (const p of paths) {
    const target = join(outDir, basename(p))
    await copyFile(p, target).catch(() => {})
    links.push(target)
  }
  return { dir: outDir, links: links.filter(() => true) }
}

/** Cap on how many preview images one tool call attaches — BiomedParse can generate one overlay per prompt (or per prompt per slice in --all_slices mode), far more than useful to show inline. */
const MAX_ATTACHED_PREVIEWS = 4

/**
 * If `value.preview_image_path` (single) or `value.preview_image_paths`
 * (array) names file(s) this process wrote and the current route can carry
 * an image, commit up to MAX_ATTACHED_PREVIEWS of them through the
 * attachment service and attach the resulting references as
 * `value.previews`. Best-effort: an unreadable temp file or a rejected save
 * is dropped rather than failing a tool call whose primary result already
 * succeeded. A known reason for attaching none (no attachment service, or a
 * route that declared it can't carry images — e.g. DeepSeek's own
 * chat-completions models, which are text-only at the wire level) is
 * recorded as `value.preview_skipped_reason`; attaching fewer than were
 * generated is recorded as `value.preview_note` — so the tool's rendered
 * text can say why, instead of images just silently not appearing.
 */
async function attachPreview(ctx, exec, value) {
  const paths = typeof value.preview_image_path === 'string'
    ? [value.preview_image_path]
    : Array.isArray(value.preview_image_paths) ? value.preview_image_paths : undefined
  if (value.status !== 'success' || paths === undefined || paths.length === 0) return value

  // The dependable way to actually see the overlay: copy the previews into the
  // session workspace and expose their paths, so the tool's text output can
  // name a clickable file — regardless of route modality. The web UI cannot
  // render plugin-supplied images inline on a text-only route.
  const saved = await savePreviewsToWorkspace(exec, paths).catch(() => null)
  if (saved !== null && saved.links.length > 0) value.preview_file_links = saved.links

  const attachments = ctx.get('attachments')
  if (attachments === undefined) {
    value.preview_skipped_reason = value.preview_file_links ? undefined : 'no attachment service is mounted in this profile'
    return value
  }
  if (!(await isImageCapableRoute(ctx, exec))) {
    value.preview_skipped_reason = value.preview_file_links ? undefined : 'the current model route does not declare image input'
    return value
  }

  const candidates = paths.slice(0, MAX_ATTACHED_PREVIEWS)
  const refs = []
  for (const path of candidates) {
    const ref = await loadPreviewAttachment(attachments, path).catch(() => undefined)
    if (ref !== undefined) refs.push({ attachmentId: ref.attachmentId, mediaType: ref.mediaType, bytes: ref.bytes, width: ref.width, height: ref.height })
  }
  if (refs.length === 0) {
    value.preview_skipped_reason = 'the preview file(s) could not be read or committed'
    return value
  }
  value.previews = refs
  if (paths.length > refs.length) {
    value.preview_note = `Showing ${refs.length} of ${paths.length} generated preview images.`
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
      input: { type: 'string', required: true, description: 'Absolute path to the frontal chest X-ray (PNG, JPG, or DICOM .dcm).' + IMAGE_INPUT_NOTE },
      lateral: { type: 'string', description: 'Absolute path to a lateral view from the same study (optional but recommended).' + IMAGE_INPUT_NOTE },
      prior: { type: 'string', description: 'Absolute path to a prior frontal X-ray (optional).' + IMAGE_INPUT_NOTE },
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
      const input = await resolveImageInput(ctx, exec, args.input)
      const cliArgs = ['--input', input]
      if (args.lateral !== undefined) cliArgs.push('--lateral', await resolveImageInput(ctx, exec, args.lateral))
      if (args.prior !== undefined) cliArgs.push('--prior', await resolveImageInput(ctx, exec, args.prior))
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
      input: { type: 'string', required: true, description: 'Absolute path to the chest X-ray (PNG, JPG, or DICOM .dcm).' + IMAGE_INPUT_NOTE },
      anatomy: { type: 'array', items: { type: 'string' }, description: 'Specific anatomical structure names to localize (optional — omit to let the model choose).' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default -1 (CPU).' },
    },
    output: {
      schema: outputSchema('success'),
      render: (_args, value) => renderAnatomy(value),
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', await resolveImageInput(ctx, exec, args.input), '--gpu', String(args.gpu ?? -1)]
      if (args.anatomy !== undefined && args.anatomy.length > 0) cliArgs.push('--anatomy', ...args.anatomy)
      const value = await run(SCRIPTS.xrayAnatomy, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'xray_longitudinal_comparison',
    description: 'Compare a prior (earlier) chest X-ray to the current one for interval change using MedGemma 1.5\'s longitudinal comparison (Improved/Stable/Worsened across consolidation, edema, pleural effusion, pneumonia, pneumothorax). A genuine two-image visual comparison — only call this when the user provided or referenced an actual prior/earlier image, not for a single-image read.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the current (most recent) frontal chest X-ray.' + IMAGE_INPUT_NOTE },
      prior: { type: 'string', required: true, description: 'Absolute path to the prior (earlier) frontal chest X-ray for the SAME patient.' + IMAGE_INPUT_NOTE },
      indication: { type: 'string', description: 'Clinical indication, e.g. "Follow-up for pneumonia."' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 2.' },
    },
    output: {
      schema: outputSchema('success'),
      render: (_args, value) => renderLongitudinal(value),
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const input = await resolveImageInput(ctx, exec, args.input)
      const prior = await resolveImageInput(ctx, exec, args.prior)
      const cliArgs = ['--input', input, '--prior', prior, '--gpu', String(args.gpu ?? 2)]
      if (args.indication !== undefined) cliArgs.push('--indication', args.indication)
      const value = await run(SCRIPTS.xrayLongitudinal, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'xray_report_medgemma',
    description: 'Generate a chest X-ray radiology report with MedGemma 4B. Plain narrative findings text. Loads a multi-GB model onto a GPU, so a single call can take minutes.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the frontal chest X-ray (PNG, JPG, or DICOM .dcm).' + IMAGE_INPUT_NOTE },
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
      const cliArgs = ['--input', await resolveImageInput(ctx, exec, args.input), '--gpu', String(args.gpu ?? 0)]
      if (args.indication !== undefined) cliArgs.push('--indication', args.indication)
      cliArgs.push('--max_new_tokens', String(args.max_new_tokens ?? 512))
      const value = await run(SCRIPTS.xrayMedgemma, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'ct_report_medgemma',
    description: 'Generate a CT radiology report candidate with MedGemma 1.5 4B. MedGemma 1.5 is trained to interpret a 3D CT volume as a sequence of per-slice images, each windowed into wide/soft-tissue/brain Hounsfield-unit channels (Google\'s own reference technique) — not a flattened montage. Treat the result as a complementary candidate, not a substitute for a native 3D CT model. Loads a multi-GB model onto a GPU; more slices means more inference time, so a single call can take several minutes.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to a .nii/.nii.gz CT volume, or a directory containing one DICOM series.' },
      study_id: { type: 'string', description: 'Study identifier to echo back in the result (optional, for your own bookkeeping).' },
      indication: { type: 'string', description: 'Clinical indication, e.g. "Abdominal pain, rule out appendicitis."' },
      n_slices: { type: 'integer', description: 'Number of axial slices uniformly sampled and sent to the model, one per SLICE block. Default 32 (Google\'s own notebook demo uses up to 85 — more slices costs proportionally more inference time).' },
      max_new_tokens: { type: 'integer', description: 'Max new tokens to generate. Default 1024.' },
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
      cliArgs.push('--n_slices', String(args.n_slices ?? 32))
      cliArgs.push('--max_new_tokens', String(args.max_new_tokens ?? 1024))
      const value = await run(SCRIPTS.ctMedgemma, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'mri_report_medgemma',
    description: 'Generate an MRI radiology report candidate with MedGemma 1.5 4B, using the same per-slice-sequence technique as ct_report_medgemma. UNOFFICIAL: Google only publishes this per-slice-sequence technique for CT (fixed Hounsfield-unit windows); no equivalent MRI reference exists, so each slice here is instead normalized by its own 0.5-99.5 percentile intensity range (matching BiomedParse\'s own MRI normalization convention) — treat this as a more speculative candidate than the CT tool, not a substitute for a native 3D MRI model. Loads a multi-GB model onto a GPU; more slices means more inference time, so a single call can take several minutes.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to a .nii/.nii.gz MRI volume, or a directory containing one DICOM series.' },
      study_id: { type: 'string', description: 'Study identifier to echo back in the result (optional, for your own bookkeeping).' },
      indication: { type: 'string', description: 'Clinical indication, e.g. "Headache, rule out mass lesion."' },
      n_slices: { type: 'integer', description: 'Number of slices uniformly sampled and sent to the model, one per SLICE block. Default 32 — more slices costs proportionally more inference time.' },
      max_new_tokens: { type: 'integer', description: 'Max new tokens to generate. Default 1024.' },
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
      cliArgs.push('--n_slices', String(args.n_slices ?? 32))
      cliArgs.push('--max_new_tokens', String(args.max_new_tokens ?? 1024))
      const value = await run(SCRIPTS.mriMedgemma, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'retinal_report_medgemma',
    description: 'Generate a retinal (fundus) ophthalmology report with MedGemma 4B. Plain narrative findings text — MedGemma\'s image encoder is pre-trained on fundus images alongside chest X-ray, dermatology, and histopathology. Loads a multi-GB model onto a GPU, so a single call can take minutes.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the retinal (fundus) photograph (PNG or JPG).' + IMAGE_INPUT_NOTE },
      indication: { type: 'string', description: 'Clinical indication, e.g. "Diabetic retinopathy screening."' },
      max_new_tokens: { type: 'integer', description: 'Max new tokens to generate. Default 512.' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 0.' },
    },
    output: {
      schema: outputSchema('success'),
      render: (_args, value) => renderReport(value),
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', await resolveImageInput(ctx, exec, args.input), '--gpu', String(args.gpu ?? 0)]
      if (args.indication !== undefined) cliArgs.push('--indication', args.indication)
      cliArgs.push('--max_new_tokens', String(args.max_new_tokens ?? 512))
      const value = await run(SCRIPTS.retinalMedgemma, cliArgs, exec.signal)
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

  const BIOMEDPARSE_SETUP_NOTE = 'The first BiomedParse call on this machine clones the model repo, installs ~15 extra Python packages plus a from-source detectron2 build (one-time, ~1-2 minutes), and downloads ~1.5GB of ungated weights (no token needed) — later calls skip straight to inference.'

  const biomedparse2dParams = {
    input: { type: 'string', required: true, description: 'Absolute path to the image (PNG or JPG — unlike the MAIRA-2/MedGemma tools, BiomedParse does not accept DICOM .dcm).' + IMAGE_INPUT_NOTE },
    prompts: { type: 'array', items: { type: 'string' }, required: true, description: 'Findings or structures to segment, e.g. ["consolidation", "pleural effusion"]. One overlay is generated per prompt.' },
    output_dir: { type: 'string', description: 'Directory to write overlay/mask images into (optional — defaults to a fresh temp directory).' },
    gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 0.' },
  }

  const biomedparse2dCliArgs = args => {
    const cliArgs = ['--input', args.input, '--gpu', String(args.gpu ?? 0), '--prompts', ...args.prompts]
    if (args.output_dir !== undefined) cliArgs.push('--output_dir', args.output_dir)
    return cliArgs
  }

  ctx.tools.register(defineTool({
    name: 'xray_segmentation_biomedparse',
    description: `Text-prompted segmentation of findings/structures in a chest X-ray with BiomedParse. ${BIOMEDPARSE_SETUP_NOTE}`,
    parameters: biomedparse2dParams,
    output: { schema: outputSchema('success'), render: (_args, value) => renderBiomedparse2d(value) },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const input = await resolveImageInput(ctx, exec, args.input)
      const value = await run(SCRIPTS.xrayBiomedparse, biomedparse2dCliArgs({ ...args, input }), exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'ultrasound_segmentation_biomedparse',
    description: `Text-prompted segmentation of findings/structures in an ultrasound image with BiomedParse. ${BIOMEDPARSE_SETUP_NOTE}`,
    parameters: biomedparse2dParams,
    output: { schema: outputSchema('success'), render: (_args, value) => renderBiomedparse2d(value) },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const input = await resolveImageInput(ctx, exec, args.input)
      const value = await run(SCRIPTS.ultrasoundBiomedparse, biomedparse2dCliArgs({ ...args, input }), exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'ultrasound_classify_biomedclip',
    description: 'Zero-shot classification of an ultrasound image with BiomedCLIP (~200M params) — picks the best-matching label from a list, it does not invent new categories. Useful to disambiguate a vague segmentation request before calling ultrasound_segmentation_biomedparse (e.g. classify anatomy/pathology first, then pass the winning label as a segmentation prompt) — a close spread across all candidate probabilities means none of them fit well. Built-in panels: anatomy (what kind of scan is this), breast, thyroid, cardiac, general (lesion/cyst/calcification/fluid/vascular); or free-form via task="cls" with custom labels.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the ultrasound image (PNG or JPG).' + IMAGE_INPUT_NOTE },
      task: { type: 'string', enum: ['anatomy', 'breast', 'thyroid', 'cardiac', 'general', 'cls'], required: true, description: 'Built-in label panel, or "cls" for a free-form list via labels.' },
      labels: { type: 'array', items: { type: 'string' }, description: 'Custom candidate labels, e.g. ["normal kidney", "kidney cyst", "kidney stone"]. Required when task is "cls"; ignored otherwise.' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 0.' },
    },
    output: {
      schema: outputSchema('success'),
      render: (_args, value) => renderClassification(value),
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      if (args.task === 'cls' && (args.labels === undefined || args.labels.length === 0)) {
        throw new Error('labels is required when task is "cls"')
      }
      const cliArgs = ['--input', await resolveImageInput(ctx, exec, args.input), '--task', args.task, '--gpu', String(args.gpu ?? 0)]
      if (args.labels !== undefined && args.labels.length > 0) cliArgs.push('--labels', args.labels.join(','))
      const value = await run(SCRIPTS.ultrasoundBiomedclip, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'retinal_segmentation_biomedparse',
    description: `Text-prompted segmentation of findings/structures in a retinal (fundus) image with BiomedParse. ${BIOMEDPARSE_SETUP_NOTE}`,
    parameters: biomedparse2dParams,
    output: { schema: outputSchema('success'), render: (_args, value) => renderBiomedparse2d(value) },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const input = await resolveImageInput(ctx, exec, args.input)
      const value = await run(SCRIPTS.retinalBiomedparse, biomedparse2dCliArgs({ ...args, input }), exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  const biomedparse3dParams = {
    input: { type: 'string', required: true, description: 'Absolute path to a NIfTI volume (.nii or .nii.gz). Unlike the _totalseg tools, BiomedParse does not accept a DICOM directory — convert to NIfTI first, or use the _totalseg tool for this modality.' },
    prompts: { type: 'array', items: { type: 'string' }, required: true, description: 'Findings or structures to segment, e.g. ["liver", "kidney"].' },
    slice_idx: { type: 'integer', description: 'Slice index along the depth axis (optional — defaults to the middle slice).' },
    all_slices: { type: 'boolean', description: 'Process every slice and reconstruct a 3D NIfTI mask per prompt, instead of just one slice. One model call per prompt per slice — slow and produces one overlay PNG per prompt per slice; only set this when a full-volume mask is actually needed.' },
    threshold: { type: 'number', description: 'Mask binarization threshold. Default 0.5.' },
    output_dir: { type: 'string', description: 'Directory to write overlay/mask images into (optional — defaults to a fresh temp directory).' },
    gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 0.' },
  }

  const biomedparse3dCliArgs = args => {
    const cliArgs = ['--input', args.input, '--gpu', String(args.gpu ?? 0)]
    if (args.slice_idx !== undefined) cliArgs.push('--slice_idx', String(args.slice_idx))
    if (args.all_slices) cliArgs.push('--all_slices')
    if (args.threshold !== undefined) cliArgs.push('--threshold', String(args.threshold))
    if (args.output_dir !== undefined) cliArgs.push('--output_dir', args.output_dir)
    cliArgs.push('--prompts', ...args.prompts)
    return cliArgs
  }

  ctx.tools.register(defineTool({
    name: 'ct_segmentation_biomedparse',
    description: `Text-prompted segmentation of findings/structures in a CT volume with BiomedParse — complementary to ct_segmentation_totalseg's fixed anatomical-structure list, since BiomedParse takes any free-text prompt (pathology or anatomy). ${BIOMEDPARSE_SETUP_NOTE}`,
    parameters: {
      ...biomedparse3dParams,
      site: { type: 'string', required: true, enum: ['abdomen', 'lung', 'pelvis', 'liver', 'colon', 'pancreas'], description: 'Anatomical site, used for CT-specific windowing.' },
    },
    output: { schema: outputSchema('success'), render: (_args, value) => renderBiomedparse3d(value) },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = [...biomedparse3dCliArgs(args), '--site', args.site]
      const value = await run(SCRIPTS.ctBiomedparse, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'mri_segmentation_biomedparse',
    description: `Text-prompted segmentation of findings/structures in an MRI volume with BiomedParse — complementary to mri_segmentation_totalseg's fixed anatomical-structure list, since BiomedParse takes any free-text prompt (pathology or anatomy). ${BIOMEDPARSE_SETUP_NOTE}`,
    parameters: {
      ...biomedparse3dParams,
      channel_idx: { type: 'integer', description: 'Channel index for a multi-channel MRI volume (e.g. BRATS-style, 0-3). Optional.' },
    },
    output: { schema: outputSchema('success'), render: (_args, value) => renderBiomedparse3d(value) },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = [...biomedparse3dCliArgs(args)]
      if (args.channel_idx !== undefined) cliArgs.push('--channel_idx', String(args.channel_idx))
      const value = await run(SCRIPTS.mriBiomedparse, cliArgs, exec.signal)
      return attachPreview(ctx, exec, value)
    },
  }))

  // Image-capable twin routes: without these, DSH rejects a pasted image on a
  // text-only route before any tool can see it. Each twin declares image input
  // and rewrites image blocks into attachment-id markers for the text model.
  applyVisionRoutes(ctx, config)
}
