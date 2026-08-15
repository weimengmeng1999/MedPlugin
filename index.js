import { existsSync } from 'node:fs'
import { spawn } from 'node:child_process'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'xray-report-generation'
export const inject = ['tools']

/**
 * @typedef {object} Config
 * @property {string} [baseDir] Root skills_scripts/ directory each entry in
 *   SCRIPTS is relative to (a MedOmni checkout's skills_scripts/ directory).
 *   Falls back to the MEDPLUGIN_SKILLS_DIR environment variable; the plugin
 *   refuses to load if neither is set or the directory doesn't exist.
 * @property {string} [pythonBin] Python interpreter to invoke each script
 *   with (each script re-execs itself into its own isolated venv on first
 *   run). Default "python3".
 * @property {number} [timeoutMs] Kill the model process if it hasn't
 *   finished after this many ms (model load + GPU inference can be slow).
 *   Default 30 minutes.
 */

const DEFAULT_TIMEOUT_MS = 30 * 60 * 1000

/**
 * Script locations, mirroring the BASE_DIR + *_SCRIPT constants in
 * MedOmni's medomni/med_teams/xray_team.py so this plugin calls the exact
 * same scripts xray_team.py's build_xray_specialist_tools() does.
 *
 * xray_team.py also exposes detection_tool (CarinaNet) and bone_tool
 * (SigLIP2) — both intentionally left out here. Unlike every script below,
 * their run_xray_detection.py / run_xray_bone_classification.py have no
 * isolated-venv bootstrap of their own: they import bare `torch` /
 * `transformers` against whatever the ambient Python environment happens to
 * have, which can silently conflict with another tool's pinned versions
 * (e.g. MAIRA-2 requires transformers>=4.48,<4.52). Every script kept here
 * re-execs itself into its own `.venv*` on first run, so tools never fight
 * over dependency versions.
 */
const SCRIPTS = {
  chexagent: 'xray_report_generation/run_xray_grounding_chexagent.py',
  llavaRad: 'xray_report_generation/run_xray_llava_rad.py',
  maira: 'xray_grounding/run_xray_grounding_maira.py',
  classification: 'xray-classification/run_xray_classification.py',
  anatomy: 'medgemma_multimodal/run_xray_anatomy_localization.py',
  longitudinal: 'medgemma_multimodal/run_xray_medgemma_longitudinal.py',
  radzero: 'xray_grounding_radzero/run_radzero.py',
  caseRetrieval: 'xray_case_retrieval/run_case_retrieval.py',
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
      // Most scripts print single-line JSON; chexagent and case_retrieval's
      // success paths use `json.dumps(..., indent=2)` and pretty-print
      // across many lines. Try the whole trimmed stdout first, falling back
      // to just the last line for a script whose stdout carries a trailing
      // non-JSON line ahead of a vendored library's own stdout chatter.
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

function renderClassification(value) {
  return renderXray(value, (v) => {
    const scores = v.scores ?? {}
    const findings = Array.isArray(v.findings) ? v.findings : []
    const critical = Array.isArray(v.critical_findings) ? v.critical_findings : []
    const top3 = Array.isArray(v.top_3) ? v.top_3 : []
    const lines = [
      `Model: ${String(v.model)}, threshold: ${String(v.threshold)}`,
      findings.length > 0
        ? `Findings above threshold: ${findings.map(f => `${f} (${(scores[f] ?? 0).toFixed(3)})`).join(', ')}`
        : 'No findings above threshold.',
    ]
    if (critical.length > 0) lines.push(`Critical findings: ${critical.join(', ')}`)
    lines.push(`Top 3 by score: ${top3.join(', ')}`)
    return lines.join('\n')
  })
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

function renderRadzero(value) {
  return renderXray(value, (v) => {
    const predictions = Array.isArray(v.predictions) ? v.predictions : []
    if (predictions.length === 0) return 'No predictions.'
    return predictions.map(p => `[${p.present ? '+' : ' '}] ${p.prompt} (score=${p.score.toFixed(3)})`).join('\n')
  })
}

function renderCaseRetrieval(value) {
  return renderXray(value, (v) => {
    const cases = Array.isArray(v.cases) ? v.cases : []
    if (cases.length === 0) return `No similar cases found (index has ${String(v.n_indexed ?? 0)} cases).`
    return cases.map(c => `sim=${c.similarity.toFixed(3)}  ${c.image_path}\n  reference: ${c.reference}`).join('\n\n')
  })
}

/**
 * @param {import('@deepseek-ai/cordis').Context} ctx
 * @param {Config} [config]
 */
export function apply(ctx, config = {}) {
  const baseDir = config.baseDir ?? process.env.MEDPLUGIN_SKILLS_DIR
  if (!baseDir) {
    throw new Error(
      'xray-report-generation: no skills scripts directory configured. Set `baseDir` in this plugin\'s '
      + 'cordis config, or the MEDPLUGIN_SKILLS_DIR environment variable, to your MedOmni checkout\'s '
      + 'skills_scripts/ directory (e.g. /path/to/MedOmni/skills_scripts). See this plugin\'s README for setup.',
    )
  }
  if (!existsSync(baseDir)) {
    throw new Error(`xray-report-generation: configured skills scripts directory does not exist: ${baseDir}`)
  }
  const pythonBin = config.pythonBin ?? 'python3'
  const timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS

  const run = (relScript, args, signal) => {
    const scriptPath = `${baseDir}/${relScript}`
    const cwd = scriptPath.slice(0, scriptPath.lastIndexOf('/'))
    return runPythonScript({ pythonBin, scriptPath, args, cwd, timeoutMs, signal })
  }

  ctx.tools.register(defineTool({
    name: 'xray_report_chexagent',
    description: 'Generate a chest X-ray radiology report with CheXagent-8b. Fast, plain narrative findings text (no bounding boxes). Loads a multi-GB model onto a GPU, so a single call can take minutes.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the frontal chest X-ray (PNG, JPG, or DICOM .dcm).' },
      indication: { type: 'string', description: 'Clinical indication, e.g. "Dyspnea."' },
      section_by_section: { type: 'boolean', description: 'Generate section by section (Airway/Breathing/Cardiac/Diaphragm/Everything else) instead of one pass.' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 1.' },
      max_tokens: { type: 'integer', description: 'Max new tokens per generation call. Default 512.' },
    },
    output: {
      schema: xrayOutputSchema('ok'),
      render: (_args, value) => [{ type: 'text', text: renderReport(value) }],
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', args.input]
      if (args.indication !== undefined) cliArgs.push('--indication', args.indication)
      if (args.section_by_section === true) cliArgs.push('--section_by_section')
      cliArgs.push('--gpu', String(args.gpu ?? 1))
      cliArgs.push('--max_tokens', String(args.max_tokens ?? 512))
      return run(SCRIPTS.chexagent, cliArgs, exec.signal)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'xray_report_llava_rad',
    description: 'Generate a chest X-ray radiology report with LLaVA-Rad. Plain narrative findings text; optionally reasons over draft reports from MAIRA-2 and MedGemma for the SAME image when both draft files are given. Loads a multi-GB model onto a GPU, so a single call can take minutes.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the frontal chest X-ray (PNG, JPG, or DICOM .dcm).' },
      indication: { type: 'string', description: 'Clinical indication, e.g. "Shortness of breath."' },
      maira2_draft_file: { type: 'string', description: 'Absolute path to a text file with MAIRA-2\'s draft report for this SAME image. Requires medgemma_draft_file too.' },
      medgemma_draft_file: { type: 'string', description: 'Absolute path to a text file with MedGemma\'s draft report for this SAME image. Requires maira2_draft_file too.' },
      temperature: { type: 'number', description: 'Sampling temperature (0 = greedy). Default 0.' },
      max_new_tokens: { type: 'integer', description: 'Max new tokens to generate. Default 256.' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 2.' },
    },
    output: {
      schema: xrayOutputSchema('success'),
      render: (_args, value) => [{ type: 'text', text: renderReport(value) }],
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', args.input]
      if (args.indication !== undefined) cliArgs.push('--indication', args.indication)
      if (args.maira2_draft_file !== undefined && args.medgemma_draft_file !== undefined) {
        cliArgs.push('--maira2_draft_file', args.maira2_draft_file)
        cliArgs.push('--medgemma_draft_file', args.medgemma_draft_file)
      }
      cliArgs.push('--temperature', String(args.temperature ?? 0))
      cliArgs.push('--max_new_tokens', String(args.max_new_tokens ?? 256))
      // Default 2, matching xray_team.py's run_xray_llava_rad_report wrapper
      // (the standalone script's own argparse default is 0).
      cliArgs.push('--gpu', String(args.gpu ?? 2))
      return run(SCRIPTS.llavaRad, cliArgs, exec.signal)
    },
  }))

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
    name: 'xray_classification',
    description: 'Classify chest X-ray pathologies (18 classes, DenseNet121). Fast — no GPU model load comparable to the report generators.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the chest X-ray (PNG, JPG, or DICOM .dcm).' },
      model: { type: 'string', description: 'Classifier model id. Default "densenet121-res224-all".' },
      threshold: { type: 'number', description: 'Score threshold for a positive finding. Default 0.5.' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 2.' },
    },
    output: {
      schema: xrayOutputSchema('success'),
      render: (_args, value) => [{ type: 'text', text: renderClassification(value) }],
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', args.input]
      if (args.model !== undefined) cliArgs.push('--model', args.model)
      cliArgs.push('--threshold', String(args.threshold ?? 0.5))
      cliArgs.push('--gpu', String(args.gpu ?? 2))
      return run(SCRIPTS.classification, cliArgs, exec.signal)
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

  ctx.tools.register(defineTool({
    name: 'xray_radzero_grounding',
    description: 'Alternate zero-shot phrase grounding using RadZero (Deepnoid) — full-sentence prompts (e.g. "There is cardiomegaly") map to a classification score, not bounding boxes. Different mechanism from xray_report_maira\'s grounding. Use when the user explicitly asks for RadZero, or wants a presence score per prompt instead of boxes.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the chest X-ray (PNG, JPG, or DICOM .dcm).' },
      prompts: { type: 'string', description: 'Comma-separated full-sentence prompts starting with "There is" (optional — omit to use the script\'s default prompt set).' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 2.' },
    },
    output: {
      schema: xrayOutputSchema('success'),
      render: (_args, value) => [{ type: 'text', text: renderRadzero(value) }],
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', args.input, '--gpu', String(args.gpu ?? 2)]
      if (args.prompts !== undefined) cliArgs.push('--prompts', args.prompts)
      return run(SCRIPTS.radzero, cliArgs, exec.signal)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'xray_case_retrieval',
    description: 'Retrieve similar prior chest X-ray cases with confirmed reference reports from a static, pre-built index (case-based precedent). Relevant when the case is genuinely ambiguous or atypical: a borderline classifier score, an unusual presentation, or a rare finding where precedent would help. Skip on clear-cut, high-confidence cases. First call downloads ~6.6GB of checkpoints from Google Drive — can be slow or hit Drive\'s download quota.',
    parameters: {
      input: { type: 'string', required: true, description: 'Absolute path to the query chest X-ray (PNG, JPG, or DICOM .dcm).' },
      k: { type: 'integer', description: 'Number of similar cases to return. Default 5.' },
      gpu: { type: 'integer', description: 'GPU index (-1 for CPU). Default 2.' },
    },
    output: {
      schema: xrayOutputSchema('success'),
      render: (_args, value) => [{ type: 'text', text: renderCaseRetrieval(value) }],
    },
    isConcurrencySafe: () => false,
    async execute(args, exec) {
      const cliArgs = ['--input', args.input, '--k', String(args.k ?? 5), '--gpu', String(args.gpu ?? 2)]
      return run(SCRIPTS.caseRetrieval, cliArgs, exec.signal)
    },
  }))
}
