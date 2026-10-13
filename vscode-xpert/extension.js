// XpertStream chat participant: a small agent loop that talks to the local
// OpenAI-compatible NPU server and executes workspace tools inside VS Code.
const vscode = require('vscode');
const path = require('path');

const EXCLUDE_GLOB = '**/{.git,node_modules,__pycache__,.venv*,build,dist,out}/**';
const MAX_LIST_FILES = 300;
const MAX_SEARCH_MATCHES = 40;

const SYSTEM_PROMPT = `You are XpertStream, a careful coding agent working inside the user's VS Code workspace.
Use the tools to inspect and change files. Rules:
- Read a file before editing it. Never invent file contents.
- To change an existing file, call replace_in_file with old_text copied exactly from read_file output (include enough lines to be unique). Keep edits small.
- Use write_file only to create new files or to rewrite very small files.
- Paths are relative to the workspace root and use forward slashes.
- When the task is complete, stop calling tools and reply with a short summary of what you changed.
- If the request is only a question, answer it directly and concisely.`;

function tool(name, description, properties, required) {
  return { type: 'function', function: { name, description, parameters: { type: 'object', properties, required } } };
}

const TOOLS = [
  tool('list_files', 'List files under a workspace folder (recursive).',
    { path: { type: 'string', description: 'Folder relative to workspace root, default "."' } }, []),
  tool('read_file', 'Read a text file. Optionally restrict to a 1-based inclusive line range.',
    { path: { type: 'string' }, start_line: { type: 'integer' }, end_line: { type: 'integer' } }, ['path']),
  tool('search_text', 'Case-insensitive text search across workspace files. Returns path:line: text.',
    { query: { type: 'string' } }, ['query']),
  tool('replace_in_file', 'Replace one exact occurrence of old_text with new_text in a file.',
    { path: { type: 'string' }, old_text: { type: 'string' }, new_text: { type: 'string' } }, ['path', 'old_text', 'new_text']),
  tool('write_file', 'Create a new file, or overwrite a small file, with the given content.',
    { path: { type: 'string' }, content: { type: 'string' } }, ['path', 'content']),
];

function config() {
  const c = vscode.workspace.getConfiguration('xpertstream');
  return {
    endpoint: c.get('endpoint', 'http://127.0.0.1:8765').replace(/\/+$/, ''),
    maxSteps: c.get('maxSteps', 12),
    maxResponseTokens: c.get('maxResponseTokens', 1024),
    maxToolOutputChars: c.get('maxToolOutputChars', 6000),
  };
}

function workspaceRoot() {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) throw new Error('Open a folder in VS Code first.');
  return folder.uri;
}

// Resolve a model-supplied path and refuse anything outside the workspace.
function resolvePath(root, p) {
  const raw = String(p ?? '.').trim().replace(/\\/g, '/') || '.';
  const abs = path.isAbsolute(raw) ? path.resolve(raw) : path.resolve(root.fsPath, raw.replace(/^\/+/, ''));
  const rel = path.relative(root.fsPath, abs);
  if (rel.startsWith('..') || path.isAbsolute(rel)) throw new Error(`Path is outside the workspace: ${p}`);
  return vscode.Uri.file(abs);
}

function relPath(root, uri) {
  return path.relative(root.fsPath, uri.fsPath).replace(/\\/g, '/') || '.';
}

function truncate(text, max) {
  return text.length <= max ? text : `${text.slice(0, max)}\n...[truncated ${text.length - max} chars]`;
}

async function exists(uri) {
  try { await vscode.workspace.fs.stat(uri); return true; } catch { return false; }
}

async function listFiles(root, folder = '.') {
  const base = resolvePath(root, folder);
  const uris = await vscode.workspace.findFiles(new vscode.RelativePattern(base, '**/*'), EXCLUDE_GLOB, MAX_LIST_FILES);
  return uris.map(u => relPath(root, u)).sort();
}

const toolImpls = {
  async list_files(root, args) {
    const files = await listFiles(root, args.path);
    return files.length ? files.join('\n') : '(no files)';
  },

  async read_file(root, args, cfg) {
    const doc = await vscode.workspace.openTextDocument(resolvePath(root, args.path));
    const total = doc.lineCount;
    const start = Math.max(1, Number(args.start_line) || 1);
    const end = Math.min(total, Number(args.end_line) || total);
    const lines = [];
    let size = 0;
    let last = start - 1;
    for (let i = start; i <= end; i++) {
      const line = doc.lineAt(i - 1).text;
      if (size + line.length > cfg.maxToolOutputChars) break;
      lines.push(line);
      size += line.length + 1;
      last = i;
    }
    const header = (start === 1 && last === total) ? `${args.path} (${total} lines)` :
      `${args.path} (lines ${start}-${last} of ${total}; use start_line/end_line to read more)`;
    return `${header}\n${lines.join('\n')}`;
  },

  async search_text(root, args) {
    const query = String(args.query || '').toLowerCase();
    if (!query) return 'Error: query is empty.';
    const uris = await vscode.workspace.findFiles(new vscode.RelativePattern(root, '**/*'), EXCLUDE_GLOB, MAX_LIST_FILES);
    const matches = [];
    for (const uri of uris) {
      let text;
      try {
        const bytes = await vscode.workspace.fs.readFile(uri);
        if (bytes.length > 200_000 || bytes.includes(0)) continue;
        text = new TextDecoder().decode(bytes);
      } catch { continue; }
      const lines = text.split(/\r?\n/);
      for (let i = 0; i < lines.length && matches.length < MAX_SEARCH_MATCHES; i++) {
        if (lines[i].toLowerCase().includes(query)) matches.push(`${relPath(root, uri)}:${i + 1}: ${lines[i].trim()}`);
      }
      if (matches.length >= MAX_SEARCH_MATCHES) break;
    }
    return matches.length ? matches.join('\n') : 'No matches.';
  },

  async replace_in_file(root, args, _cfg, stream) {
    const uri = resolvePath(root, args.path);
    const doc = await vscode.workspace.openTextDocument(uri);
    const text = doc.getText();
    const eol = doc.eol === vscode.EndOfLine.CRLF ? '\r\n' : '\n';
    const oldText = String(args.old_text ?? '').replace(/\r?\n/g, eol);
    const newText = String(args.new_text ?? '').replace(/\r?\n/g, eol);
    if (!oldText) return 'Error: old_text is empty.';
    const first = text.indexOf(oldText);
    if (first < 0) return 'Error: old_text was not found. Call read_file again and copy the exact text, including indentation.';
    if (text.indexOf(oldText, first + 1) >= 0) return 'Error: old_text matches more than once. Include more surrounding lines.';
    const edit = new vscode.WorkspaceEdit();
    edit.replace(uri, new vscode.Range(doc.positionAt(first), doc.positionAt(first + oldText.length)), newText);
    if (!await vscode.workspace.applyEdit(edit)) return 'Error: VS Code rejected the edit.';
    await doc.save();
    stream.markdown(`\n\nEdited `);
    stream.anchor(uri, args.path);
    stream.markdown('\n\n');
    return `OK: replaced 1 occurrence in ${args.path}.`;
  },

  async write_file(root, args, _cfg, stream) {
    const uri = resolvePath(root, args.path);
    const content = String(args.content ?? '');
    const existed = await exists(uri);
    if (existed) {
      const doc = await vscode.workspace.openTextDocument(uri);
      const edit = new vscode.WorkspaceEdit();
      edit.replace(uri, new vscode.Range(0, 0, doc.lineCount, 0), content);
      if (!await vscode.workspace.applyEdit(edit)) return 'Error: VS Code rejected the edit.';
      await doc.save();
    } else {
      await vscode.workspace.fs.createDirectory(vscode.Uri.file(path.dirname(uri.fsPath)));
      await vscode.workspace.fs.writeFile(uri, new TextEncoder().encode(content));
    }
    stream.markdown(`\n\n${existed ? 'Rewrote' : 'Created'} `);
    stream.anchor(uri, args.path);
    stream.markdown('\n\n');
    return `OK: ${existed ? 'overwrote' : 'created'} ${args.path} (${content.length} chars).`;
  },
};

function describeCall(name, args) {
  const target = args.path || args.query || '.';
  return `${name.replace(/_/g, ' ')}: ${target}`;
}

async function runTool(root, call, cfg, stream) {
  const name = call.function?.name;
  const impl = toolImpls[name];
  if (!impl) return `Error: unknown tool "${name}". Available: ${Object.keys(toolImpls).join(', ')}.`;
  let args;
  try {
    args = typeof call.function.arguments === 'string' ? JSON.parse(call.function.arguments || '{}') : (call.function.arguments || {});
  } catch {
    return 'Error: arguments were not valid JSON.';
  }
  stream.progress(describeCall(name, args));
  try {
    return truncate(await impl(root, args, cfg, stream), cfg.maxToolOutputChars);
  } catch (err) {
    return `Error: ${err.message || err}`;
  }
}

// Calls the server with streaming; forwards text deltas and collects tool calls.
async function callModel(cfg, messages, onText, signal, tools = TOOLS) {
  const res = await fetch(`${cfg.endpoint}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'local', messages, tools: tools.length ? tools : undefined, stream: true, temperature: 0, max_tokens: cfg.maxResponseTokens }),
    signal,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const err = new Error(body.error?.message || `Server returned HTTP ${res.status}`);
    err.code = body.error?.code;
    throw err;
  }

  const decoder = new TextDecoder();
  let buffer = '';
  let content = '';
  let finish = null;
  const toolCalls = [];
  for await (const chunk of res.body) {
    buffer += decoder.decode(chunk, { stream: true });
    let sep;
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const event = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of event.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (!data || data === '[DONE]') continue;
        const parsed = JSON.parse(data);
        if (parsed.error) {
          const err = new Error(parsed.error.message || 'Server error');
          err.code = parsed.error.code;
          throw err;
        }
        const choice = parsed.choices?.[0];
        if (!choice) continue;
        const delta = choice.delta || {};
        if (delta.content) {
          content += delta.content;
          onText(delta.content);
        }
        for (const tc of delta.tool_calls || []) {
          toolCalls[tc.index ?? toolCalls.length] = { id: tc.id, type: 'function', function: { ...tc.function } };
        }
        if (choice.finish_reason) finish = choice.finish_reason;
      }
    }
  }
  return { content, toolCalls: toolCalls.filter(Boolean), finish };
}

// Frees context by blanking the oldest large tool output. Returns false when nothing is left to drop.
function compactMessages(messages) {
  const victim = messages.find(m => m.role === 'tool' && !m.compacted && m.content.length > 200);
  if (!victim) return false;
  victim.content = '[output removed to save context; call the tool again if you still need it]';
  victim.compacted = true;
  return true;
}

function historyMessages(context) {
  const out = [];
  for (const turn of context.history.slice(-6)) {
    if (turn instanceof vscode.ChatRequestTurn) {
      out.push({ role: 'user', content: truncate(turn.prompt, 1000) });
    } else if (turn instanceof vscode.ChatResponseTurn) {
      const text = turn.response
        .filter(p => p instanceof vscode.ChatResponseMarkdownPart)
        .map(p => p.value.value).join('').trim();
      if (text) out.push({ role: 'assistant', content: truncate(text, 1000) });
    }
  }
  return out;
}

async function referencesText(root, references, cfg) {
  const parts = [];
  for (const ref of references) {
    let uri, range;
    if (ref.value instanceof vscode.Uri) uri = ref.value;
    else if (ref.value instanceof vscode.Location) ({ uri, range } = ref.value);
    else if (typeof ref.value === 'string') { parts.push(ref.value); continue; }
    else continue;
    try {
      const doc = await vscode.workspace.openTextDocument(uri);
      const text = truncate(doc.getText(range), cfg.maxToolOutputChars);
      const label = relPath(root, uri) + (range ? ` (lines ${range.start.line + 1}-${range.end.line + 1})` : '');
      parts.push(`Attached ${label}:\n\`\`\`\n${text}\n\`\`\``);
    } catch { /* unreadable reference, skip */ }
  }
  return parts.join('\n\n');
}

async function handler(request, context, stream, token) {
  const cfg = config();
  let root;
  try { root = workspaceRoot(); } catch (err) { stream.markdown(err.message); return; }

  const files = await listFiles(root);
  const active = vscode.window.activeTextEditor?.document.uri;
  let system = `${SYSTEM_PROMPT}\n\nWorkspace files:\n${files.slice(0, 80).join('\n')}`;
  if (files.length > 80) system += `\n... (${files.length - 80} more, use list_files)`;
  if (active && active.scheme === 'file' && !path.relative(root.fsPath, active.fsPath).startsWith('..')) {
    system += `\n\nThe user currently has ${relPath(root, active)} open.`;
  }

  const attached = await referencesText(root, request.references, cfg);
  const messages = [
    { role: 'system', content: system },
    ...historyMessages(context),
    { role: 'user', content: attached ? `${request.prompt}\n\n${attached}` : request.prompt },
  ];

  const abort = new AbortController();
  token.onCancellationRequested(() => abort.abort());

  let device;
  try {
    const health = await (await fetch(`${cfg.endpoint}/health`, { signal: AbortSignal.timeout(5000) })).json();
    if (health.status !== 'ok') {
      stream.markdown(`The XpertStream server on ${health.device} is in an error state: \`${health.error}\`. Restart \`AgentServer.cmd\`.`);
      return;
    }
    device = health.device || 'local model';
  } catch {
    stream.markdown(`Cannot reach the XpertStream server at \`${cfg.endpoint}\`. Start it with \`AgentServer.cmd\` (or wait until it prints "listening") and try again.`);
    return;
  }

  let steps = 0;
  while (steps < cfg.maxSteps && !token.isCancellationRequested) {
    stream.progress(steps === 0 ? `Thinking on ${device}...` : `Step ${steps + 1} on ${device}...`);
    let result;
    try {
      result = await callModel(cfg, messages, text => stream.markdown(text), abort.signal);
    } catch (err) {
      if (token.isCancellationRequested) return;
      if (err.code === 'context_length_exceeded' && compactMessages(messages)) {
        stream.progress('Context full, dropping old tool output...');
        continue;
      }
      if (err.cause?.code === 'ECONNREFUSED' || /fetch failed/i.test(err.message)) {
        stream.markdown(`Cannot reach the XpertStream server at \`${cfg.endpoint}\`. Start it with \`AgentServer.cmd\` and try again.`);
      } else {
        stream.markdown(`\n\n**Error:** ${err.message}`);
      }
      return;
    }
    steps++;

    const assistant = { role: 'assistant', content: result.content || null };
    if (result.toolCalls.length) assistant.tool_calls = result.toolCalls;
    messages.push(assistant);
    if (!result.toolCalls.length) {
      if (result.finish === 'length') stream.markdown('\n\n_Response hit the token limit._');
      return;
    }
    for (const call of result.toolCalls) {
      if (token.isCancellationRequested) return;
      const output = await runTool(root, call, cfg, stream);
      messages.push({ role: 'tool', tool_call_id: call.id, content: output });
    }
  }
  if (!token.isCancellationRequested) stream.markdown(`\n\n_Stopped after ${cfg.maxSteps} steps._`);
}

function partText(part) {
  if (part instanceof vscode.LanguageModelTextPart) return part.value;
  if (typeof part?.value === 'string') return part.value;
  return '';
}

// Converts VS Code chat messages (from Copilot Ask/Agent mode) to OpenAI chat messages.
function toOpenAIMessages(messages) {
  const out = [];
  for (const m of messages) {
    const role = m.role === vscode.LanguageModelChatMessageRole.User ? 'user'
      : m.role === vscode.LanguageModelChatMessageRole.Assistant ? 'assistant' : 'system';
    let text = '';
    const toolCalls = [];
    const toolResults = [];
    for (const part of m.content) {
      if (part instanceof vscode.LanguageModelToolCallPart) {
        toolCalls.push({ id: part.callId, type: 'function', function: { name: part.name, arguments: JSON.stringify(part.input ?? {}) } });
      } else if (part instanceof vscode.LanguageModelToolResultPart) {
        toolResults.push({ role: 'tool', tool_call_id: part.callId, content: part.content.map(partText).join('') });
      } else {
        text += partText(part);
      }
    }
    out.push(...toolResults);
    if (role === 'assistant' && toolCalls.length) out.push({ role, content: text || null, tool_calls: toolCalls });
    else if (text) out.push({ role, content: text });
  }
  return out;
}

function shorten(text, max) {
  const s = String(text).replace(/\s+/g, ' ').trim();
  const firstSentence = s.match(/^.*?[.!?](\s|$)/)?.[0].trim() ?? s;
  return firstSentence.length <= max ? firstSentence : `${firstSentence.slice(0, max - 3)}...`;
}

// Copilot's tool schemas are verbose; keep names, types, enums and required fields, trim prose.
function compactSchema(node, depth = 0) {
  if (Array.isArray(node)) return node.map(n => compactSchema(n, depth + 1));
  if (!node || typeof node !== 'object') return node;
  const out = {};
  for (const [key, value] of Object.entries(node)) {
    if (['examples', 'example', '$schema', 'markdownDescription', 'default', 'title'].includes(key)) continue;
    if (key === 'description') {
      if (depth <= 2) out.description = shorten(value, 80);
      continue;
    }
    if (key === 'properties' && value && typeof value === 'object' && !Array.isArray(value)) {
      // Keys here are parameter names, not schema keywords, so never filter them.
      out.properties = Object.fromEntries(Object.entries(value).map(([name, sub]) => [name, compactSchema(sub, depth + 1)]));
      continue;
    }
    out[key] = compactSchema(value, depth + 1);
  }
  return out;
}

class XpertModelProvider {
  constructor() {
    this._changed = new vscode.EventEmitter();
    this.onDidChangeLanguageModelChatInformation = this._changed.event;
    this._advertised = undefined;
    // Copilot caches model info, so re-announce when the server restarts with different limits.
    this._timer = setInterval(() => this._checkForChanges(), 15000);
  }

  dispose() {
    clearInterval(this._timer);
    this._changed.dispose();
  }

  async _fetchHealth() {
    try {
      return await (await fetch(`${config().endpoint}/health`, { signal: AbortSignal.timeout(3000) })).json();
    } catch {
      return null;
    }
  }

  async _checkForChanges() {
    const health = await this._fetchHealth();
    const key = health ? `${health.device}/${health.max_prompt_len}/${health.max_response_len}` : 'offline';
    if (this._advertised !== undefined && key !== this._advertised) this._changed.fire();
  }

  async provideLanguageModelChatInformation() {
    const cfg = config();
    const health = await this._fetchHealth();
    this._advertised = health ? `${health.device}/${health.max_prompt_len}/${health.max_response_len}` : 'offline';
    return [{
      id: 'xpertstream-local',
      name: 'XpertStream (local)',
      family: 'xpertstream',
      version: '0.2.0',
      // Headroom for the chat template and tool schemas, which Copilot does not count.
      maxInputTokens: Math.floor((health?.max_prompt_len ?? 4096) * 0.85),
      maxOutputTokens: health?.max_response_len ?? cfg.maxResponseTokens,
      tooltip: `Local OpenVINO model served by AgentServer.cmd at ${cfg.endpoint}`,
      detail: health ? `${health.model} on ${health.device}` : 'server offline',
      capabilities: { toolCalling: true, imageInput: false },
    }];
  }

  async provideLanguageModelChatResponse(_model, messages, options, progress, token) {
    const cfg = config();
    const compact = vscode.workspace.getConfiguration('xpertstream').get('compactTools', true);
    const tools = (options.tools || []).map(t => {
      const parameters = t.inputSchema ?? { type: 'object', properties: {} };
      return {
        type: 'function',
        function: compact
          ? { name: t.name, description: shorten(t.description || '', 160), parameters: compactSchema(parameters) }
          : { name: t.name, description: t.description, parameters },
      };
    });
    const abort = new AbortController();
    token.onCancellationRequested(() => abort.abort());
    const result = await callModel(cfg, toOpenAIMessages(messages),
      text => progress.report(new vscode.LanguageModelTextPart(text)), abort.signal, tools);
    for (const call of result.toolCalls) {
      let input = {};
      try { input = JSON.parse(call.function.arguments || '{}'); } catch { /* keep empty input */ }
      progress.report(new vscode.LanguageModelToolCallPart(call.id, call.function.name, input));
    }
  }

  async provideTokenCount(_model, text) {
    const s = typeof text === 'string' ? text : text.content.map(p => partText(p) || JSON.stringify(p.input ?? '')).join('');
    return Math.ceil(s.length / 3);
  }
}

function activate(context) {
  const participant = vscode.chat.createChatParticipant('xpertstream.agent', handler);
  participant.iconPath = new vscode.ThemeIcon('chip');
  context.subscriptions.push(participant);
  const provider = new XpertModelProvider();
  context.subscriptions.push(provider, vscode.lm.registerLanguageModelChatProvider('xpertstream', provider));
}

function deactivate() {}

module.exports = { activate, deactivate };
