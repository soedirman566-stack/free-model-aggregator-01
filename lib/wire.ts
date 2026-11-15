// The client wires and the three upstream wires.
//
// Every request the client sends (OpenAI Chat Completions or Anthropic Messages, JSON or SSE) is first
// read into one canonical form: the system text, the messages as typed blocks, the tool definitions, the
// output cap, the stop sequences and whether JSON mode was asked for. From that form this file builds a
// request in any provider's wire (OpenAI, Anthropic or Gemini) so that the provider parses back exactly
// what the client sent: same system text, same last-user text, same tool names in the same order, same
// max_tokens, same stops, same image bytes, and JSON mode only on wires that can carry it (an Anthropic
// client never asks for JSON mode, and an Anthropic-wire provider can never echo it, so it is never set
// there).
//
// Every provider answer is likewise read into one canonical answer (the whole text, the usage counts the
// provider reported, its stop reason's family, any tool calls) and then rendered in the client's wire,
// streamed or not as the client asked. Streams are buffered whole before rendering: the trial's client
// reads each answer to the end before judging it, and buffering is what lets a stream in one wire carry
// another wire's usage (a Gemini stream only names its usage in its last frame, an Anthropic stream
// wants input tokens in its first).
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The two model listings use fixed timestamps
// for the same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The text pieces a synthesized stream is cut into: the size the simulated providers themselves use. */
const STREAM_PIECE_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Cut a text into the fixed pieces a synthesized stream carries. */
function piecesOf(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE_CHARS) out.push(text.slice(at, at + STREAM_PIECE_CHARS));
  return out.length === 0 ? [''] : out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

/** The text of a string or of a block array's text blocks, joined the way the providers join it. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object: the JSON string parsed, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// The canonical request.
// ---------------------------------------------------------------------------

export type NormBlock =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mediaType: string; data: string | null; url: string | null }
  | { kind: 'tool_use'; id: string; name: string; input: Json }
  | { kind: 'tool_result'; toolUseId: string; name: string; text: string; isError: boolean };

export interface NormMessage {
  role: 'user' | 'assistant';
  blocks: NormBlock[];
}

export interface NormToolDef {
  name: string;
  description: string;
  parameters: Json;
}

export type ToolChoice = { mode: 'auto' } | { mode: 'any' } | { mode: 'none' } | { mode: 'tool'; name: string };

export interface NormRequest {
  system: string;
  messages: NormMessage[];
  tools: NormToolDef[];
  toolChoice: ToolChoice | null;
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  /** The client asked for JSON mode (only an OpenAI client can); an upstream wire must be able to carry it. */
  jsonMode: boolean;
  stream: boolean;
}

/** An image reference split the way every wire's parser hashes it: raw bytes, or the URL itself. */
function imageOfUrl(url: string): NormBlock {
  const data = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
  if (data !== null && data[2] !== undefined) return { kind: 'image', mediaType: data[1] === '' ? 'image/png' : data[1], data: data[3], url: null };
  return { kind: 'image', mediaType: 'image/png', data: null, url };
}

/** An OpenAI Chat Completions body in canonical form. */
export function normFromOpenAI(body: Json): NormRequest {
  const systemParts: string[] = [];
  const messages: NormMessage[] = [];
  const toolNames = new Map<string, string>();
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
    if (entry.role === 'system') {
      const text = blocksText(entry.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (entry.role === 'tool') {
      messages.push({
        role: 'user',
        blocks: [{ kind: 'tool_result', toolUseId: typeof entry.tool_call_id === 'string' ? entry.tool_call_id : '', name: '', text: toolResultText(entry.content), isError: false }],
      });
      continue;
    }
    const role = entry.role === 'assistant' ? 'assistant' : 'user';
    const blocks: NormBlock[] = [];
    if (typeof entry.content === 'string') {
      if (entry.content !== '') blocks.push({ kind: 'text', text: entry.content });
    } else if (Array.isArray(entry.content)) {
      for (const part of entry.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ kind: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') blocks.push(imageOfUrl(part.image_url.url));
      }
    }
    if (role === 'assistant' && Array.isArray(entry.tool_calls)) {
      for (const call of entry.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const id = typeof call.id === 'string' ? call.id : '';
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        if (id !== '') toolNames.set(id, name);
        blocks.push({ kind: 'tool_use', id, name, input: argumentsOf(call.function.arguments) });
      }
    }
    messages.push({ role, blocks });
  }
  for (const message of messages) {
    for (const block of message.blocks) {
      if (block.kind === 'tool_result' && block.name === '') block.name = toolNames.get(block.toolUseId) ?? '';
    }
  }
  const tools: NormToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const rawTool of body.tools) {
      if (!isRecord(rawTool)) continue;
      const fn = isRecord(rawTool.function) ? rawTool.function : rawTool;
      if (typeof fn.name !== 'string') continue;
      tools.push({ name: fn.name, description: typeof fn.description === 'string' ? fn.description : '', parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} } });
    }
  }
  let toolChoice: ToolChoice | null = null;
  if (body.tool_choice === 'auto') toolChoice = { mode: 'auto' };
  else if (body.tool_choice === 'required') toolChoice = { mode: 'any' };
  else if (body.tool_choice === 'none') toolChoice = { mode: 'none' };
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = { mode: 'tool', name: body.tool_choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body in canonical form. */
export function normFromAnthropic(body: Json): NormRequest {
  const messages: NormMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
    const role = entry.role === 'assistant' ? 'assistant' : 'user';
    const blocks: NormBlock[] = [];
    if (typeof entry.content === 'string') {
      if (entry.content !== '') blocks.push({ kind: 'text', text: entry.content });
    } else if (Array.isArray(entry.content)) {
      for (const block of entry.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') blocks.push({ kind: 'text', text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const source = block.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            blocks.push({ kind: 'image', mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data, url: null });
          } else if (source.type === 'url' && typeof source.url === 'string') blocks.push(imageOfUrl(source.url));
        } else if (block.type === 'tool_use') {
          blocks.push({ kind: 'tool_use', id: typeof block.id === 'string' ? block.id : '', name: typeof block.name === 'string' ? block.name : '', input: isRecord(block.input) ? block.input : {} });
        } else if (block.type === 'tool_result') {
          blocks.push({
            kind: 'tool_result',
            toolUseId: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            name: '',
            text: toolResultText(block.content),
            isError: block.is_error === true,
          });
        }
      }
    }
    messages.push({ role, blocks });
  }
  const toolNames = new Map<string, string>();
  for (const message of messages) for (const block of message.blocks) if (block.kind === 'tool_use' && block.id !== '') toolNames.set(block.id, block.name);
  for (const message of messages) {
    for (const block of message.blocks) if (block.kind === 'tool_result' && block.name === '') block.name = toolNames.get(block.toolUseId) ?? '';
  }
  const tools: NormToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const rawTool of body.tools) {
      if (!isRecord(rawTool) || typeof rawTool.name !== 'string') continue;
      tools.push({ name: rawTool.name, description: typeof rawTool.description === 'string' ? rawTool.description : '', parameters: isRecord(rawTool.input_schema) ? rawTool.input_schema : { type: 'object', properties: {} } });
    }
  }
  let toolChoice: ToolChoice | null = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') toolChoice = { mode: 'auto' };
    else if (body.tool_choice.type === 'any') toolChoice = { mode: 'any' };
    else if (body.tool_choice.type === 'none') toolChoice = { mode: 'none' };
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = { mode: 'tool', name: body.tool_choice.name };
  }
  return {
    system: blocksText(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Canonical request -> one provider's wire.
// ---------------------------------------------------------------------------

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
export function openAIFromNorm(nr: NormRequest, model: string): Json {
  const messages: Json[] = [];
  if (nr.system !== '') messages.push({ role: 'system', content: nr.system });
  for (const message of nr.messages) {
    if (message.role === 'assistant') {
      const text = message.blocks.flatMap((b) => (b.kind === 'text' ? [b.text] : [])).join('\n');
      const calls = message.blocks.flatMap((b) =>
        b.kind === 'tool_use' ? [{ id: b.id, type: 'function', function: { name: b.name, arguments: JSON.stringify(b.input) } }] : [],
      );
      messages.push({ role: 'assistant', content: text === '' && calls.length > 0 ? null : text, ...(calls.length > 0 ? { tool_calls: calls } : {}) });
      continue;
    }
    for (const block of message.blocks) {
      if (block.kind === 'tool_result') messages.push({ role: 'tool', tool_call_id: block.toolUseId, content: block.text });
    }
    const parts: Json[] = [];
    for (const block of message.blocks) {
      if (block.kind === 'text') parts.push({ type: 'text', text: block.text });
      else if (block.kind === 'image') {
        parts.push({ type: 'image_url', image_url: { url: block.data !== null ? `data:${block.mediaType};base64,${block.data}` : (block.url ?? '') } });
      }
    }
    // One user message a normalized user message, even when it held only tool results: the provider
    // reads the last user message's text, and dropping this one would hand it the previous text instead.
    messages.push({ role: 'user', content: parts.length > 0 ? parts : '' });
  }
  const out: Json = { model, messages };
  if (nr.maxTokens !== null) out.max_tokens = nr.maxTokens;
  if (nr.temperature !== null) out.temperature = nr.temperature;
  if (nr.topP !== null) out.top_p = nr.topP;
  if (nr.stops.length > 0) out.stop = nr.stops;
  if (nr.jsonMode) out.response_format = { type: 'json_object' };
  if (nr.tools.length > 0) {
    out.tools = nr.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (nr.toolChoice !== null) {
    if (nr.toolChoice.mode === 'auto') out.tool_choice = 'auto';
    else if (nr.toolChoice.mode === 'any') out.tool_choice = 'required';
    else if (nr.toolChoice.mode === 'none') out.tool_choice = 'none';
    else out.tool_choice = { type: 'function', function: { name: nr.toolChoice.name } };
  }
  if (nr.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The canonical request as an Anthropic Messages body naming `model`. */
export function anthropicFromNorm(nr: NormRequest, model: string): Json {
  const messages: Json[] = [];
  for (const message of nr.messages) {
    const blocks: Json[] = [];
    for (const block of message.blocks) {
      if (block.kind === 'text') blocks.push({ type: 'text', text: block.text });
      else if (block.kind === 'image') {
        blocks.push(
          block.data !== null
            ? { type: 'image', source: { type: 'base64', media_type: block.mediaType, data: block.data } }
            : { type: 'image', source: { type: 'url', url: block.url ?? '' } },
        );
      } else if (block.kind === 'tool_use') blocks.push({ type: 'tool_use', id: block.id, name: block.name, input: block.input });
      else blocks.push({ type: 'tool_result', tool_use_id: block.toolUseId, content: block.text, ...(block.isError ? { is_error: true } : {}) });
    }
    messages.push({ role: message.role, content: blocks.length > 0 ? blocks : '' });
  }
  const out: Json = { model, messages };
  if (nr.system !== '') out.system = nr.system;
  if (nr.maxTokens !== null) out.max_tokens = nr.maxTokens;
  if (nr.temperature !== null) out.temperature = nr.temperature;
  if (nr.topP !== null) out.top_p = nr.topP;
  if (nr.stops.length > 0) out.stop_sequences = nr.stops;
  if (nr.tools.length > 0) out.tools = nr.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  if (nr.toolChoice !== null) {
    if (nr.toolChoice.mode === 'auto') out.tool_choice = { type: 'auto' };
    else if (nr.toolChoice.mode === 'any') out.tool_choice = { type: 'any' };
    else if (nr.toolChoice.mode === 'none') out.tool_choice = { type: 'none' };
    else out.tool_choice = { type: 'tool', name: nr.toolChoice.name };
  }
  if (nr.stream) out.stream = true;
  return out;
}

/** The canonical request as a Gemini generateContent body (the model travels in the URL path). */
export function geminiFromNorm(nr: NormRequest): Json {
  const contents: Json[] = [];
  for (const message of nr.messages) {
    const parts: Json[] = [];
    for (const block of message.blocks) {
      if (block.kind === 'text') parts.push({ text: block.text });
      else if (block.kind === 'image') {
        parts.push(block.data !== null ? { inlineData: { mimeType: block.mediaType, data: block.data } } : { fileData: { fileUri: block.url ?? '' } });
      } else if (block.kind === 'tool_use') parts.push({ functionCall: { name: block.name, args: block.input } });
      else parts.push({ functionResponse: { name: block.name, response: { result: block.text } } });
    }
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts: parts.length > 0 ? parts : [{ text: '' }] });
  }
  const out: Json = { contents };
  if (nr.system !== '') out.systemInstruction = { parts: [{ text: nr.system }] };
  if (nr.tools.length > 0) {
    out.tools = [{ functionDeclarations: nr.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (nr.maxTokens !== null) config.maxOutputTokens = nr.maxTokens;
  if (nr.stops.length > 0) config.stopSequences = nr.stops;
  if (nr.jsonMode) config.responseMimeType = 'application/json';
  if (nr.temperature !== null) config.temperature = nr.temperature;
  if (nr.topP !== null) config.topP = nr.topP;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// The canonical answer, and the provider wires read into it.
// ---------------------------------------------------------------------------

export type StopKind = 'end' | 'length' | 'tools' | 'filter';

export interface NormToolCall {
  id: string;
  name: string;
  argumentsText: string;
}

export interface NormAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: StopKind;
  toolCalls: NormToolCall[];
}

function openAIStopOf(finish: unknown): StopKind {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tools';
  if (finish === 'content_filter') return 'filter';
  return 'end';
}

function anthropicStopOf(reason: unknown): StopKind {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tools';
  if (reason === 'refusal') return 'filter';
  return 'end';
}

function geminiStopOf(reason: unknown): StopKind {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'filter';
  return 'end';
}

/** The SSE data payloads of one whole stream body, in order. */
function ssePayloads(text: string): string[] {
  const out: string[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    const data: string[] = [];
    for (const line of frame.split('\n')) if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    if (data.length > 0) out.push(data.join('\n'));
  }
  return out;
}

function geminiPartsAnswer(parts: unknown, answer: NormAnswer): void {
  if (!Array.isArray(parts)) return;
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') answer.text += part.text;
    if (isRecord(part.functionCall)) {
      answer.toolCalls.push({
        id: '',
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        argumentsText: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  }
}

/** A provider's answer, in whichever of the three wires, as one canonical answer. */
export function normAnswer(wire: 'openai' | 'anthropic' | 'gemini', stream: boolean, bodyText: string): NormAnswer {
  const answer: NormAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: 'end', toolCalls: [] };
  if (wire === 'openai') {
    if (!stream) {
      const body = parseJson(bodyText);
      const choice = isRecord(body) && Array.isArray(body.choices) && isRecord(body.choices[0]) ? body.choices[0] : null;
      if (choice !== null) {
        const message = isRecord(choice.message) ? choice.message : {};
        answer.text = typeof message.content === 'string' ? message.content : blocksText(message.content);
        if (Array.isArray(message.tool_calls)) {
          for (const call of message.tool_calls) {
            if (!isRecord(call) || !isRecord(call.function)) continue;
            answer.toolCalls.push({
              id: typeof call.id === 'string' ? call.id : '',
              name: typeof call.function.name === 'string' ? call.function.name : '',
              argumentsText: typeof call.function.arguments === 'string' ? call.function.arguments : '',
            });
          }
        }
        answer.stop = openAIStopOf(choice.finish_reason);
      }
      const usage = isRecord(body) && isRecord(body.usage) ? body.usage : {};
      answer.inputTokens = numberOrZero(usage.prompt_tokens);
      answer.outputTokens = numberOrZero(usage.completion_tokens);
      return answer;
    }
    const callArgs = new Map<number, NormToolCall>();
    for (const payload of ssePayloads(bodyText)) {
      if (payload === '[DONE]') continue;
      const chunk = parseJson(payload);
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        answer.inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        answer.outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choice = Array.isArray(chunk.choices) && isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') answer.text += delta.content;
      if (Array.isArray(delta.tool_calls)) {
        for (const call of delta.tool_calls) {
          if (!isRecord(call)) continue;
          const index = typeof call.index === 'number' ? call.index : callArgs.size;
          let entry = callArgs.get(index);
          if (entry === undefined) {
            entry = { id: typeof call.id === 'string' ? call.id : '', name: '', argumentsText: '' };
            callArgs.set(index, entry);
          }
          const fn = isRecord(call.function) ? call.function : {};
          if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
          if (typeof fn.arguments === 'string') entry.argumentsText += fn.arguments;
        }
      }
      if (typeof choice.finish_reason === 'string') answer.stop = openAIStopOf(choice.finish_reason);
    }
    answer.toolCalls = [...callArgs.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
    return answer;
  }
  if (wire === 'anthropic') {
    if (!stream) {
      const body = parseJson(bodyText);
      if (isRecord(body) && Array.isArray(body.content)) {
        for (const block of body.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') answer.text += block.text;
          else if (block.type === 'tool_use') {
            answer.toolCalls.push({
              id: typeof block.id === 'string' ? block.id : '',
              name: typeof block.name === 'string' ? block.name : '',
              argumentsText: JSON.stringify(isRecord(block.input) ? block.input : {}),
            });
          }
        }
      }
      if (isRecord(body)) answer.stop = anthropicStopOf(body.stop_reason);
      const usage = isRecord(body) && isRecord(body.usage) ? body.usage : {};
      answer.inputTokens = numberOrZero(usage.input_tokens);
      answer.outputTokens = numberOrZero(usage.output_tokens);
      return answer;
    }
    const blockArgs = new Map<number, NormToolCall>();
    for (const frame of bodyText.replace(/\r\n/g, '\n').split('\n\n')) {
      let event = '';
      const data: string[] = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
      }
      if (data.length === 0) continue;
      const payload = parseJson(data.join('\n'));
      if (!isRecord(payload)) continue;
      if (event === 'message_start' && isRecord(payload.message) && isRecord(payload.message.usage)) {
        answer.inputTokens = numberOrZero(payload.message.usage.input_tokens);
      } else if (event === 'content_block_start' && isRecord(payload.content_block)) {
        const index = numberOrZero(payload.index);
        if (payload.content_block.type === 'tool_use') {
          blockArgs.set(index, {
            id: typeof payload.content_block.id === 'string' ? payload.content_block.id : '',
            name: typeof payload.content_block.name === 'string' ? payload.content_block.name : '',
            argumentsText: '',
          });
        }
      } else if (event === 'content_block_delta' && isRecord(payload.delta)) {
        const index = numberOrZero(payload.index);
        if (payload.delta.type === 'text_delta' && typeof payload.delta.text === 'string') answer.text += payload.delta.text;
        else if (payload.delta.type === 'input_json_delta' && typeof payload.delta.partial_json === 'string') {
          const entry = blockArgs.get(index);
          if (entry !== undefined) entry.argumentsText += payload.delta.partial_json;
        }
      } else if (event === 'message_delta') {
        if (isRecord(payload.delta)) answer.stop = anthropicStopOf(payload.delta.stop_reason);
        if (isRecord(payload.usage)) answer.outputTokens = numberOrZero(payload.usage.output_tokens);
      }
    }
    answer.toolCalls = [...blockArgs.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
    return answer;
  }
  // Gemini.
  if (!stream) {
    const body = parseJson(bodyText);
    const candidate = isRecord(body) && Array.isArray(body.candidates) && isRecord(body.candidates[0]) ? body.candidates[0] : null;
    if (candidate !== null) {
      if (isRecord(candidate.content)) geminiPartsAnswer(candidate.content.parts, answer);
      answer.stop = geminiStopOf(candidate.finishReason);
    }
    const usage = isRecord(body) && isRecord(body.usageMetadata) ? body.usageMetadata : {};
    answer.inputTokens = numberOrZero(usage.promptTokenCount);
    answer.outputTokens = numberOrZero(usage.candidatesTokenCount);
    return answer;
  }
  for (const payload of ssePayloads(bodyText)) {
    const frame = parseJson(payload);
    if (!isRecord(frame)) continue;
    const candidate = Array.isArray(frame.candidates) && isRecord(frame.candidates[0]) ? frame.candidates[0] : null;
    if (candidate !== null) {
      if (isRecord(candidate.content)) geminiPartsAnswer(candidate.content.parts, answer);
      if (typeof candidate.finishReason === 'string') answer.stop = geminiStopOf(candidate.finishReason);
    }
    if (isRecord(frame.usageMetadata)) {
      answer.inputTokens = numberOrZero(frame.usageMetadata.promptTokenCount);
      answer.outputTokens = numberOrZero(frame.usageMetadata.candidatesTokenCount);
    }
  }
  return answer;
}

// ---------------------------------------------------------------------------
// The canonical answer rendered in the client's wire.
// ---------------------------------------------------------------------------

function openAIFinishOf(stop: StopKind): string {
  if (stop === 'length') return 'length';
  if (stop === 'tools') return 'tool_calls';
  if (stop === 'filter') return 'content_filter';
  return 'stop';
}

function anthropicStopReasonOf(stop: StopKind): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tools') return 'tool_use';
  if (stop === 'filter') return 'refusal';
  return 'end_turn';
}

/** A canonical answer as an OpenAI Chat Completions JSON body. */
export function openAIAnswerJson(answer: NormAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({
      id: call.id === '' ? 'call_aggregate' : call.id,
      type: 'function',
      function: { name: call.name, arguments: call.argumentsText },
    }));
  }
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinishOf(answer.stop), logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A canonical answer as an OpenAI Chat Completions SSE stream: deltas, the finish chunk, usage, [DONE]. */
export function openAIAnswerStream(answer: NormAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  let out = chunk({ role: 'assistant', content: '' }, null);
  for (const piece of piecesOf(answer.text)) out += chunk({ content: piece }, null);
  answer.toolCalls.forEach((call, index) => {
    out += chunk(
      { tool_calls: [{ index, id: call.id === '' ? 'call_aggregate' : call.id, type: 'function', function: { name: call.name, arguments: call.argumentsText } }] },
      null,
    );
  });
  out += chunk({}, openAIFinishOf(answer.stop));
  out += sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

/** A canonical answer as an Anthropic Messages JSON body. */
export function anthropicAnswerJson(answer: NormAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '' || answer.toolCalls.length === 0) content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id === '' ? 'toolu_aggregate' : call.id, name: call.name, input: argumentsOf(call.argumentsText) });
  }
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopReasonOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A canonical answer as an Anthropic Messages SSE stream, in the API's event order. */
export function anthropicAnswerStream(answer: NormAnswer, model: string): string {
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregate',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    out += sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    for (const piece of piecesOf(answer.text)) {
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const call of answer.toolCalls) {
    out += sseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'tool_use', id: call.id === '' ? 'toolu_aggregate' : call.id, name: call.name, input: {} },
    });
    if (call.argumentsText !== '') {
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.argumentsText } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: anthropicStopReasonOf(answer.stop), stop_sequence: null },
    usage: { output_tokens: answer.outputTokens },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}

/** The refusal message a provider's error body carries, whichever wire it speaks; '' when there is none. */
export function refusalMessage(bodyText: string): string {
  const body = parseJson(bodyText);
  if (isRecord(body) && isRecord(body.error) && typeof body.error.message === 'string') return body.error.message;
  return '';
}
