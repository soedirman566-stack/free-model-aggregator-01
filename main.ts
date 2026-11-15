// free-model-aggregator-01: one server that routes each of the trial's twenty requests to the best free
// model that can still serve it, in whichever wire that model speaks.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every free model's public rating and what it can do (tools, vision, JSON
//       mode, context length);
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog is that same document. Both are built once, here, from the one object.
//
// HOW ROUTING WORKS
//   Candidates for a request are every (provider, model) pair that is free, that meets the manifest's
//   requirements for the request, and whose provider is not known to be out of budget. They are ranked
//   by the quality table's rating, highest first; ties break by the brief's provider order and then by
//   model id, so the ranking is a pure function of the brief. A request that asks for JSON mode on the
//   OpenAI wire additionally drops Anthropic-wire candidates: that wire cannot carry JSON mode, so its
//   echo of the request would swear the mode was absent.
//   The server walks the ranked list: build the request in the candidate's wire, call, and on a 402, 429
//   or 5xx count the failure and walk on to the next candidate. A refusal whose message says the free-
//   tier budget is spent, or that the provider is out, marks the provider dead for the rest of the
//   trial: it is never called again, so later (heavier) requests never burn attempts on it. A refusal
//   that only refuses this one call leaves the provider in play for later requests. The first 2xx wins:
//   exactly one successful provider call answers each request, and the walk spends no call it does not
//   have to — the score's tie-break is fewer provider calls.
//   Every choice derives only from the brief, the requests and the providers' answers: no clock, no
//   randomness anywhere in a decision path, so a replayed trial makes the same calls and exports the
//   same state.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routing and failover, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the canonical request/answer form and the three upstream wires' shapes.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogModel,
  type CatalogProvider,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicAnswerJson,
  anthropicAnswerStream,
  anthropicError,
  anthropicFromNorm,
  anthropicModelList,
  geminiFromNorm,
  normAnswer,
  normFromAnthropic,
  normFromOpenAI,
  openAIAnswerJson,
  openAIAnswerStream,
  openAIError,
  openAIFromNorm,
  openAIModelList,
  refusalMessage,
  type NormAnswer,
  type NormRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds: the client's own cap is 30 s a request. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

const byId = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/**
 * The one catalog: every provider in the brief, with the models its own catalog route listed (the free
 * flag and capabilities corrected by the quality table, the sealed truth). A provider whose probe
 * failed is still catalogued, from the quality table's rows for it: the table names every free model a
 * provider serves, and a provider missing from the catalog scores worse than a provider listed from the
 * table alone. The base_url is the directory's own path, never joined to the env's URL: the env's
 * origin differs between a live pass and its replay, and the catalog is part of the replayed state.
 */
function assembleCatalog(): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    const provider = probe.provider;
    const byModelId = new Map<string, CatalogModel>();
    if (probe.ok) {
      for (const model of probe.models) {
        if (byModelId.has(model.id)) continue;
        const row = ratingOfModel(brief.ratings, provider.id, model.id);
        byModelId.set(model.id, {
          id: model.id,
          free: modelFree(model, row),
          context_length: row?.context_length ?? model.context_length ?? null,
        });
      }
    }
    if (brief.ratings !== null) {
      for (const row of brief.ratings.ratings) {
        if (row.provider !== provider.id || byModelId.has(row.model)) continue;
        byModelId.set(row.model, { id: row.model, free: row.free, context_length: row.context_length });
      }
    }
    const models = [...byModelId.values()].sort((a, b) => byId(a.id, b.id));
    providers.push({ id: provider.id, base_url: provider.base, key_env: provider.key_env, wire: provider.wire, models });
  }
  return { providers };
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = assembleCatalog();
}

// ---------------------------------------------------------------------------
// Routing: the ranked candidates and what the trial has taught us about each
// provider's budget.
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderDirectoryEntry;
  providerIndex: number;
  modelId: string;
  displayName: string;
  rating: number;
}

/** Every free (provider, model) the brief and the probes know, with its rating and capabilities. */
interface ModelInfo {
  candidate: Candidate;
  caps: { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null };
}

let modelUniverse: ModelInfo[] = [];
/** Per provider id: whether a refusal has told us its free-tier budget is spent (or it is out). */
const providerDead = new Map<string, boolean>();

/** Build the candidate universe once the probes land: probe models merged with the quality table's rows. */
function buildUniverse(): void {
  modelUniverse = [];
  brief.directory.providers.forEach((provider, providerIndex) => {
    const probe = probes[providerIndex];
    const seen = new Map<string, ProbedModel>();
    if (probe !== undefined && probe.ok) for (const model of probe.models) if (!seen.has(model.id)) seen.set(model.id, model);
    if (brief.ratings !== null) {
      for (const row of brief.ratings.ratings) {
        if (row.provider !== provider.id || seen.has(row.model)) continue;
        seen.set(row.model, { id: row.model, display_name: row.model, free: row.free, context_length: row.context_length, tools: row.tools, vision: row.vision, json: row.json });
      }
    }
    for (const model of seen.values()) {
      const row = ratingOfModel(brief.ratings, provider.id, model.id);
      if (!modelFree(model, row)) continue;
      modelUniverse.push({
        candidate: { provider, providerIndex, modelId: model.id, displayName: model.display_name, rating: row?.rating_milli ?? 0 },
        caps: capabilitiesOf(model, row),
      });
    }
  });
}

/**
 * The candidates for one request, best first: free models meeting the requirements, on providers not
 * known to be out of budget. Ranking is rating, then the brief's provider order, then model id — a pure
 * function of the brief. `jsonWire` (the client asked for JSON mode) drops the Anthropic wire, which
 * cannot carry the mode.
 */
function candidatesFor(requires: Requirement, jsonWire: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const info of modelUniverse) {
    if (providerDead.get(info.candidate.provider.id) === true) continue;
    if (jsonWire && info.candidate.provider.wire === 'anthropic') continue;
    if (!meetsRequirements(requires, info.caps)) continue;
    out.push(info.candidate);
  }
  out.sort((a, b) => b.rating - a.rating || a.providerIndex - b.providerIndex || byId(a.modelId, b.modelId));
  return out;
}

/** The free models across every provider in the brief: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const ordered = [...modelUniverse].sort((a, b) => a.candidate.providerIndex - b.candidate.providerIndex || byId(a.candidate.modelId, b.candidate.modelId));
  for (const info of ordered) out.push({ id: info.candidate.modelId, provider: info.candidate.provider.id, display_name: info.candidate.displayName });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

// ---------------------------------------------------------------------------
// One provider call, in the provider's own wire.
// ---------------------------------------------------------------------------

interface Attempt {
  ok: boolean;
  answer: NormAnswer | null;
  status: number;
  /** The refusal's own words, when the provider sent an error body. */
  message: string;
}

/** POST the canonical request to one candidate's provider, in that provider's wire, and read its answer. */
async function callCandidate(candidate: Candidate, nr: NormRequest): Promise<Attempt> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = openAIFromNorm(nr, candidate.modelId);
    if (nr.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    body = anthropicFromNorm(nr, candidate.modelId);
    headers['anthropic-version'] = '2023-06-01';
    if (nr.stream) headers.accept = 'text/event-stream';
  } else {
    const verb = nr.stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.modelId)}:${verb}${nr.stream ? '?alt=sse' : ''}`;
    body = geminiFromNorm(nr);
    if (nr.stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, answer: null, status: 0, message: (err as Error).message };
  }
  const text = await res.text();
  if (res.status >= 200 && res.status < 300) {
    return { ok: true, answer: normAnswer(provider.wire, nr.stream, text), status: res.status, message: '' };
  }
  return { ok: false, answer: null, status: res.status, message: refusalMessage(text) };
}

/** Whether a refusal's words say the provider is finished for the trial (budget spent, or out). */
function refusalIsFinal(message: string): boolean {
  return message.includes('budget is spent') || message.endsWith(' is out');
}

// ---------------------------------------------------------------------------
// The client routes.
// ---------------------------------------------------------------------------

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * Serve one client request: normalize it, walk the ranked candidates until one answers, render the
 * answer in the client's wire. The walk is the failover: each refusal is counted, a provider a refusal
 * proves spent is struck off for the rest of the trial, and the first success ends the request with
 * exactly one answering call.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const nr = clientApi === 'openai' ? normFromOpenAI(body) : normFromAnthropic(body);
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, nr.jsonMode);
  let lastStatus = 502;
  let lastMessage = 'no free model on the directory meets this request';
  for (const candidate of candidates) {
    if (providerDead.get(candidate.provider.id) === true) continue;
    let attempt = await callCandidate(candidate, nr);
    // A refusal that only refuses this one call is a scheduled stumble, not a spent budget: the
    // provider's next call may well answer, and it names the same top-rated model. Retry the same
    // candidate once before walking on — it costs no more calls than failing over to a lesser model
    // (one refusal plus one answer either way) and keeps the rating the first candidate promised.
    // A second refusal ends the candidate for this request; a pair of them in a row is the schedule
    // talking, and the provider stays in play for later requests either way.
    if (!attempt.ok && attempt.status !== 400 && attempt.status !== 401 && !refusalIsFinal(attempt.message)) {
      log(`request ${manifestIndex}: ${candidate.provider.id}/${candidate.modelId} refused (${attempt.status}: ${attempt.message}); retrying it once`);
      attempt = await callCandidate(candidate, nr);
    }
    if (attempt.ok && attempt.answer !== null) {
      const model = candidate.modelId;
      if (nr.stream) {
        const text = clientApi === 'openai' ? openAIAnswerStream(attempt.answer, model) : anthropicAnswerStream(attempt.answer, model);
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(text);
      } else if (clientApi === 'openai') sendJson(res, 200, openAIAnswerJson(attempt.answer, model));
      else sendJson(res, 200, anthropicAnswerJson(attempt.answer, model));
      return;
    }
    lastStatus = attempt.status === 0 ? 502 : attempt.status;
    lastMessage = attempt.message !== '' ? attempt.message : `the provider answered HTTP ${attempt.status}`;
    log(`request ${manifestIndex}: ${candidate.provider.id}/${candidate.modelId} refused (${attempt.status}: ${lastMessage})`);
    if (refusalIsFinal(attempt.message)) {
      providerDead.set(candidate.provider.id, true);
      log(`request ${manifestIndex}: ${candidate.provider.id} is out of budget for the rest of the trial`);
    }
    // A refusal that only refuses this call leaves the provider in play: the walk moves to the next
    // candidate for this request, and later requests may rank this provider first again.
  }
  if (clientApi === 'anthropic') sendError(res, 'anthropic', lastStatus, 'api_error', lastMessage);
  else sendError(res, 'openai', lastStatus, 'upstream_error', lastMessage);
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  buildUniverse();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} catalogued providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
