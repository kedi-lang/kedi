// Kedi's reference bridge, bundled from kedi-lang/json-rpc ts/reference/openai_chat_bridge.ts (npm run build). Do not edit: change the source.

// openai_chat_bridge.ts
import { spawn } from "node:child_process";

// ../sdk/src/stdout.ts
var write = process.stdout.write.bind(process.stdout);
var protocolOutput = {
  write(frame) {
    return write(frame);
  }
};
var toStderr = (...values) => console.error(...values);
console.log = toStderr;
console.info = toStderr;
console.debug = toStderr;
process.stdout.write = ((chunk, ...rest) => process.stderr.write(chunk, ...rest));

// ../sdk/src/peer.ts
var PROTOCOL_VERSION = 1;
var MAX_FRAME_BYTES = 8 * 1024 * 1024;
var ErrorCode = {
  parseError: -32700,
  invalidRequest: -32600,
  methodNotFound: -32601,
  invalidParams: -32602,
  internalError: -32603,
  unsupported: -32001,
  runFailed: -32002,
  cancelled: -32003,
  overloaded: -32004,
  unknownRun: -32005,
  interrupted: -32006
};
var RpcError = class extends Error {
  code;
  data;
  constructor(code, message, data) {
    super(message);
    this.name = "RpcError";
    this.code = code;
    this.data = data;
  }
};
var Peer = class {
  #output;
  #ended = false;
  #next = 1;
  #pending = /* @__PURE__ */ new Map();
  #requests = /* @__PURE__ */ new Map();
  #notifications = /* @__PURE__ */ new Map();
  #closed = [];
  #observers = [];
  constructor(input, output) {
    this.#output = output;
    let buffered = "";
    input.setEncoding("utf8");
    input.on("data", (chunk) => {
      if (this.#ended) return;
      buffered += chunk;
      let end;
      while ((end = buffered.indexOf("\n")) !== -1) {
        const line = buffered.slice(0, end);
        buffered = buffered.slice(end + 1);
        if (line.trim()) this.#receive(line);
        if (this.#ended) return;
      }
      if (Buffer.byteLength(buffered) > MAX_FRAME_BYTES) this.#fail(new Error("Kedi sent a frame over 8 MiB"));
    });
    input.on("end", () => this.#fail(new Error("Kedi closed the connection")));
  }
  onRequest(method, handler) {
    this.#requests.set(method, handler);
  }
  onNotification(method, handler) {
    this.#notifications.set(method, handler);
  }
  /** Sees every frame Kedi sends, before it is handled. */
  onMessage(listener) {
    this.#observers.push(listener);
  }
  onClose(listener) {
    this.#closed.push(listener);
  }
  get ended() {
    return this.#ended;
  }
  request(method, params) {
    if (this.#ended) return Promise.reject(new Error("the connection to Kedi has ended"));
    const id = this.#next++;
    this.#send({ id, method, params });
    return new Promise((resolve2, reject) => this.#pending.set(id, { resolve: resolve2, reject }));
  }
  notify(method, params) {
    this.#send({ method, params });
  }
  #send(message) {
    if (this.#ended) return;
    this.#output.write(`${JSON.stringify({ jsonrpc: "2.0", ...message })}
`);
  }
  #receive(line) {
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      this.#send({ id: null, error: { code: ErrorCode.parseError, message: "not JSON" } });
      this.#fail(new Error("Kedi sent a line that is not JSON"));
      return;
    }
    for (const observe of this.#observers) observe(message);
    const method = message.method;
    if (typeof method === "string") {
      const params = message.params ?? {};
      if (message.id === void 0) {
        this.#notifications.get(method)?.(params);
        return;
      }
      void this.#answer(message.id, method, params);
      return;
    }
    const waiter = this.#pending.get(message.id);
    if (!waiter) return;
    this.#pending.delete(message.id);
    const error = message.error;
    if (error) waiter.reject(new RpcError(error.code, String(error.message ?? ""), error.data));
    else waiter.resolve(message.result ?? {});
  }
  async #answer(id, method, params) {
    const handler = this.#requests.get(method);
    if (!handler) {
      this.#send({ id, error: { code: ErrorCode.methodNotFound, message: `unknown method ${method}` } });
      return;
    }
    try {
      this.#send({ id, result: await handler(params) });
    } catch (error) {
      this.#send({ id, error: errorFrame(error) });
    }
  }
  #fail(error) {
    if (this.#ended) return;
    this.#ended = true;
    for (const waiter of this.#pending.values()) waiter.reject(error);
    this.#pending.clear();
    for (const listener of this.#closed) listener(error);
  }
};
function errorFrame(error) {
  if (error instanceof RpcError) {
    const frame = { code: error.code, message: error.message };
    if (error.data !== void 0) frame.data = error.data;
    return frame;
  }
  return { code: ErrorCode.runFailed, message: String(error?.message ?? error), data: { harness: harnessError(error) } };
}
function harnessError(error, depth = 0) {
  const value = error;
  const frame = {
    type: value?.name ?? typeof error,
    message: value?.message ?? String(error)
  };
  if (value?.cause !== void 0 && depth < 4) frame.cause = harnessError(value.cause, depth + 1);
  return frame;
}

// ../sdk/src/run.ts
var RunStopped = class extends Error {
  kind;
  constructor(kind) {
    super(kind === "cancel" ? "the run was cancelled" : "the run was interrupted");
    this.name = "RunStopped";
    this.kind = kind;
  }
};
var StepDenied = class extends RpcError {
  constructor(reason, data) {
    super(ErrorCode.runFailed, reason, data);
    this.name = "StepDenied";
  }
};
var BridgeUsageError = class extends Error {
  code;
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "BridgeUsageError";
    this.code = code;
  }
};
var RISK_ORDER = { read_only: 0, mutating: 1, sensitive: 2 };
var Run = class {
  params;
  testMode;
  #peer;
  #capabilities;
  #controller = new AbortController();
  #queued = [];
  #handed = [];
  #onInput;
  #onArrival = false;
  #snapshot;
  // Runtime checks of the protocol's invariants: usage only for admitted requests, nothing after
  // the run ended.
  #admitted = 0;
  #reported = 0;
  #finished = false;
  constructor(peer, params, capabilities, testMode) {
    this.#peer = peer;
    this.params = params;
    this.#capabilities = capabilities;
    this.testMode = testMode;
  }
  get id() {
    return this.params.runId;
  }
  /** Aborts when Kedi cancels or interrupts the run; pass it to every wait. */
  get signal() {
    return this.#controller.signal;
  }
  /**
   * Live input (`run/input`): `handler` gets each input's text as a user turn, right before
   * the next `modelRequest` builds its request. With `{ on: "arrival" }` it gets them as they
   * arrive, for a harness that queues input itself (steering). Either way an input is answered
   * delivered once a request's `describe()` carries its text.
   */
  onInput(handler, options = {}) {
    this.#onInput = handler;
    this.#onArrival = options.on === "arrival";
    if (this.#onArrival) this.#hand();
  }
  /** What an interrupted attempt hands back: `{state}`, `{messages}` or `{transcript}`. */
  onInterrupt(snapshot) {
    this.#snapshot = snapshot;
  }
  /**
   * One model request, with every step Kedi decides around it: live input, history
   * processing, admission per model tried (fallbacks included), the test model, usage.
   */
  async modelRequest(request) {
    this.#throwIfStopped();
    const turn = request.purpose !== "compaction";
    if (turn && this.#queued.length) {
      if (!this.#onInput) {
        throw new BridgeUsageError("KB201", "live input arrived, but the run has no run.onInput handler");
      }
      this.#hand();
    }
    if (turn && this.params.processHistory) {
      if (!request.history) {
        throw new BridgeUsageError("KB202", "the run asks for history processing; the request has no history");
      }
      const kept = await this.#ask("history/process", { runId: this.id, messages: request.history.get() });
      await request.history.set(kept.messages);
    }
    const models = [this.params.model, ...this.#capabilities.modelFallback ? this.params.fallbackModels ?? [] : []];
    let failure;
    for (const model of models) {
      if (this.#capabilities.governed) await this.admit("request");
      if (turn) this.#confirmCarried(request);
      try {
        const reply = this.testMode ? await this.#testReply(request, model) : await this.#race(request.send(model, this.signal));
        if (this.#capabilities.usage) this.reportUsage(request.usage?.(reply) ?? {});
        return reply;
      } catch (error) {
        if (this.signal.aborted) throw this.signal.reason;
        failure = error;
      }
    }
    throw failure;
  }
  /**
   * A call of one of the bridge's own tools, through every step Kedi decides: the
   * `pre_tool_use` hook, approval, admission, the call, the post hooks, artifact admission.
   * Returns what the model should read.
   */
  async nativeTool(call, execute) {
    const before = await this.beforeNativeTool(call);
    if (!before.run) return { ok: false, error: before.reason };
    let result;
    try {
      result = await this.#race(execute(before.arguments, this.signal));
    } catch (error) {
      if (this.signal.aborted) throw this.signal.reason;
      return this.afterNativeTool(call, before.arguments, { error });
    }
    return this.afterNativeTool(call, before.arguments, { result });
  }
  /**
   * The steps before a call of one of the bridge's own tools, for a harness that runs its
   * tools itself: the `pre_tool_use` hook, approval (asked again when edited arguments are
   * riskier), and admission. Returns the arguments to run, or why not to run it.
   */
  async beforeNativeTool(call) {
    if (!this.#capabilities.nativeTools) {
      throw new BridgeUsageError("KB203", `${call.name} is the bridge's own tool, but the bridge does not declare nativeTools`);
    }
    const base = this.#toolBase(call);
    const riskOf = (args2) => typeof call.risk === "function" ? call.risk(args2) : call.risk ?? "mutating";
    let args = call.arguments;
    if (this.#hooks().has("pre_tool_use")) {
      const pre = await this.#ask("hook/invoke", {
        ...base,
        event: "pre_tool_use",
        arguments: args,
        ...call.inputSchema ? { inputSchema: call.inputSchema } : {}
      });
      if (pre.decision === "deny") return { run: false, reason: String(pre.reason ?? "denied") };
      if (pre.decision === "edit") args = pre.arguments;
    }
    let approved = -1;
    while (approved < RISK_ORDER[riskOf(args)]) {
      const risk = riskOf(args);
      const approval = await this.#ask("approval/request", {
        ...base,
        arguments: args,
        risk,
        ...call.description ? { description: call.description } : {}
      });
      if (approval.decision === "deny") return { run: false, reason: String(approval.reason ?? "denied") };
      approved = RISK_ORDER[risk];
      if (approval.decision === "edit") args = approval.arguments;
    }
    if (this.#capabilities.governed) await this.admit("toolAttempt");
    return { run: true, arguments: args };
  }
  /**
   * The steps after a call of one of the bridge's own tools: `post_tool_use` or
   * `post_tool_use_failure`, then artifact admission. Returns what the model should read.
   */
  async afterNativeTool(call, args, outcome) {
    const base = this.#toolBase(call);
    const hooks = this.#hooks();
    if ("error" in outcome) {
      const error = outcome.error;
      const message = String(error?.message ?? outcome.error);
      if (hooks.has("post_tool_use_failure")) {
        const failure = { type: error?.name ?? "Error", message };
        await this.#ask("hook/invoke", { ...base, event: "post_tool_use_failure", arguments: args, error: failure });
      }
      return { ok: false, error: message };
    }
    let result = outcome.result;
    if (hooks.has("post_tool_use")) {
      await this.#ask("hook/invoke", { ...base, event: "post_tool_use", arguments: args, result });
    }
    if (this.params.admitResults) result = (await this.#ask("artifact/admit", { ...base, result })).result;
    return { ok: true, result };
  }
  #toolBase(call) {
    return { runId: this.id, callId: call.callId, tool: call.name, origin: call.origin ?? "adapter_native" };
  }
  #hooks() {
    return new Set(this.params.hooks ?? []);
  }
  /** Calls one of Kedi's tools; Kedi applies the program's hooks, approval and budgets. */
  async callTool(callId, name, args) {
    const answer = await this.#ask("tool/call", { runId: this.id, callId, name, arguments: args });
    return answer.ok ? { ok: true, result: answer.result } : { ok: false, error: String(answer.error) };
  }
  /** Asks Kedi to admit a step; throws `StepDenied` if it does not. `modelRequest` and `nativeTool` do this. */
  async admit(dimension) {
    const answer = await this.#ask("budget/admit", { runId: this.id, dimension });
    if (!answer.admitted) throw new StepDenied(String(answer.reason ?? `${dimension} denied`));
    if (dimension === "request") this.#admitted++;
  }
  /** The bytes of a conversation's media, by its `ref`. */
  async media(ref) {
    return this.#ask("media/get", { runId: this.id, ref });
  }
  /** Reports one model request's token counts; `modelRequest` does this. */
  reportUsage(usage) {
    this.#checkLive("run/usage");
    if (this.#capabilities.governed) {
      if (this.#reported >= this.#admitted) {
        throw new BridgeUsageError("KB204", "run/usage reports a model request Kedi never admitted; admit each request first (modelRequest does)");
      }
      this.#reported++;
    }
    this.#peer.notify("run/usage", { runId: this.id, ...usage });
  }
  /** A `run/event`, with the `events` capability. */
  event(type, fields = {}) {
    this.#checkLive("run/event");
    if (this.#capabilities.events) this.#peer.notify("run/event", { runId: this.id, type, ...fields });
  }
  // The connection's side.
  input(items) {
    return new Promise((answer) => {
      let left = items.length;
      let delivered = true;
      for (const item of items) {
        this.#queued.push({
          ...item,
          answer: (value) => {
            delivered &&= value;
            if (--left === 0) answer(delivered);
          }
        });
      }
      if (this.#onArrival) this.#hand();
    });
  }
  stop(kind) {
    if (!this.signal.aborted) this.#controller.abort(new RunStopped(kind));
  }
  snapshot() {
    return this.#snapshot?.();
  }
  finish() {
    this.#finished = true;
    for (const input of [...this.#queued.splice(0), ...this.#handed.splice(0)]) input.answer(false);
    if (!this.signal.aborted) this.#controller.abort(new RunStopped("cancel"));
  }
  /** Hands queued input to the run's `onInput` handler. */
  #hand() {
    if (!this.#onInput || !this.#queued.length) return;
    const handed = this.#queued.splice(0);
    this.#handed.push(...handed);
    this.#onInput(handed.map((input) => input.text));
  }
  /** Answers delivered the handed input whose text the request carries. */
  #confirmCarried(request) {
    if (!this.#handed.length) return;
    const carried = strings(request.describe());
    this.#handed = this.#handed.filter((input) => {
      if (!carried.some((text2) => text2.includes(input.text))) return true;
      input.answer(true);
      return false;
    });
  }
  async #testReply(request, model) {
    const call = request.describe();
    const params = { runId: this.id, model, ...call };
    if (request.purpose) params.purpose = request.purpose;
    const answer = await this.#ask("model/complete", params);
    return request.reply(answer.message, answer.usage);
  }
  #ask(method, params) {
    this.#checkLive(method);
    return this.#race(this.#peer.request(method, params));
  }
  #checkLive(method) {
    if (this.#finished) throw new BridgeUsageError("KB205", `${method} after run ${this.id} ended; a run's steps belong inside its run()`);
  }
  /** Waits for `promise` unless the run stops first: then rejects with the reason. */
  #race(promise) {
    const signal = this.signal;
    if (signal.aborted) return Promise.reject(signal.reason);
    return new Promise((resolve2, reject) => {
      const stop = () => reject(signal.reason);
      signal.addEventListener("abort", stop, { once: true });
      promise.then(
        (value) => {
          signal.removeEventListener("abort", stop);
          resolve2(value);
        },
        (error) => {
          signal.removeEventListener("abort", stop);
          reject(error);
        }
      );
    });
  }
  #throwIfStopped() {
    if (this.signal.aborted) throw this.signal.reason;
  }
};
function strings(value) {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap(strings);
  if (value && typeof value === "object") return Object.values(value).flatMap(strings);
  return [];
}

// ../sdk/src/connection.ts
function connect(bridge, input, output, refusal) {
  const peer = new Peer(input, output);
  const runs = /* @__PURE__ */ new Map();
  const capabilities = bridge.handshake.capabilities;
  let testMode = false;
  let ready = false;
  let close;
  const closed = new Promise((resolve2) => close = resolve2);
  peer.onClose(() => {
    for (const run2 of runs.values()) run2.stop("cancel");
    close();
  });
  peer.onRequest("initialize", async (params) => {
    if (refusal) throw refusal;
    if (params.protocolVersion !== PROTOCOL_VERSION) {
      throw new RpcError(ErrorCode.unsupported, `this bridge speaks protocol version ${PROTOCOL_VERSION}, not ${params.protocolVersion}`);
    }
    if (params.testModel && !capabilities.testModel) {
      throw new RpcError(ErrorCode.unsupported, "Kedi asked for the test model; this bridge does not declare testModel");
    }
    testMode = Boolean(params.testModel);
    ready = true;
    return { protocolVersion: PROTOCOL_VERSION, ...bridge.handshake };
  });
  peer.onRequest("run", async (params) => {
    if (!ready) throw new RpcError(ErrorCode.invalidRequest, "run before initialize");
    const run2 = new Run(peer, params, capabilities, testMode);
    if (runs.has(run2.id)) throw new RpcError(ErrorCode.invalidParams, `run ${run2.id} is already in progress`);
    runs.set(run2.id, run2);
    try {
      const answer = await bridge.run(run2);
      checkAnswer(run2, answer);
      return answer;
    } catch (error) {
      throw stopped(run2, error);
    } finally {
      runs.delete(run2.id);
      run2.finish();
    }
  });
  peer.onRequest("run/input", async (params) => {
    if (!capabilities.liveInput) throw new RpcError(ErrorCode.unsupported, "this bridge does not declare liveInput");
    const run2 = runs.get(String(params.runId));
    if (!run2) throw new RpcError(ErrorCode.unknownRun, `no run ${params.runId} is in progress`);
    return { delivered: await run2.input(params.items) };
  });
  peer.onNotification("run/cancel", (params) => runs.get(String(params.runId))?.stop("cancel"));
  peer.onNotification("run/interrupt", (params) => {
    if (capabilities.interrupt) runs.get(String(params.runId))?.stop("interrupt");
  });
  peer.onRequest("shutdown", async () => {
    for (const run2 of runs.values()) run2.stop("cancel");
    setImmediate(close);
    return {};
  });
  return { peer, closed };
}
function stopped(run2, error) {
  const reason = run2.signal.aborted ? run2.signal.reason : error;
  if (reason instanceof RunStopped) {
    if (reason.kind === "cancel") return new RpcError(ErrorCode.cancelled, "the run was cancelled");
    return new RpcError(ErrorCode.interrupted, "the run was interrupted", run2.snapshot() ?? null);
  }
  return error;
}
function checkAnswer(run2, answer) {
  const kinds = ["text", "data", "content", "invalid"].filter((key) => key in answer);
  if (kinds.length !== 1) {
    throw new BridgeUsageError("KB210", `a run answers with one of text, data, content or invalid; this one has ${kinds.join(", ") || "none"}`);
  }
  if ("invalid" in answer) return;
  if (run2.params.output && !run2.params.output.media && !("data" in answer)) {
    throw new BridgeUsageError("KB211", "the run has an output schema: answer with data");
  }
  if (Array.isArray(run2.params.history)) {
    const key = run2.params.processHistory ? "transcript" : "messages";
    if (!Array.isArray(answer[key])) {
      throw new BridgeUsageError("KB212", `a transcript run answers with ${key}: ${key === "messages" ? "the turn's new messages, the user turn included" : "the whole conversation after the turn"}`);
    }
  }
}

// ../sdk/src/diagnostics.ts
var SPEC = "https://github.com/kedi-lang/json-rpc/blob/main/spec/protocol.md";
function docs(section) {
  return `${SPEC}#${section}`;
}
function formatDiagnostics(diagnostics) {
  return diagnostics.map((diagnostic) => {
    const lines = [`${diagnostic.severity}[${diagnostic.code}]: ${diagnostic.title}`];
    for (const line of diagnostic.detail ?? []) lines.push(`  \u2502  ${line}`);
    if (diagnostic.help) lines.push(`  = help: ${diagnostic.help}`);
    if (diagnostic.docs) lines.push(`  = docs: ${diagnostic.docs}`);
    return lines.join("\n");
  }).join("\n\n");
}

// ../sdk/src/handshake.ts
function defineHandshake(handshake) {
  return handshake;
}
var KNOWN_CAPABILITIES = /* @__PURE__ */ new Set([
  "structuredOutput",
  "tools",
  "nativeStructuredOutput",
  "nativeToolCalling",
  "nativeTools",
  "events",
  "mediaInput",
  "sessions",
  "transcript",
  "governed",
  "subagents",
  "liveInput",
  "interrupt",
  "usage",
  "modelFallback",
  "imageOutput",
  "audioOutput",
  "testModel"
]);
var KEDI_SETTINGS = /* @__PURE__ */ new Set(["context_limit", "media_limit", "request_retries", "tool_reason", "voice"]);
var EFFORT_LEVELS = /* @__PURE__ */ new Set(["minimal", "low", "medium", "high", "xhigh", "max"]);
var MEDIA_KINDS = /* @__PURE__ */ new Set(["image", "audio", "video", "document"]);
var TRANSPORTS = /* @__PURE__ */ new Set(["stdio", "http", "sse"]);
function checkHandshake(handshake) {
  const found = [];
  const has = handshake.capabilities;
  const requires = (code, capability, needed, why) => {
    if (has[capability] && !has[needed]) {
      found.push({
        code,
        severity: "error",
        title: `${capability} is declared without ${needed}`,
        detail: [why],
        help: `declare ${needed} and implement it, or drop ${capability}`,
        docs: docs("4-capabilities")
      });
    }
  };
  for (const name of Object.keys(has)) {
    if (!KNOWN_CAPABILITIES.has(name)) {
      found.push({
        code: "KB008",
        severity: "error",
        title: `${name} is not a protocol version 1 capability`,
        detail: ["Kedi ignores a capability it does not know, so the bridge would run without it"],
        help: "check its spelling against the capability table",
        docs: docs("4-capabilities")
      });
    }
  }
  requires("KB001", "subagents", "governed", "Kedi admits each of a child's requests against its limits");
  requires("KB002", "liveInput", "subagents", "run/input reaches a running child");
  requires("KB003", "interrupt", "liveInput", "an interrupt comes with the input that caused it");
  if (handshake.mcp?.transports.length && !has.nativeTools) {
    found.push({
      code: "KB007",
      severity: "error",
      title: "mcp is declared without nativeTools",
      detail: ["MCP tools are the bridge's own: their calls need hook/invoke, approval/request and artifact/admit"],
      help: "declare nativeTools and take MCP calls through those steps, or drop mcp",
      docs: docs("4-capabilities")
    });
  }
  if (handshake.compaction && !has.sessions) {
    found.push({
      code: "KB011",
      severity: "error",
      title: "compaction is declared without sessions",
      detail: ["the backend compacts its own session state, which only sessions hand back to Kedi"],
      help: "declare sessions and hand back the compacted state, or drop compaction",
      docs: docs("4-capabilities")
    });
  }
  if (handshake.codemode && !has.tools) {
    found.push({
      code: "KB012",
      severity: "error",
      title: "codemode is declared without tools",
      detail: ["the backend's code calls Kedi's tools, which tools/call reaches"],
      help: "declare tools and offer them to the code, or drop codemode",
      docs: docs("4-capabilities")
    });
  }
  if (handshake.codemode && !["python", "javascript"].includes(handshake.codemode.language)) {
    found.push({ code: "KB013", severity: "error", title: "codemode names no language", help: "declare codemode: {language: python} or {language: javascript}" });
  }
  if (has.mediaInput && !handshake.mediaKinds?.length) {
    found.push({
      code: "KB004",
      severity: "error",
      title: "mediaInput is declared without mediaKinds",
      help: "list the kinds the backend takes: image, audio, video, document",
      docs: docs("4-capabilities")
    });
  }
  for (const kind of handshake.mediaKinds ?? []) {
    if (!MEDIA_KINDS.has(kind)) {
      found.push({ code: "KB010", severity: "error", title: `${kind} is not a media kind`, help: "use image, audio, video or document" });
    }
  }
  const own = (handshake.settings ?? []).filter((name) => KEDI_SETTINGS.has(name));
  if (own.length) {
    found.push({
      code: "KB005",
      severity: "error",
      title: `settings lists Kedi's own settings: ${own.join(", ")}`,
      detail: ["Kedi applies these itself and never sends them"],
      help: "list only the settings your backend applies",
      docs: docs("4-capabilities")
    });
  }
  for (const level of handshake.effort ?? []) {
    if (!EFFORT_LEVELS.has(level)) {
      found.push({ code: "KB006", severity: "error", title: `${level} is not an effort level`, help: "use minimal, low, medium, high, xhigh or max" });
    }
  }
  for (const transport of handshake.mcp?.transports ?? []) {
    if (!TRANSPORTS.has(transport)) {
      found.push({ code: "KB009", severity: "error", title: `${transport} is not an MCP transport`, help: "use stdio, http or sse" });
    }
  }
  return found;
}

// ../sdk/src/preflight.ts
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { PassThrough } from "node:stream";
import { fileURLToPath } from "node:url";

// ../sdk/src/fixtures.generated.ts
var FIXTURES = [
  {
    "name": "cancel",
    "code": "KB109",
    "description": "run/cancel stops the run: it answers -32003.",
    "failure": "run/cancel does not stop the run",
    "help": "pass run.signal to every wait; a cancelled run answers -32003",
    "spec": "5-runs",
    "onModelRequest": {
      "1": [
        {
          "send": "run/cancel"
        }
      ]
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "Too late."
          }
        ]
      }
    ],
    "expect": {
      "error": {
        "code": -32003
      }
    }
  },
  {
    "name": "codemode",
    "code": "KB119",
    "description": "With code mode on, Kedi's tools reach the model only through the backend's code tool.",
    "failure": "code mode still offers Kedi's tools to the model as tools of their own",
    "help": "with run.params.codemode true, turn the backend's code mode on and offer Kedi's tools to its code only",
    "spec": "5-runs",
    "capabilities": [
      "codemode",
      "tools"
    ],
    "run": {
      "tools": [
        {
          "name": "lookup",
          "description": "Look up a city's country.",
          "inputSchema": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string"
              }
            },
            "required": [
              "city"
            ]
          },
          "outputSchema": null,
          "requiredBeforeOutput": false
        }
      ],
      "codemode": true
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "Ready."
          }
        ]
      }
    ],
    "expect": {
      "result": {
        "text": "Ready."
      },
      "modelTools": [
        {
          "request": 1,
          "atLeast": 1,
          "excludes": [
            "lookup"
          ]
        }
      ]
    }
  },
  {
    "name": "compaction",
    "code": "KB118",
    "description": "With a native compaction policy, a session past its threshold is compacted: the summary request reaches the test model with purpose compaction, is admitted like any request, reports no event, and the run still answers with its state.",
    "failure": "the bridge does not compact its session at the program's threshold",
    "help": "with run.params.compaction mode native, compact the session once it passes threshold input tokens, sending the summary request through run.modelRequest with purpose compaction",
    "spec": "7-history",
    "capabilities": [
      "sessions",
      "compactionThreshold"
    ],
    "run": {
      "user": "Where is Lyon?",
      "tools": [
        {
          "name": "lookup",
          "description": "Look up a city's country.",
          "inputSchema": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string"
              }
            },
            "required": [
              "city"
            ]
          },
          "outputSchema": null,
          "requiredBeforeOutput": false
        }
      ],
      "session": {
        "id": "s1",
        "turn": 0,
        "state": null
      },
      "compaction": {
        "mode": "native",
        "threshold": 1
      }
    },
    "model": [
      {
        "parts": [
          {
            "type": "toolCall",
            "id": "t1",
            "name": "lookup",
            "arguments": {
              "city": "Lyon"
            }
          }
        ],
        "usage": {
          "inputTokens": 5e3,
          "outputTokens": 10
        }
      },
      {
        "parts": [
          {
            "type": "text",
            "text": "Lyon is in France."
          }
        ],
        "usage": {
          "inputTokens": 6e3,
          "outputTokens": 10
        }
      }
    ],
    "summaries": [
      {
        "parts": [
          {
            "type": "text",
            "text": "The user asked where Lyon is."
          }
        ],
        "usage": {
          "inputTokens": 5e3,
          "outputTokens": 20
        }
      }
    ],
    "tools": {
      "lookup": {
        "result": "France"
      }
    },
    "expect": {
      "result": {
        "text": "Lyon is in France."
      },
      "resultKeys": [
        "state"
      ],
      "compacted": true
    }
  },
  {
    "name": "continuation",
    "code": "KB112",
    "description": "A continuation replays the rejected answer and Kedi's correction to the model.",
    "failure": "a continuation does not replay the rejected answer and Kedi's correction",
    "help": "replay run.params.continuation: its answer as the model's turn, its message as a user turn",
    "spec": "5-runs",
    "run": {
      "continuation": {
        "answer": {
          "text": "Paris."
        },
        "message": "That was not asked. Answer again."
      }
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "Hello."
          }
        ]
      }
    ],
    "expect": {
      "result": {
        "text": "Hello."
      },
      "modelSaw": [
        {
          "request": 1,
          "contains": "Paris."
        },
        {
          "request": 1,
          "contains": "That was not asked. Answer again."
        }
      ]
    }
  },
  {
    "name": "events",
    "code": "KB113",
    "description": "With events, the model's text beside its tool calls is reported as commentary.",
    "failure": "the bridge's events do not report the model's commentary",
    "help": "send run/event message.completed with phase commentary for text beside tool calls (run.event)",
    "spec": "4-capabilities",
    "capabilities": [
      "events",
      "tools"
    ],
    "run": {
      "tools": [
        {
          "name": "lookup",
          "description": "Look up a city's country.",
          "inputSchema": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string"
              }
            },
            "required": [
              "city"
            ]
          },
          "outputSchema": null,
          "requiredBeforeOutput": false
        }
      ]
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "Let me look it up."
          },
          {
            "type": "toolCall",
            "id": "t1",
            "name": "lookup",
            "arguments": {
              "city": "Lyon"
            }
          }
        ]
      },
      {
        "parts": [
          {
            "type": "text",
            "text": "Lyon is in France."
          }
        ]
      }
    ],
    "tools": {
      "lookup": {
        "result": "France"
      }
    },
    "expect": {
      "result": {
        "text": "Lyon is in France."
      },
      "events": [
        {
          "type": "message.completed",
          "phase": "commentary",
          "text": "Let me look it up."
        }
      ]
    }
  },
  {
    "name": "governed-admits-each-request",
    "code": "KB103",
    "description": "A governed bridge asks budget/admit before every model request.",
    "failure": "model requests are not all admitted",
    "help": "send every model request through run.modelRequest, and turn off the harness's own retries",
    "spec": "4-capabilities",
    "capabilities": [
      "governed",
      "tools"
    ],
    "run": {
      "tools": [
        {
          "name": "lookup",
          "description": "Look up a city's country.",
          "inputSchema": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string"
              }
            },
            "required": [
              "city"
            ]
          },
          "outputSchema": null,
          "requiredBeforeOutput": false
        }
      ]
    },
    "model": [
      {
        "parts": [
          {
            "type": "toolCall",
            "id": "t1",
            "name": "lookup",
            "arguments": {
              "city": "Lyon"
            }
          }
        ]
      },
      {
        "parts": [
          {
            "type": "text",
            "text": "Done."
          }
        ]
      }
    ],
    "tools": {
      "lookup": {
        "result": "France"
      }
    },
    "expect": {
      "result": {
        "text": "Done."
      },
      "order": [
        "budget/admit",
        "model/complete",
        "tool/call",
        "budget/admit",
        "model/complete"
      ],
      "count": [
        {
          "method": "budget/admit",
          "times": 2
        }
      ]
    }
  },
  {
    "name": "governed-denied-request",
    "code": "KB104",
    "description": "A denied request never reaches the model, and the run fails.",
    "failure": "a denied request does not end the run",
    "help": "when budget/admit denies a request, skip it and end the run (run.modelRequest throws StepDenied)",
    "spec": "6-steps-kedi-decides",
    "capabilities": [
      "governed",
      "tools"
    ],
    "budget": {
      "request": 1
    },
    "run": {
      "tools": [
        {
          "name": "lookup",
          "description": "Look up a city's country.",
          "inputSchema": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string"
              }
            },
            "required": [
              "city"
            ]
          },
          "outputSchema": null,
          "requiredBeforeOutput": false
        }
      ]
    },
    "model": [
      {
        "parts": [
          {
            "type": "toolCall",
            "id": "t1",
            "name": "lookup",
            "arguments": {
              "city": "Lyon"
            }
          }
        ]
      },
      {
        "parts": [
          {
            "type": "text",
            "text": "Never sent."
          }
        ]
      }
    ],
    "tools": {
      "lookup": {
        "result": "France"
      }
    },
    "expect": {
      "failed": true,
      "modelRequests": 1
    }
  },
  {
    "name": "history-processing",
    "code": "KB116",
    "description": "With processHistory, the request carries what the program's history processor kept, and the run answers the whole transcript.",
    "failure": "the history processor's result does not decide what the request carries",
    "help": "ask history/process before each request, send what it returns, and answer transcript",
    "spec": "7-history",
    "capabilities": [
      "transcript"
    ],
    "historyProcess": {
      "keepLast": 1
    },
    "run": {
      "user": "What is my name?",
      "history": [
        {
          "id": "m1",
          "role": "user",
          "parts": [
            {
              "type": "text",
              "text": "My name is Ada."
            }
          ]
        },
        {
          "id": "m2",
          "role": "assistant",
          "parts": [
            {
              "type": "text",
              "text": "Hello, Ada."
            }
          ]
        }
      ],
      "processHistory": true
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "I do not know."
          }
        ]
      }
    ],
    "expect": {
      "result": {
        "text": "I do not know."
      },
      "resultKeys": [
        "transcript"
      ],
      "modelMissed": [
        {
          "request": 1,
          "contains": "My name is Ada."
        }
      ],
      "count": [
        {
          "method": "history/process",
          "times": 1
        }
      ]
    }
  },
  {
    "name": "interrupt-session",
    "code": "KB108",
    "description": "An interrupted run in a session answers -32006 with its state, so the conversation continues.",
    "failure": "an interrupted session run does not hand back its state",
    "help": "register run.onInterrupt to return {state} as the run has it so far",
    "spec": "8-children-and-live-input",
    "capabilities": [
      "interrupt",
      "sessions"
    ],
    "run": {
      "session": {
        "id": "s1",
        "turn": 0,
        "state": null
      }
    },
    "onModelRequest": {
      "1": [
        {
          "send": "run/interrupt"
        }
      ]
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "Too late."
          }
        ]
      }
    ],
    "expect": {
      "error": {
        "code": -32006
      },
      "errorDataKeys": [
        "state"
      ]
    }
  },
  {
    "name": "interrupt-transcript",
    "code": "KB117",
    "description": "An interrupted transcript run answers -32006 with the turn's messages so far.",
    "failure": "an interrupted transcript run does not hand back its messages",
    "help": "register run.onInterrupt to return {messages}, the turn so far in Kedi's schema",
    "spec": "8-children-and-live-input",
    "capabilities": [
      "interrupt",
      "transcript"
    ],
    "run": {
      "history": [
        {
          "id": "m1",
          "role": "user",
          "parts": [
            {
              "type": "text",
              "text": "My name is Ada."
            }
          ]
        },
        {
          "id": "m2",
          "role": "assistant",
          "parts": [
            {
              "type": "text",
              "text": "Hello, Ada."
            }
          ]
        }
      ]
    },
    "onModelRequest": {
      "1": [
        {
          "send": "run/interrupt"
        }
      ]
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "Too late."
          }
        ]
      }
    ],
    "expect": {
      "error": {
        "code": -32006
      },
      "errorDataKeys": [
        "messages"
      ]
    }
  },
  {
    "name": "interrupt",
    "code": "KB107",
    "description": "run/interrupt stops the attempt: the run answers -32006.",
    "failure": "run/interrupt does not stop the attempt",
    "help": "pass run.signal to every wait; an interrupted run answers -32006",
    "spec": "8-children-and-live-input",
    "capabilities": [
      "interrupt"
    ],
    "onModelRequest": {
      "1": [
        {
          "send": "run/interrupt"
        }
      ]
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "Too late."
          }
        ]
      }
    ],
    "expect": {
      "error": {
        "code": -32006
      }
    }
  },
  {
    "name": "kedi-tool",
    "code": "KB101",
    "description": "The model calls a Kedi tool; its result reaches the next model request.",
    "failure": "Kedi's tools do not round-trip",
    "help": "call Kedi's tools with tool/call (run.callTool) and send their results to the model",
    "spec": "6-steps-kedi-decides",
    "capabilities": [
      "tools"
    ],
    "run": {
      "tools": [
        {
          "name": "lookup",
          "description": "Look up a city's country.",
          "inputSchema": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string"
              }
            },
            "required": [
              "city"
            ]
          },
          "outputSchema": null,
          "requiredBeforeOutput": false
        }
      ]
    },
    "model": [
      {
        "parts": [
          {
            "type": "toolCall",
            "id": "t1",
            "name": "lookup",
            "arguments": {
              "city": "Lyon"
            }
          }
        ]
      },
      {
        "parts": [
          {
            "type": "text",
            "text": "Lyon is in France."
          }
        ]
      }
    ],
    "tools": {
      "lookup": {
        "result": "France"
      }
    },
    "expect": {
      "result": {
        "text": "Lyon is in France."
      },
      "modelRequests": 2,
      "modelSaw": [
        {
          "request": 2,
          "contains": "France"
        }
      ],
      "order": [
        "model/complete",
        "tool/call",
        "model/complete"
      ]
    }
  },
  {
    "name": "line-separators",
    "code": "KB111",
    "description": "U+2028, U+2029 and CR stay inside one frame.",
    "failure": "U+2028, U+2029 or CR change the prompt",
    "help": "split frames at LF only and pass the text through as given",
    "spec": "2-transport",
    "run": {
      "user": "one\u2028two\u2029three\rfour"
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "Read."
          }
        ]
      }
    ],
    "expect": {
      "result": {
        "text": "Read."
      },
      "modelSaw": [
        {
          "request": 1,
          "contains": "one\u2028two\u2029three\rfour"
        }
      ]
    }
  },
  {
    "name": "live-input",
    "code": "KB106",
    "description": "Input sent while the run runs reaches the next model request, and is acknowledged delivered.",
    "failure": "live input does not reach the next model request",
    "help": "register run.onInput and add each text as a user turn; run.modelRequest answers it delivered",
    "spec": "8-children-and-live-input",
    "capabilities": [
      "liveInput",
      "tools"
    ],
    "run": {
      "tools": [
        {
          "name": "lookup",
          "description": "Look up a city's country.",
          "inputSchema": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string"
              }
            },
            "required": [
              "city"
            ]
          },
          "outputSchema": null,
          "requiredBeforeOutput": false
        }
      ]
    },
    "onModelRequest": {
      "1": [
        {
          "send": "run/input",
          "items": [
            {
              "id": "i1",
              "text": "Use the blue one."
            }
          ]
        }
      ]
    },
    "model": [
      {
        "parts": [
          {
            "type": "toolCall",
            "id": "t1",
            "name": "lookup",
            "arguments": {
              "city": "Lyon"
            }
          }
        ]
      },
      {
        "parts": [
          {
            "type": "text",
            "text": "Blue it is."
          }
        ]
      }
    ],
    "tools": {
      "lookup": {
        "result": "France"
      }
    },
    "expect": {
      "result": {
        "text": "Blue it is."
      },
      "modelSaw": [
        {
          "request": 2,
          "contains": "Use the blue one."
        }
      ],
      "answers": [
        {
          "method": "run/input",
          "result": {
            "delivered": true
          }
        }
      ]
    }
  },
  {
    "name": "media-input",
    "code": "KB114",
    "description": "An image of the user turn reaches the model.",
    "failure": "the user turn's media does not reach the model",
    "help": "send run.params.userContent's media parts to the model where Kedi put them",
    "spec": "5-runs",
    "capabilities": [
      "mediaInput"
    ],
    "run": {
      "user": "Look at photo please.",
      "userContent": [
        {
          "type": "text",
          "text": "Look at "
        },
        {
          "type": "media",
          "name": "photo",
          "kind": "image",
          "mediaType": "image/png",
          "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
        },
        {
          "type": "text",
          "text": " please."
        }
      ]
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "A pixel."
          }
        ]
      }
    ],
    "expect": {
      "result": {
        "text": "A pixel."
      },
      "modelSaw": [
        {
          "request": 1,
          "contains": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
        }
      ]
    }
  },
  {
    "name": "model-fallback",
    "code": "KB115",
    "description": "When the primary model fails, the request is tried on the fallback model.",
    "failure": "a failed request is not tried on the fallback models",
    "help": "with modelFallback, try run.params.fallbackModels in order when a request fails (run.modelRequest does)",
    "spec": "4-capabilities",
    "capabilities": [
      "modelFallback"
    ],
    "failModels": [
      "primary"
    ],
    "run": {
      "model": "primary",
      "fallbackModels": [
        "backup"
      ]
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "From the backup."
          }
        ]
      }
    ],
    "expect": {
      "result": {
        "text": "From the backup."
      },
      "modelRequests": 2,
      "modelSaw": [
        {
          "request": 2,
          "contains": "backup"
        }
      ]
    }
  },
  {
    "name": "structured",
    "code": "KB102",
    "description": "A structured run answers with data.",
    "failure": "a structured run does not answer with data",
    "help": "parse the model's answer against run.output.schema and answer {data}",
    "spec": "5-runs",
    "capabilities": [
      "structuredOutput"
    ],
    "run": {
      "callKind": "produce",
      "output": {
        "schema": {
          "type": "object",
          "properties": {
            "capital": {
              "type": "string"
            }
          },
          "required": [
            "capital"
          ]
        },
        "toolName": "final_result"
      }
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": '{"capital": "Ankara"}'
          }
        ]
      }
    ],
    "expect": {
      "result": {
        "data": {
          "capital": "Ankara"
        }
      },
      "modelRequests": 1
    }
  },
  {
    "name": "text",
    "code": "KB100",
    "description": "Kedi's prompt reaches the model and its text is the run's answer.",
    "failure": "a text run does not answer with the model's text",
    "help": "answer {text} with the model's reply",
    "spec": "5-runs",
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "Hello there."
          }
        ]
      }
    ],
    "expect": {
      "result": {
        "text": "Hello there."
      },
      "modelRequests": 1,
      "modelSaw": [
        {
          "request": 1,
          "contains": "Hello."
        }
      ]
    }
  },
  {
    "name": "transcript",
    "code": "KB105",
    "description": "Kedi's transcript reaches the model; the turn's messages come back.",
    "failure": "Kedi's transcript does not reach the model, or the turn's messages do not come back",
    "help": "build the conversation from run.params.history and answer the turn's messages",
    "spec": "7-history",
    "capabilities": [
      "transcript"
    ],
    "run": {
      "user": "What is my name?",
      "history": [
        {
          "id": "m1",
          "role": "user",
          "parts": [
            {
              "type": "text",
              "text": "My name is Ada."
            }
          ]
        },
        {
          "id": "m2",
          "role": "assistant",
          "parts": [
            {
              "type": "text",
              "text": "Hello, Ada."
            }
          ]
        }
      ]
    },
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "Ada."
          }
        ]
      }
    ],
    "expect": {
      "result": {
        "text": "Ada."
      },
      "resultKeys": [
        "messages"
      ],
      "modelSaw": [
        {
          "request": 1,
          "contains": "My name is Ada."
        }
      ]
    }
  },
  {
    "name": "usage",
    "code": "KB110",
    "description": "With usage, one run/usage per model request.",
    "failure": "a model request's usage is not reported",
    "help": "return the reply's token counts from the request's usage()",
    "spec": "9-usage",
    "capabilities": [
      "usage"
    ],
    "model": [
      {
        "parts": [
          {
            "type": "text",
            "text": "Counted."
          }
        ]
      }
    ],
    "expect": {
      "result": {
        "text": "Counted."
      },
      "count": [
        {
          "method": "run/usage",
          "times": 1
        }
      ]
    }
  }
];

// ../sdk/src/version.ts
var SDK_VERSION = "0.1.0";

// ../sdk/src/preflight.ts
var HERE = fileURLToPath(import.meta.url);
var SDK_SOURCES = basename(HERE) === "preflight.ts" ? dirname(HERE) : void 0;
var FIXTURE_TIMEOUT_MS = 1e4;
var RUN_DEFAULTS = {
  callKind: "invoke",
  promptKind: "raw",
  model: "test",
  fallbackModels: [],
  settings: {},
  effort: null,
  system: "You are being tested.",
  instructionParts: [],
  userContent: null,
  output: null,
  tools: [],
  continuation: null,
  session: null,
  subagent: null,
  hooks: [],
  mcpServers: [],
  admitResults: false
};
function loadFixtures(directory) {
  if (!directory) return structuredClone(FIXTURES);
  return readdirSync(directory).filter((name) => name.endsWith(".json")).sort().map((name) => JSON.parse(readFileSync(join(directory, name), "utf8")));
}
async function preflight(bridge, fixtures = loadFixtures()) {
  const report = { diagnostics: checkHandshake(bridge.handshake), passed: [], skipped: [], cached: false };
  if (report.diagnostics.some((diagnostic) => diagnostic.severity === "error")) return report;
  const declared = {
    ...bridge.handshake.capabilities,
    compaction: bridge.handshake.compaction !== void 0,
    compactionThreshold: bridge.handshake.compaction?.threshold === true,
    codemode: bridge.handshake.codemode !== void 0
  };
  if (!declared.testModel) {
    report.diagnostics.push({
      code: "KB090",
      severity: "warning",
      title: "the bridge has no test model, so its capabilities were not exercised",
      detail: [`${fixtures.length} conformance fixtures need testModel to run without a model endpoint`],
      help: "declare testModel and send model requests through run.modelRequest",
      docs: docs("10-test-model")
    });
    return report;
  }
  for (const fixture of fixtures) {
    const missing = (fixture.capabilities ?? []).filter((name) => !declared[name]);
    if (missing.length) {
      report.skipped.push({ name: fixture.name, reason: `needs ${missing.join(", ")}` });
      continue;
    }
    const problems = await runFixture(bridge, fixture);
    if (problems.length === 0) {
      report.passed.push(fixture.name);
      continue;
    }
    report.diagnostics.push({
      code: fixture.code,
      severity: "error",
      title: fixture.failure,
      detail: [`fixture ${fixture.name}: ${fixture.description}`, ...problems],
      help: fixture.help,
      docs: docs(fixture.spec)
    });
  }
  return report;
}
async function cachedPreflight(bridge, entry, cacheDirectory = cacheDir()) {
  const key = fingerprint(bridge, entry);
  const file = join(cacheDirectory, `${key}.json`);
  try {
    const stored = JSON.parse(readFileSync(file, "utf8"));
    return { ...stored, cached: true };
  } catch {
  }
  const report = await preflight(bridge);
  try {
    mkdirSync(cacheDirectory, { recursive: true });
    const temporary = `${file}.${process.pid}.tmp`;
    writeFileSync(temporary, JSON.stringify(report));
    renameSync(temporary, file);
  } catch {
  }
  return report;
}
function cacheDir() {
  if (process.env.KEDI_BRIDGE_CACHE) return process.env.KEDI_BRIDGE_CACHE;
  return join(process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache"), "kedi", "bridge-checks");
}
function fingerprint(bridge, entry) {
  const hash = createHash("sha256");
  hash.update(`sdk ${SDK_VERSION}
`);
  hash.update(`fixtures ${JSON.stringify(FIXTURES)}
`);
  hash.update(`handshake ${JSON.stringify(bridge.handshake)}
`);
  const files = /* @__PURE__ */ new Set([resolve(entry)]);
  if (SDK_SOURCES) {
    const root = dirname(resolve(entry));
    const own = existsSync(join(root, "package.json")) ? root : void 0;
    for (const directory of [own, SDK_SOURCES]) {
      if (directory) for (const file of walk(directory)) files.add(file);
    }
    for (const name of ["package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lock", "bun.lockb"]) {
      if (own && existsSync(join(own, name))) files.add(join(own, name));
    }
  }
  for (const file of [...files].sort()) {
    hash.update(`${file}
`);
    hash.update(readFileSync(file));
  }
  for (const name of bridge.files ?? []) {
    const file = resolve(dirname(resolve(entry)), name);
    hash.update(`declared ${name}
`);
    hash.update(existsSync(file) ? readFileSync(file) : "missing");
  }
  return hash.digest("hex").slice(0, 32);
}
var SKIPPED_DIRECTORIES = /* @__PURE__ */ new Set(["node_modules", ".git", "dist", "coverage"]);
function* walk(directory) {
  for (const name of readdirSync(directory).sort()) {
    if (SKIPPED_DIRECTORIES.has(name) || name.startsWith(".")) continue;
    const path = join(directory, name);
    if (statSync(path).isDirectory()) yield* walk(path);
    else yield path;
  }
}
async function runFixture(bridge, fixture) {
  const toBridge = new PassThrough();
  const fromBridge = new PassThrough();
  const connection = connect(bridge, toBridge, { write: (frame) => fromBridge.write(frame) });
  const kedi = new Peer(fromBridge, { write: (frame) => toBridge.write(frame) });
  const sent = [];
  const seen = [];
  const events = [];
  const answers = [];
  const script = [...fixture.model ?? []];
  const summaries = [...fixture.summaries ?? []];
  const summarized = [];
  const budget = { ...fixture.budget ?? {} };
  let requestsAdmitted = 0;
  kedi.onMessage((message) => {
    if (typeof message.method === "string") sent.push(message.method);
    if (message.method === "run/event") events.push(message.params);
  });
  kedi.onRequest("model/complete", async (params) => {
    if (params.purpose === "compaction") {
      summarized.push(params);
      const { usage: usage2, ...reply2 } = summaries.shift() ?? { parts: [{ type: "text", text: "A summary." }] };
      return { message: { role: "assistant", ...reply2 }, ...usage2 ? { usage: usage2 } : {} };
    }
    seen.push(params);
    for (const action of fixture.onModelRequest?.[String(seen.length)] ?? []) {
      const { send, ...rest } = action;
      const params2 = { runId: "run-1", ...rest };
      if (send === "run/input") answers.push(settle(send, kedi.request(send, params2)));
      else kedi.notify(send, params2);
    }
    if (fixture.failModels?.includes(String(params.model))) throw new RpcError(-32002, `model ${params.model} is down`);
    const { usage, ...reply } = script.shift() ?? { parts: [{ type: "text", text: "done" }] };
    return { message: { role: "assistant", ...reply }, ...usage ? { usage } : {} };
  });
  kedi.onRequest("budget/admit", async (params) => {
    const dimension = String(params.dimension);
    const left = budget[dimension];
    if (left !== void 0 && left <= 0) {
      setImmediate(() => kedi.notify("run/cancel", { runId: "run-1" }));
      return { admitted: false, dimension, reason: "budget exhausted" };
    }
    if (left !== void 0) budget[dimension] = left - 1;
    if (dimension === "request") requestsAdmitted += 1;
    return { admitted: true };
  });
  kedi.onRequest("history/process", async (params) => {
    const keep = fixture.historyProcess?.keepLast;
    const messages = params.messages;
    return { messages: keep ? messages.slice(-keep) : messages };
  });
  kedi.onRequest("tool/call", async (params) => {
    const tool = fixture.tools?.[String(params.name)] ?? { error: "unknown tool" };
    return "result" in tool ? { ok: true, result: tool.result } : { ok: false, error: String(tool.error) };
  });
  const problems = [];
  try {
    await withTimeout(kedi.request("initialize", { protocolVersion: 1, client: { name: "kedi-bridge-preflight" }, testModel: true }), "initialize");
    const params = { ...RUN_DEFAULTS, user: "Hello.", ...fixture.run ?? {}, runId: "run-1" };
    const answer = await withTimeout(settle("run", kedi.request("run", params)), "the run");
    const replies = await withTimeout(Promise.all(answers), "run/input");
    problems.push(...expectations(fixture.expect ?? {}, answer.answer, seen, sent, replies, events));
    if (fixture.expect?.compacted) {
      if (!summarized.length) problems.push("no summary request (model/complete with purpose compaction) reached Kedi");
      const governed = bridge.handshake.capabilities.governed === true;
      if (governed && requestsAdmitted !== seen.length + summarized.length) {
        problems.push(`${requestsAdmitted} requests admitted, for ${seen.length} turn and ${summarized.length} summary requests: each needs budget/admit`);
      }
      for (const summary of fixture.summaries ?? []) {
        const said = JSON.stringify(summary.parts);
        if (events.some((event) => typeof event.text === "string" && said.includes(event.text) && event.text.length > 0)) {
          problems.push("a run/event reported the summary request's text: a summary is no turn of the run");
        }
      }
    }
  } catch (error) {
    problems.push(String(error.message ?? error));
  } finally {
    toBridge.end();
    fromBridge.end();
    await connection.closed;
  }
  return problems;
}
function settle(method, request) {
  return request.then(
    (result) => ({ method, answer: { result } }),
    (error) => {
      if (!(error instanceof RpcError)) throw error;
      return { method, answer: { error: { code: error.code, message: error.message, data: error.data } } };
    }
  );
}
function withTimeout(promise, what) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${what} did not answer within ${FIXTURE_TIMEOUT_MS / 1e3}s`)), FIXTURE_TIMEOUT_MS);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}
function expectations(expect, answer, seen, sent, replies, events) {
  const found = [];
  const shown = answer.error ? `an error, ${answer.error.code}: ${answer.error.message}` : JSON.stringify(answer.result);
  if ("result" in expect && !contains(answer.result, expect.result)) found.push(`the run answered ${shown}, expected ${JSON.stringify(expect.result)}`);
  if ("error" in expect && !contains(answer.error, expect.error)) found.push(`the run answered ${shown}, expected the error ${JSON.stringify(expect.error)}`);
  if (expect.failed && !answer.error) found.push(`the run answered ${shown}, expected it to fail`);
  for (const key of expect.errorDataKeys ?? []) {
    const data = answer.error?.data;
    if (!data || typeof data !== "object" || !(key in data)) found.push(`the run's error data has no ${key}`);
  }
  for (const key of expect.resultKeys ?? []) {
    if (!answer.result || !(key in answer.result)) found.push(`the run's result has no ${key}`);
  }
  for (const wanted of expect.answers ?? []) {
    const held = replies.some((reply) => reply.method === wanted.method && contains(reply.answer.result, wanted.result));
    if (!held) found.push(`no ${wanted.method} answer held ${JSON.stringify(wanted.result)}`);
  }
  if ("modelRequests" in expect && seen.length !== expect.modelRequests) {
    found.push(`the bridge sent ${seen.length} model requests, expected ${expect.modelRequests}`);
  }
  for (const check of expect.modelSaw ?? []) {
    const request = seen[check.request - 1];
    if (!strings2(request).some((text2) => text2.includes(check.contains))) {
      found.push(`model request ${check.request} did not carry ${JSON.stringify(check.contains)}`);
    }
  }
  for (const check of expect.modelMissed ?? []) {
    const request = seen[check.request - 1];
    if (strings2(request).some((text2) => text2.includes(check.contains))) {
      found.push(`model request ${check.request} carried ${JSON.stringify(check.contains)}`);
    }
  }
  for (const check of expect.modelTools ?? []) {
    const request = seen[check.request - 1];
    const names = (request?.tools ?? []).map((tool) => tool.name);
    if (names.length < (check.atLeast ?? 0)) {
      found.push(`model request ${check.request} offered ${names.length} tools, expected at least ${check.atLeast}`);
    }
    for (const name of check.excludes ?? []) {
      if (names.includes(name)) found.push(`model request ${check.request} offered ${JSON.stringify(name)} as a tool of its own`);
    }
  }
  for (const wanted of expect.events ?? []) {
    if (!events.some((event) => contains(event, wanted))) found.push(`no run/event held ${JSON.stringify(wanted)}; the bridge sent ${JSON.stringify(events)}`);
  }
  const order = expect.order;
  if (order) {
    let position = 0;
    for (const method of sent) if (method === order[position]) position++;
    if (position < order.length) found.push(`the bridge's requests ${JSON.stringify(sent)} do not follow ${JSON.stringify(order)}`);
  }
  for (const counted of expect.count ?? []) {
    const times = sent.filter((method) => method === counted.method).length;
    if (times !== counted.times) found.push(`the bridge sent ${counted.method} ${times} times, expected ${counted.times}`);
  }
  return found;
}
function strings2(value) {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap(strings2);
  if (value && typeof value === "object") return Object.values(value).flatMap(strings2);
  return [];
}
function contains(actual, expected) {
  if (expected && typeof expected === "object" && !Array.isArray(expected)) {
    if (!actual || typeof actual !== "object") return false;
    return Object.entries(expected).every(([key, value]) => key in actual && contains(actual[key], value));
  }
  return JSON.stringify(actual) === JSON.stringify(expected);
}

// ../sdk/src/report.ts
function failed(report) {
  return report.diagnostics.some((diagnostic) => diagnostic.severity === "error");
}
function reportJson(report) {
  return JSON.stringify({ ok: !failed(report), passed: report.passed, skipped: report.skipped, diagnostics: report.diagnostics, cached: report.cached });
}
function jsonDiagnostics(argv) {
  return argv.includes("--diagnostics=json") || argv.some((arg, index) => arg === "--diagnostics" && argv[index + 1] === "json");
}
function describe(report) {
  const errors = report.diagnostics.filter((diagnostic) => diagnostic.severity === "error").length;
  const warnings = report.diagnostics.length - errors;
  const parts = [
    `${report.passed.length} passed`,
    ...errors ? [`${errors} failed`] : [],
    ...warnings ? [`${warnings} warning${warnings === 1 ? "" : "s"}`] : [],
    ...report.skipped.length ? [`${report.skipped.length} skipped (${report.skipped.map((skip) => `${skip.name}: ${skip.reason}`).join("; ")})`] : []
  ];
  const summary = `kedi bridge check: ${parts.join(", ")}${report.cached ? " (cached: the bridge is unchanged)" : ""}`;
  return report.diagnostics.length ? `${formatDiagnostics(report.diagnostics)}

${summary}` : summary;
}

// ../sdk/src/serve.ts
async function serve(bridge, options = {}) {
  const entry = options.entry ?? process.argv[1] ?? "";
  const asJson = jsonDiagnostics(process.argv.slice(2));
  if (process.argv.includes("--check")) {
    const report2 = await preflight(bridge);
    if (asJson) protocolOutput.write(`${reportJson(report2)}
`);
    else process.stderr.write(`${describe(report2)}
`);
    process.exitCode = failed(report2) ? 1 : 0;
    return;
  }
  const mode2 = process.env.KEDI_BRIDGE_PREFLIGHT ?? options.preflight ?? "cached";
  const report = mode2 === "off" ? { diagnostics: checkHandshake(bridge.handshake), passed: [], skipped: [], cached: false } : mode2 === "always" ? await preflight(bridge) : await cachedPreflight(bridge, entry);
  if (report.diagnostics.length) process.stderr.write(`${asJson ? reportJson(report) : describe(report)}
`);
  const errors = report.diagnostics.filter((diagnostic) => diagnostic.severity === "error");
  const refusal = errors.length ? new RpcError(ErrorCode.internalError, refusalMessage(errors, entry), { diagnostics: errors }) : void 0;
  const { closed } = connect(bridge, process.stdin, protocolOutput, refusal);
  await closed;
  process.exit(errors.length ? 1 : 0);
}
function refusalMessage(errors, entry) {
  const codes = errors.map((diagnostic) => `${diagnostic.code} ${diagnostic.title}`).join("; ");
  return `the bridge failed its preflight: ${codes}. Run \`node ${entry} --check\` for the details`;
}

// openai_chat_bridge.ts
function option(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1];
}
var baseUrl = option("base-url");
var mode = option("mode", "native");
var apiKey = process.env[option("api-key-env", "OPENAI_API_KEY")] ?? "unused";
var maxSteps = Number(option("max-steps", "8"));
var native = mode === "native";
var imageOutput = native && process.argv.includes("--image-output");
var audioOutput = native && process.argv.includes("--audio-output");
function log(...parts) {
  process.stderr.write(`[openai-chat-bridge] ${parts.join(" ")}
`);
}
var SETTINGS = {
  temperature: "temperature",
  max_tokens: "max_tokens",
  top_p: "top_p",
  seed: "seed",
  stop_sequences: "stop",
  presence_penalty: "presence_penalty",
  frequency_penalty: "frequency_penalty"
};
var EFFORT = ["minimal", "low", "medium", "high"];
async function post(body, model, signal) {
  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ ...body, model: model ?? "default", stream: false }),
    signal
  });
  if (response.ok) return await response.json();
  const failure = new RpcError(ErrorCode.runFailed, `model ${model ?? "default"} answered ${response.status}: ${await response.text()}`);
  log(failure.message);
  throw failure;
}
function text(content) {
  return typeof content === "string" ? content : "";
}
function calls(parts) {
  return parts.flatMap(
    (part) => part.type === "toolCall" ? [{ id: part.id, type: "function", function: { name: part.name, arguments: JSON.stringify(part.arguments) } }] : []
  );
}
function chatToKedi(message) {
  if (message.role !== "user" || typeof message.content === "string") {
    return message.role === "user" ? { role: "user", parts: [{ type: "text", text: text(message.content) }] } : toKedi(message);
  }
  return {
    role: "user",
    parts: (message.content ?? []).map((part) => {
      if (part.type === "text") return { type: "text", text: part.text };
      const found = /^data:([^;,]+);base64,(.*)$/s.exec(part.image_url.url);
      return found ? { type: "media", kind: "image", mediaType: found[1], data: found[2] } : { type: "media", kind: "image", url: part.image_url.url };
    })
  };
}
function kediToChat(message) {
  const said = message.parts.flatMap((part) => part.type === "text" ? [part.text] : []).join("");
  const toolCalls = calls(message.parts);
  return { role: "assistant", content: said || null, ...toolCalls.length ? { tool_calls: toolCalls } : {} };
}
function userParts(content) {
  return content.map((part) => {
    if (part.type === "text") return { type: "text", text: part.text };
    const url = part.data === void 0 ? String(part.url) : `data:${part.mediaType};base64,${part.data}`;
    return { type: "image_url", image_url: { url } };
  });
}
var mediaCache = /* @__PURE__ */ new Map();
async function mediaFor(run2, ref) {
  if (!mediaCache.has(ref)) mediaCache.set(ref, await run2.media(ref));
  return mediaCache.get(ref);
}
async function fromTranscript(run2, history) {
  const out = [];
  for (const message of history) {
    const parts = message.parts;
    if (message.role === "user") {
      const content = [];
      for (const part of parts) {
        if (part.type === "text") content.push({ type: "text", text: part.text });
        if (part.type === "media") content.push(...userParts([await mediaFor(run2, String(part.ref))]));
      }
      const plain = content.length === 1 && content[0].type === "text";
      out.push({ role: "user", content: plain ? content[0].text : content });
    } else if (message.role === "assistant") {
      const said = parts.flatMap((part) => part.type === "text" ? [part.text] : []).join("");
      const toolCalls = calls(parts);
      out.push({ role: "assistant", content: said || null, ...toolCalls.length ? { tool_calls: toolCalls } : {} });
    } else {
      for (const part of parts) {
        if (part.type !== "toolResult") continue;
        const content = typeof part.result === "string" ? part.result : JSON.stringify(part.result);
        out.push({ role: "tool", tool_call_id: part.id, content });
      }
    }
  }
  return out;
}
function toTranscript(params, turn, withUser) {
  const [first, ...rest] = turn;
  const user = params.userContent ? params.userContent.map((part) => part.type === "text" ? { type: "text", text: part.text } : { type: "media", ref: part.name }) : [{ type: "text", text: params.user }];
  const head = withUser ? [{ role: "user", parts: user }] : first ? [toKedi(first)] : [];
  return [...head, ...rest.map(toKedi)];
}
function toKedi(message) {
  if (message.role === "tool") {
    return { role: "tool", parts: [{ type: "toolResult", id: String(message.tool_call_id), result: message.content }] };
  }
  const parts = message.content ? [{ type: "text", text: String(message.content) }] : [];
  for (const call of message.tool_calls ?? []) {
    const args = JSON.parse(call.function.arguments || "{}");
    parts.push({ type: "toolCall", id: call.id, name: call.function.name, arguments: args });
  }
  return { role: message.role, parts };
}
function answerText(answer) {
  if ("invalid" in answer) return String(answer.invalid);
  if (!native) return JSON.stringify({ action: "final", ...answer });
  return "text" in answer ? String(answer.text) : JSON.stringify(answer.data);
}
var McpClient = class {
  // One stdio MCP server for the run: JSON-RPC, one message per line.
  #child;
  #next = 1;
  #pending = /* @__PURE__ */ new Map();
  constructor(server) {
    this.#child = spawn(String(server.command), server.args ?? [], {
      env: { ...process.env, ...server.env ?? {} },
      stdio: ["pipe", "pipe", "inherit"]
    });
    let buffered = "";
    this.#child.stdout.setEncoding("utf8");
    this.#child.stdout.on("data", (chunk) => {
      buffered += chunk;
      let end;
      while ((end = buffered.indexOf("\n")) !== -1) {
        const line = buffered.slice(0, end);
        buffered = buffered.slice(end + 1);
        if (!line.trim()) continue;
        const message = JSON.parse(line);
        const waiter = this.#pending.get(Number(message.id));
        if (!waiter) continue;
        this.#pending.delete(Number(message.id));
        if (message.error) waiter.reject(new Error(message.error.message));
        else waiter.resolve(message.result ?? {});
      }
    });
    const fail = (error) => {
      for (const waiter of this.#pending.values()) waiter.reject(error);
      this.#pending.clear();
    };
    this.#child.on("error", fail);
    this.#child.on("exit", () => fail(new Error(`MCP server ${server.command} exited`)));
  }
  #write(message) {
    this.#child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", ...message })}
`);
  }
  #request(method, params) {
    const id = this.#next++;
    this.#write({ id, method, params });
    return new Promise((resolve2, reject) => this.#pending.set(id, { resolve: resolve2, reject }));
  }
  async start() {
    const client = { name: "openai-chat-bridge", version: "1" };
    await this.#request("initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: client });
    this.#write({ method: "notifications/initialized", params: {} });
    return (await this.#request("tools/list", {})).tools ?? [];
  }
  async call(name, args) {
    const reply = await this.#request("tools/call", { name, arguments: args });
    const content = reply.content ?? [];
    const said = content.flatMap((part) => part.type === "text" ? [String(part.text)] : []).join("\n");
    if (reply.isError) throw new Error(said || `MCP tool ${name} failed`);
    return reply.structuredContent ?? said;
  }
  close() {
    this.#child.kill();
  }
};
async function run(bridgeRun) {
  const params = bridgeRun.params;
  const clients = [];
  const mcp = /* @__PURE__ */ new Map();
  const definitions = [];
  try {
    for (const server of params.mcpServers ?? []) {
      const client = new McpClient(server);
      clients.push(client);
      for (const tool of await client.start()) {
        if (mcp.has(tool.name) || params.tools.some((kedi) => kedi.name === tool.name)) {
          throw new RpcError(ErrorCode.runFailed, `MCP tool ${tool.name} has the name of another tool`);
        }
        mcp.set(tool.name, client);
        definitions.push({ type: "function", function: { name: tool.name, description: tool.description ?? "", parameters: tool.inputSchema } });
      }
    }
    return await converse(bridgeRun, mcp, definitions);
  } finally {
    for (const client of clients) client.close();
  }
}
async function converse(bridgeRun, mcp, definitions) {
  const params = bridgeRun.params;
  const messages = [];
  if (params.system) messages.push({ role: "system", content: params.system });
  const transcript = Array.isArray(params.history);
  const history = transcript ? await fromTranscript(bridgeRun, params.history) : params.session?.state?.messages ?? [];
  messages.push(...history);
  const turnStart = messages.length;
  messages.push({ role: "user", content: params.userContent ? userParts(params.userContent) : params.user });
  if (params.continuation) {
    messages.push({ role: "assistant", content: answerText(params.continuation.answer) });
    messages.push({ role: "user", content: params.continuation.message });
  }
  const body = { messages };
  for (const [name, value] of Object.entries(params.settings ?? {})) body[SETTINGS[name]] = value;
  if (params.effort) body.reasoning_effort = params.effort;
  if (native && params.tools.length + definitions.length) {
    body.tools = params.tools.map((tool) => ({
      type: "function",
      function: { name: tool.name, description: tool.description ?? "", parameters: tool.inputSchema }
    }));
    body.tools.push(...definitions);
  }
  const output = params.output;
  const media = output?.media;
  if (media === "image") body.modalities = ["image", "text"];
  if (media === "audio") {
    body.modalities = ["text", "audio"];
    body.audio = { voice: output?.media === "audio" ? output.voice ?? "alloy" : "alloy", format: "wav" };
  }
  if (native && output && !media) {
    body.response_format = { type: "json_schema", json_schema: { name: output.toolName, schema: output.schema } };
  }
  let base = transcript ? params.history : [];
  let mark = turnStart;
  let userPending = true;
  const kediTurn = (turn) => toTranscript(params, turn, userPending);
  const finish = (result, content) => {
    const turn = [...messages.slice(turnStart), { role: "assistant", content }];
    const since = [...messages.slice(mark), { role: "assistant", content }];
    if (params.processHistory) return { ...result, transcript: [...base, ...kediTurn(since)] };
    if (transcript) return { ...result, messages: toTranscript(params, turn, true) };
    if (!params.session) return result;
    return { ...result, state: { messages: [...history, ...turn] } };
  };
  bridgeRun.onInterrupt(() => {
    if (params.processHistory) return { transcript: [...base, ...kediTurn(messages.slice(mark))] };
    if (transcript) return { messages: toTranscript(params, messages.slice(turnStart), true) };
    return params.session ? { state: { messages: [...history, ...messages.slice(turnStart)] } } : {};
  });
  bridgeRun.onInput((texts) => {
    for (const said of texts) messages.push({ role: "user", content: said });
  });
  const request = {
    send: (model, signal) => post(body, model, signal),
    describe: () => {
      const [system, ...rest] = messages[0]?.role === "system" ? messages : [null, ...messages];
      const format = body.response_format?.json_schema;
      return {
        system: system ? text(system.content) : null,
        messages: rest.map((message) => chatToKedi(message)),
        tools: (body.tools ?? []).map((tool) => ({
          name: tool.function.name,
          description: tool.function.description,
          inputSchema: tool.function.parameters
        })),
        output: format ? { schema: format.schema, toolName: format.name } : null
      };
    },
    reply: (message, usage) => ({
      choices: [{ message: kediToChat(message) }],
      usage: { prompt_tokens: usage?.inputTokens ?? 0, completion_tokens: usage?.outputTokens ?? 0, total_tokens: usage?.totalTokens ?? 0 }
    }),
    usage: (reply) => ({
      inputTokens: reply.usage?.prompt_tokens,
      outputTokens: reply.usage?.completion_tokens,
      totalTokens: reply.usage?.total_tokens
    }),
    history: {
      get: () => [...base, ...kediTurn(messages.slice(mark))],
      set: async (kept) => {
        base = kept;
        userPending = false;
        const system = params.system ? 1 : 0;
        messages.splice(system, messages.length - system, ...await fromTranscript(bridgeRun, base));
        mark = messages.length;
      }
    }
  };
  for (let step = 0; step < maxSteps; step++) {
    const message = (await bridgeRun.modelRequest(request)).choices[0].message;
    if (native) {
      const toolCalls = message.tool_calls ?? [];
      const said2 = text(message.content);
      if (!toolCalls.length) {
        if (media) return finish(produced(message, output), said2);
        if (!output) return finish({ text: said2 }, said2);
        try {
          return finish({ data: JSON.parse(said2) }, said2);
        } catch (error) {
          return { invalid: said2, error: String(error) };
        }
      }
      if (said2) bridgeRun.event("message.completed", { messageId: `step-${step}`, text: said2, phase: "commentary" });
      messages.push({ role: "assistant", content: message.content ?? null, tool_calls: toolCalls });
      for (const call of toolCalls) {
        const args = JSON.parse(call.function.arguments || "{}");
        const client = mcp.get(call.function.name);
        const outcome2 = client ? await bridgeRun.nativeTool(
          { callId: call.id, name: call.function.name, arguments: args, origin: "mcp", risk: "mutating" },
          (approved) => client.call(call.function.name, approved)
        ) : await bridgeRun.callTool(call.id, call.function.name, args);
        messages.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify(outcome2) });
      }
      continue;
    }
    const said = text(message.content);
    let action;
    try {
      action = JSON.parse(said);
    } catch (error) {
      return { invalid: said, error: String(error) };
    }
    if (action.action === "final") {
      const result = output ? { data: action.data } : { text: String(action.text) };
      return finish(result, said);
    }
    if (action.action !== "call_tool") {
      return { invalid: said, error: `unknown action ${JSON.stringify(action.action)}` };
    }
    messages.push({ role: "assistant", content: said });
    const callId = String(action.callId);
    const outcome = await bridgeRun.callTool(callId, String(action.name), action.arguments ?? {});
    messages.push({ role: "tool", tool_call_id: callId, content: JSON.stringify(outcome) });
  }
  throw new RpcError(ErrorCode.runFailed, `no final answer after ${maxSteps} model requests`);
}
function produced(message, output) {
  const said = text(message.content);
  const parts = said ? [{ type: "text", text: said }] : [];
  if (output.media === "audio" && message.audio?.data) {
    return { content: [...parts, { type: "media", kind: "audio", mediaType: "audio/wav", data: message.audio.data }] };
  }
  const url = message.images?.[0]?.image_url?.url;
  if (output.media === "image" && url) {
    const found = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    const part = found ? { type: "media", kind: "image", mediaType: found[1], data: found[2] } : { type: "media", kind: "image", url };
    return { content: [...parts, part] };
  }
  if (output.media === "image" && output.schema && said) {
    try {
      return { data: JSON.parse(said) };
    } catch {
      return { data: said.trim() };
    }
  }
  throw new RpcError(ErrorCode.runFailed, `the model gave no ${output.media === "audio" ? "speech" : "image"}`);
}
await serve(
  {
    handshake: defineHandshake({
      backend: { name: "openai-chat-bridge", version: "1" },
      kind: "agent-framework",
      capabilities: {
        structuredOutput: true,
        tools: true,
        nativeStructuredOutput: native,
        nativeToolCalling: native,
        // Its MCP servers' tools are its own (native mode).
        nativeTools: native,
        events: true,
        mediaInput: true,
        sessions: true,
        // Kedi holds the conversation (transcript), or this bridge does (sessions).
        transcript: true,
        governed: true,
        // Kedi runs its children here as ordinary runs; params.subagent says whose they are.
        subagents: true,
        liveInput: true,
        interrupt: true,
        usage: true,
        modelFallback: true,
        imageOutput,
        audioOutput,
        // Kedi's test model can answer the model requests instead of the endpoint.
        testModel: true
      },
      settings: Object.keys(SETTINGS),
      effort: [...EFFORT],
      mediaKinds: ["image"],
      mcp: { transports: native ? ["stdio"] : [] }
    }),
    run
  },
  // The fixtures script native replies; in protocol mode the model writes Kedi's protocol as
  // text, so only the handshake is checked.
  { preflight: native ? "cached" : "off" }
);
