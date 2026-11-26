#!/usr/bin/env node
// Kedi JSON-RPC bridge (protocol version 1) for an OpenAI-compatible chat endpoint.
//
// Kedi renders every model-visible text; this bridge transports it. It sends Kedi's system
// text and prompt verbatim, Kedi's tools and output schema as native fields (--mode native)
// or nothing extra (--mode protocol: Kedi's protocol parts are already in the system text),
// and calls Kedi tools back with tool/call. It writes no instructions of its own.
//
// It is governed: it asks Kedi (budget/admit) before every request to the endpoint, and
// makes no requests of its own besides those: no retries. With fallbackModels it tries the
// next model when a request fails, each attempt admitted like any other request.
//
// With --image-output it asks image models to draw (OpenRouter's chat completions: modalities
// image, message.images); with --audio-output it asks audio models to speak (OpenAI's chat
// completions audio). Both need --mode native.
//
// Usage: node openai_chat_bridge.mjs --base-url URL [--mode native|protocol]
//        [--api-key-env NAME] [--max-steps N] [--image-output] [--audio-output]
//
// In native mode it connects stdio MCP servers (> mcp:) for each run and offers their tools,
// each call going through Kedi's hooks, approval, tool-attempt budget and artifact admission.
// stdout carries only protocol frames; diagnostics go to stderr.


import { spawn } from "node:child_process";

const PROTOCOL_VERSION = 1;
const RUN_FAILED = -32002;
const CANCELLED = -32003;
const INTERRUPTED = -32006;
const METHOD_NOT_FOUND = -32601;

function option(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1];
}

const baseUrl = option("base-url");
const mode = option("mode", "native");
const apiKey = process.env[option("api-key-env", "OPENAI_API_KEY")] ?? "unused";
const maxSteps = Number(option("max-steps", "8"));
const native = mode === "native";
const imageOutput = native && process.argv.includes("--image-output");
// Set by initialize: Kedi's test model answers the model requests (model/complete).
let testMode = false;
const audioOutput = native && process.argv.includes("--audio-output");

function log(...parts) {
  process.stderr.write(`[openai-chat-bridge] ${parts.join(" ")}\n`);
}

function send(message) {
  process.stdout.write(`${JSON.stringify({ jsonrpc: "2.0", ...message })}\n`);
}

class RpcError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

let nextId = 1;
const pending = new Map();
const runs = new Map();

function request(method, params) {
  const id = nextId++;
  send({ id, method, params });
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

// Kedi's > settings: names this bridge applies, and the OpenAI fields they become.
const SETTINGS = {
  temperature: "temperature",
  max_tokens: "max_tokens",
  top_p: "top_p",
  seed: "seed",
  stop_sequences: "stop",
  presence_penalty: "presence_penalty",
  frequency_penalty: "frequency_penalty",
};
// Kedi's > effort: levels this bridge sends as reasoning_effort.
const EFFORT = ["minimal", "low", "medium", "high"];

async function complete(runId, body, models, signal) {
  // Tries each model in order when a request fails, asking Kedi before every attempt.
  // Returns the model's message; reports the request's token counts with run/usage.
  let failure;
  for (const model of models) {
    const admission = await request("budget/admit", { runId, dimension: "request" });
    if (!admission.admitted) {
      // Kedi fails the run with its own error; the request is never sent.
      throw new RpcError(RUN_FAILED, admission.reason);
    }
    if (testMode) return reported(runId, await testCompletion(runId, { ...body, model }));
    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ ...body, model, stream: false }),
      signal,
    });
    if (response.ok) return reported(runId, await response.json());
    failure = new RpcError(RUN_FAILED, `model ${model} answered ${response.status}: ${await response.text()}`);
    log(failure.message);
  }
  throw failure;
}

async function testCompletion(runId, body) {
  // One request answered by Kedi's test model, in Kedi's schema both ways.
  const [system, ...rest] = body.messages[0]?.role === "system" ? body.messages : [null, ...body.messages];
  const format = body.response_format?.json_schema;
  const answer = await request("model/complete", {
    runId,
    model: body.model,
    system: system?.content ?? null,
    messages: rest.map(chatToKedi),
    tools: (body.tools ?? []).map((tool) => ({
      name: tool.function.name,
      description: tool.function.description,
      inputSchema: tool.function.parameters,
    })),
    output: format ? { schema: format.schema, toolName: format.name } : null,
  });
  const parts = answer.message.parts;
  const text = parts.filter((part) => part.type === "text").map((part) => part.text).join("");
  const calls = parts
    .filter((part) => part.type === "toolCall")
    .map((part) => ({
      id: part.id,
      type: "function",
      function: { name: part.name, arguments: JSON.stringify(part.arguments) },
    }));
  const message = { role: "assistant", content: text || null, ...(calls.length ? { tool_calls: calls } : {}) };
  return { choices: [{ message }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } };
}

function chatToKedi(message) {
  // A chat message in Kedi's schema; a test model gets media inline.
  if (message.role !== "user" || typeof message.content === "string") {
    return message.role === "user" ? { role: "user", parts: [{ type: "text", text: message.content }] } : toKedi(message);
  }
  return {
    role: "user",
    parts: message.content.map((part) => {
      if (part.type === "text") return { type: "text", text: part.text };
      const found = /^data:([^;,]+);base64,(.*)$/s.exec(part.image_url.url);
      return found
        ? { type: "media", kind: "image", mediaType: found[1], data: found[2] }
        : { type: "media", kind: "image", url: part.image_url.url };
    }),
  };
}

function reported(runId, reply) {
  const usage = reply.usage ?? {};
  send({
    method: "run/usage",
    params: {
      runId,
      inputTokens: usage.prompt_tokens,
      outputTokens: usage.completion_tokens,
      totalTokens: usage.total_tokens,
    },
  });
  return reply.choices[0].message;
}

function userParts(content) {
  // Kedi's layout of the user turn: text parts and images where their labels put them.
  return content.map((part) => {
    if (part.type === "text") return { type: "text", text: part.text };
    const url = part.data === undefined ? part.url : `data:${part.mediaType};base64,${part.data}`;
    return { type: "image_url", image_url: { url } };
  });
}

// Transcript mode: Kedi holds the conversation in its own schema.

const mediaCache = new Map();

async function mediaFor(runId, ref, current) {
  // The bytes of a conversation's media: this turn's, ones seen before, or Kedi's copy.
  const part = current?.find((item) => item.type === "media" && item.name === ref);
  if (part) return part;
  if (!mediaCache.has(ref)) mediaCache.set(ref, await request("media/get", { runId, ref }));
  return mediaCache.get(ref);
}

async function fromTranscript(runId, history) {
  // Kedi's transcript as chat messages.
  const out = [];
  for (const message of history) {
    const parts = message.parts;
    if (message.role === "user") {
      const content = [];
      for (const part of parts) {
        if (part.type === "text") content.push({ type: "text", text: part.text });
        if (part.type === "media") content.push(...userParts([await mediaFor(runId, part.ref)]));
      }
      const plain = content.length === 1 && content[0].type === "text";
      out.push({ role: "user", content: plain ? content[0].text : content });
    } else if (message.role === "assistant") {
      const text = parts.filter((part) => part.type === "text").map((part) => part.text).join("");
      const calls = parts
        .filter((part) => part.type === "toolCall")
        .map((part) => ({
          id: part.id,
          type: "function",
          function: { name: part.name, arguments: JSON.stringify(part.arguments) },
        }));
      out.push({ role: "assistant", content: text || null, ...(calls.length ? { tool_calls: calls } : {}) });
    } else {
      for (const part of parts) {
        const content = typeof part.result === "string" ? part.result : JSON.stringify(part.result);
        out.push({ role: "tool", tool_call_id: part.id, content });
      }
    }
  }
  return out;
}

function toTranscript(params, turn, withUser) {
  // Chat messages in Kedi's schema. The turn's own user message names its media by ref.
  const [first, ...rest] = turn;
  const user = params.userContent
    ? params.userContent.map((part) =>
        part.type === "text" ? { type: "text", text: part.text } : { type: "media", ref: part.name },
      )
    : [{ type: "text", text: params.user }];
  const head = withUser ? [{ role: "user", parts: user }] : first ? [first].map(toKedi) : [];
  return [...head, ...rest.map(toKedi)];
}

function toKedi(message) {
  // One chat message in Kedi's schema.
  if (message.role === "tool") {
    return { role: "tool", parts: [{ type: "toolResult", id: message.tool_call_id, result: message.content }] };
  }
  const parts = message.content ? [{ type: "text", text: String(message.content) }] : [];
  for (const call of message.tool_calls ?? []) {
    const args = JSON.parse(call.function.arguments || "{}");
    parts.push({ type: "toolCall", id: call.id, name: call.function.name, arguments: args });
  }
  return { role: message.role, parts };
}

function answerText(answer) {
  // A previous answer, replayed before Kedi's correction, as the model wrote it.
  if ("invalid" in answer) return answer.invalid;
  if (!native) return JSON.stringify({ action: "final", ...answer });
  return "text" in answer ? answer.text : JSON.stringify(answer.data);
}

async function callTool(runId, id, name, args) {
  const reply = await request("tool/call", { runId, callId: id, name, arguments: args });
  return JSON.stringify(reply);
}

class McpClient {
  // One stdio MCP server for the run: JSON-RPC, one message per line.
  constructor(server) {
    this.child = spawn(server.command, server.args ?? [], {
      env: { ...process.env, ...(server.env ?? {}) },
      stdio: ["pipe", "pipe", "inherit"],
    });
    this.next = 1;
    this.pending = new Map();
    let buffered = "";
    this.child.stdout.setEncoding("utf8");
    this.child.stdout.on("data", (chunk) => {
      buffered += chunk;
      let end;
      while ((end = buffered.indexOf("\n")) !== -1) {
        const line = buffered.slice(0, end);
        buffered = buffered.slice(end + 1);
        if (!line.trim()) continue;
        const message = JSON.parse(line);
        const waiter = this.pending.get(message.id);
        if (!waiter) continue;
        this.pending.delete(message.id);
        if (message.error) waiter.reject(new Error(message.error.message));
        else waiter.resolve(message.result);
      }
    });
    const fail = (error) => {
      for (const waiter of this.pending.values()) waiter.reject(error);
      this.pending.clear();
    };
    this.child.on("error", fail);
    this.child.on("exit", () => fail(new Error(`MCP server ${server.command} exited`)));
  }

  write(message) {
    this.child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", ...message })}\n`);
  }

  request(method, params) {
    const id = this.next++;
    this.write({ id, method, params });
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }

  async start() {
    const client = { name: "openai-chat-bridge", version: "1" };
    await this.request("initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: client });
    this.write({ method: "notifications/initialized", params: {} });
    return (await this.request("tools/list", {})).tools ?? [];
  }

  async call(name, args) {
    const reply = await this.request("tools/call", { name, arguments: args });
    const text = (reply.content ?? []).filter((part) => part.type === "text").map((part) => part.text).join("\n");
    if (reply.isError) throw new Error(text || `MCP tool ${name} failed`);
    return reply.structuredContent ?? text;
  }

  close() {
    this.child.kill();
  }
}

async function mcpTool(params, call, args, client) {
  // An MCP tool is this bridge's own: Kedi's hooks, approval, budget and artifacts apply.
  const hooks = params.hooks ?? [];
  const base = { runId: params.runId, callId: call.id, tool: call.function.name, origin: "mcp" };
  if (hooks.includes("pre_tool_use")) {
    const pre = await request("hook/invoke", { ...base, event: "pre_tool_use", arguments: args });
    if (pre.decision === "deny") return JSON.stringify({ ok: false, error: pre.reason });
    if (pre.decision === "edit") args = pre.arguments;
  }
  const approval = await request("approval/request", { ...base, arguments: args, risk: "mutating" });
  if (approval.decision === "deny") return JSON.stringify({ ok: false, error: approval.reason });
  if (approval.decision === "edit") args = approval.arguments;
  const admission = await request("budget/admit", { runId: params.runId, dimension: "toolAttempt" });
  if (!admission.admitted) throw new RpcError(RUN_FAILED, admission.reason);
  let result;
  try {
    result = await client.call(call.function.name, args);
  } catch (error) {
    if (hooks.includes("post_tool_use_failure")) {
      const failure = { type: error.name ?? "Error", message: String(error.message ?? error) };
      await request("hook/invoke", { ...base, event: "post_tool_use_failure", arguments: args, error: failure });
    }
    return JSON.stringify({ ok: false, error: String(error.message ?? error) });
  }
  if (hooks.includes("post_tool_use")) {
    await request("hook/invoke", { ...base, event: "post_tool_use", arguments: args, result });
  }
  // A large result reaches the model as Kedi's artifact reference.
  if (params.admitResults) result = (await request("artifact/admit", { ...base, result })).result;
  return JSON.stringify({ ok: true, result });
}

async function run(params, state) {
  // Connects the run's MCP servers, converses, and closes them.
  const clients = [];
  const mcp = new Map();
  const definitions = [];
  try {
    for (const server of params.mcpServers ?? []) {
      const client = new McpClient(server);
      clients.push(client);
      for (const tool of await client.start()) {
        if (mcp.has(tool.name) || params.tools.some((kedi) => kedi.name === tool.name)) {
          throw new RpcError(RUN_FAILED, `MCP tool ${tool.name} has the name of another tool`);
        }
        mcp.set(tool.name, client);
        definitions.push({
          type: "function",
          function: { name: tool.name, description: tool.description ?? "", parameters: tool.inputSchema },
        });
      }
    }
    return await converse(params, state, mcp, definitions);
  } finally {
    for (const client of clients) client.close();
  }
}

async function converse(params, state, mcp, definitions) {
  const signal = state.controller.signal;
  const messages = [];
  if (params.system) messages.push({ role: "system", content: params.system });
  // Kedi holds the transcript, or, in a session, hands back the state this bridge returned.
  const transcript = Array.isArray(params.history);
  const history = transcript
    ? await fromTranscript(params.runId, params.history)
    : (params.session?.state?.messages ?? []);
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
  const models = [params.model ?? "default", ...(params.fallbackModels ?? [])];
  if (native && params.tools.length + definitions.length) {
    body.tools = params.tools.map((tool) => ({
      type: "function",
      function: { name: tool.name, description: tool.description ?? "", parameters: tool.inputSchema },
    }));
    body.tools.push(...definitions);
  }
  const media = params.output?.media;
  if (media === "image") body.modalities = ["image", "text"];
  if (media === "audio") {
    body.modalities = ["text", "audio"];
    body.audio = { voice: params.output.voice ?? "alloy", format: "wav" };
  }
  // A drawing answers with an image; one that may name a request image writes its name.
  if (native && params.output && !media) {
    body.response_format = {
      type: "json_schema",
      json_schema: { name: params.output.toolName, schema: params.output.schema },
    };
  }
  // With a history processor, Kedi's processed conversation replaces ours before each request:
  // base is the processed part in Kedi's schema, and messages from mark on follow it.
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
  // What an interrupted attempt hands Kedi: the conversation so far, for the next attempt.
  state.partial = () => {
    if (params.processHistory) return { transcript: [...base, ...kediTurn(messages.slice(mark))] };
    if (transcript) return { messages: toTranscript(params, messages.slice(turnStart), true) };
    return params.session ? { state: { messages: [...history, ...messages.slice(turnStart)] } } : null;
  };
  const processHistory = async () => {
    const current = [...base, ...kediTurn(messages.slice(mark))];
    base = (await request("history/process", { runId: params.runId, messages: current })).messages;
    userPending = false;
    const system = params.system ? 1 : 0;
    messages.splice(system, messages.length - system, ...(await fromTranscript(params.runId, base)));
    mark = messages.length;
  };
  for (let step = 0; step < maxSteps; step++) {
    // Live input from Kedi joins the conversation before the next model request.
    const delivering = state.inputs.splice(0);
    for (const { texts } of delivering) for (const text of texts) messages.push({ role: "user", content: text });
    if (params.processHistory) await processHistory();
    const message = await complete(params.runId, body, models, signal);
    for (const { id } of delivering) send({ id, result: { delivered: true } });
    if (native) {
      const calls = message.tool_calls ?? [];
      if (!calls.length) {
        if (media) return finish(produced(media, message, params.output), message.content ?? "");
        if (!params.output) return finish({ text: message.content ?? "" }, message.content ?? "");
        try {
          return finish({ data: JSON.parse(message.content) }, message.content);
        } catch (error) {
          return { invalid: message.content ?? "", error: String(error) };
        }
      }
      if (message.content) {
        // The model's text alongside its tool calls, for Kedi's event observers.
        send({
          method: "run/event",
          params: {
            runId: params.runId,
            type: "message.completed",
            messageId: `step-${step}`,
            text: message.content,
            phase: "commentary",
          },
        });
      }
      messages.push({ role: "assistant", content: message.content ?? null, tool_calls: calls });
      for (const call of calls) {
        const args = JSON.parse(call.function.arguments || "{}");
        const client = mcp.get(call.function.name);
        const content = client
          ? await mcpTool(params, call, args, client)
          : await callTool(params.runId, call.id, call.function.name, args);
        messages.push({ role: "tool", tool_call_id: call.id, content });
      }
      continue;
    }
    let action;
    try {
      action = JSON.parse(message.content);
    } catch (error) {
      return { invalid: message.content ?? "", error: String(error) };
    }
    if (action.action === "final") {
      const result = params.output ? { data: action.data } : { text: action.text };
      return finish(result, message.content);
    }
    if (action.action !== "call_tool") {
      return { invalid: message.content, error: `unknown action ${JSON.stringify(action.action)}` };
    }
    messages.push({ role: "assistant", content: message.content });
    const content = await callTool(params.runId, action.callId, action.name, action.arguments ?? {});
    messages.push({ role: "tool", tool_call_id: action.callId, content });
  }
  throw new RpcError(RUN_FAILED, `no final answer after ${maxSteps} model requests`);
}

function produced(media, message, output) {
  // The run's answer for a media output: the image or speech the model made, or, for an image
  // that may name one of the request, the name it wrote.
  const text = message.content ? [{ type: "text", text: message.content }] : [];
  if (media === "audio" && message.audio?.data) {
    const part = { type: "media", kind: "audio", mediaType: "audio/wav", data: message.audio.data };
    return { content: [...text, part] };
  }
  const url = message.images?.[0]?.image_url?.url;
  if (media === "image" && url) {
    const found = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    const part = found
      ? { type: "media", kind: "image", mediaType: found[1], data: found[2] }
      : { type: "media", kind: "image", url };
    return { content: [...text, part] };
  }
  if (media === "image" && output.schema && message.content) {
    try {
      return { data: JSON.parse(message.content) };
    } catch {
      return { data: message.content.trim() };
    }
  }
  throw new RpcError(RUN_FAILED, `the model gave no ${media === "audio" ? "speech" : "image"}`);
}

async function handle(message) {
  const { id, method, params } = message;
  if (method === "initialize") {
    testMode = params.testModel === true;
    send({
      id,
      result: {
        protocolVersion: PROTOCOL_VERSION,
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
          testModel: true,
        },
        settings: Object.keys(SETTINGS),
        effort: EFFORT,
        mediaKinds: ["image"],
        mcp: { transports: native ? ["stdio"] : [] },
      },
    });
    return;
  }
  if (method === "run") {
    const state = { controller: new AbortController(), inputs: [], interrupted: false, partial: () => null };
    runs.set(params.runId, state);
    try {
      send({ id, result: await run(params, state) });
    } catch (error) {
      if (state.interrupted) {
        send({ id, error: { code: INTERRUPTED, message: "interrupted", data: state.partial() ?? {} } });
      } else {
        const code = state.controller.signal.aborted ? CANCELLED : (error.code ?? RUN_FAILED);
        log("run failed:", error.message);
        send({ id, error: { code, message: error.message } });
      }
    } finally {
      runs.delete(params.runId);
      // Input the model never saw goes back to Kedi for the next attempt.
      for (const { id: inputId } of state.inputs) send({ id: inputId, result: { delivered: false } });
    }
    return;
  }
  if (method === "run/input") {
    const state = runs.get(params.runId);
    if (!state) {
      send({ id, result: { delivered: false } });
      return;
    }
    state.inputs.push({ id, texts: params.items.map((item) => item.text) });
    return;
  }
  if (method === "shutdown") {
    send({ id, result: {} });
    process.stdout.end(() => process.exit(0));
    return;
  }
  send({ id, error: { code: METHOD_NOT_FOUND, message: `unknown method ${method}` } });
}

function receive(line) {
  if (!line.trim()) return;
  const message = JSON.parse(line);
  if (message.method === "run/cancel") {
    runs.get(message.params.runId)?.controller.abort();
    return;
  }
  if (message.method === "run/interrupt") {
    const state = runs.get(message.params.runId);
    if (state) {
      state.interrupted = true;
      state.controller.abort();
    }
    return;
  }
  if (message.method !== undefined) {
    handle(message);
    return;
  }
  const waiter = pending.get(message.id);
  if (!waiter) return;
  pending.delete(message.id);
  if (message.error) waiter.reject(new RpcError(message.error.code, message.error.message));
  else waiter.resolve(message.result);
}

// Frames end at LF only. node:readline also ends lines at CR, U+2028 and U+2029, which JSON
// strings may hold unescaped, so it would split a frame whose text has one.
let buffered = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buffered += chunk;
  let end;
  while ((end = buffered.indexOf("\n")) !== -1) {
    const line = buffered.slice(0, end);
    buffered = buffered.slice(end + 1);
    receive(line);
  }
});
process.stdin.on("end", () => process.exit(0));
