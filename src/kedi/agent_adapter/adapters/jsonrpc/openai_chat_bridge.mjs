#!/usr/bin/env node
// Kedi JSON-RPC bridge (protocol version 1) for an OpenAI-compatible chat endpoint.
//
// Kedi renders every model-visible text; this bridge transports it. It sends Kedi's system
// text and prompt verbatim, Kedi's tools and output schema as native fields (--mode native)
// or nothing extra (--mode protocol: Kedi's protocol parts are already in the system text),
// and calls Kedi tools back with tool/call. It writes no instructions of its own.
//
// Usage: node openai_chat_bridge.mjs --base-url URL [--mode native|protocol]
//        [--api-key-env NAME] [--max-steps N]
// stdout carries only protocol frames; diagnostics go to stderr.

import { createInterface } from "node:readline";

const PROTOCOL_VERSION = 1;
const RUN_FAILED = -32002;
const CANCELLED = -32003;
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

async function complete(body, signal) {
  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ ...body, stream: false }),
    signal,
  });
  if (!response.ok) {
    throw new RpcError(RUN_FAILED, `model endpoint answered ${response.status}: ${await response.text()}`);
  }
  return (await response.json()).choices[0].message;
}

function userParts(content) {
  // Kedi's layout of the user turn: text parts and images where their labels put them.
  return content.map((part) => {
    if (part.type === "text") return { type: "text", text: part.text };
    const url = part.data === undefined ? part.url : `data:${part.mediaType};base64,${part.data}`;
    return { type: "image_url", image_url: { url } };
  });
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

async function run(params, signal) {
  const messages = [];
  if (params.system) messages.push({ role: "system", content: params.system });
  // In a session, Kedi hands back the state this bridge returned for the last turn.
  const history = params.session?.state?.messages ?? [];
  messages.push(...history);
  const turnStart = messages.length;
  messages.push({ role: "user", content: params.userContent ? userParts(params.userContent) : params.user });
  if (params.continuation) {
    messages.push({ role: "assistant", content: answerText(params.continuation.answer) });
    messages.push({ role: "user", content: params.continuation.message });
  }
  const body = { model: params.model ?? "default", messages };
  if (native && params.tools.length) {
    body.tools = params.tools.map((tool) => ({
      type: "function",
      function: { name: tool.name, description: tool.description ?? "", parameters: tool.inputSchema },
    }));
  }
  if (native && params.output) {
    body.response_format = {
      type: "json_schema",
      json_schema: { name: params.output.toolName, schema: params.output.schema },
    };
  }
  const finish = (result, content) => {
    if (!params.session) return result;
    const turn = [...messages.slice(turnStart), { role: "assistant", content }];
    return { ...result, state: { messages: [...history, ...turn] } };
  };
  for (let step = 0; step < maxSteps; step++) {
    const message = await complete(body, signal);
    if (native) {
      const calls = message.tool_calls ?? [];
      if (!calls.length) {
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
        const content = await callTool(
          params.runId,
          call.id,
          call.function.name,
          JSON.parse(call.function.arguments || "{}"),
        );
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

async function handle(message) {
  const { id, method, params } = message;
  if (method === "initialize") {
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
          events: true,
          mediaInput: true,
          sessions: true,
        },
        mediaKinds: ["image"],
      },
    });
    return;
  }
  if (method === "run") {
    const controller = new AbortController();
    runs.set(params.runId, controller);
    try {
      send({ id, result: await run(params, controller.signal) });
    } catch (error) {
      const code = controller.signal.aborted ? CANCELLED : (error.code ?? RUN_FAILED);
      log("run failed:", error.message);
      send({ id, error: { code, message: error.message } });
    } finally {
      runs.delete(params.runId);
    }
    return;
  }
  if (method === "shutdown") {
    send({ id, result: {} });
    process.stdout.end(() => process.exit(0));
    return;
  }
  send({ id, error: { code: METHOD_NOT_FOUND, message: `unknown method ${method}` } });
}

const lines = createInterface({ input: process.stdin });
lines.on("line", (line) => {
  if (!line.trim()) return;
  const message = JSON.parse(line);
  if (message.method === "run/cancel") {
    runs.get(message.params.runId)?.abort();
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
});
lines.on("close", () => process.exit(0));
