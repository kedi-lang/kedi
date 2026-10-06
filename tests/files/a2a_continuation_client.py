"""Isolated official A2A client; local certificate trust never changes the host."""

import asyncio
import json
import os
import sys

from kedi.a2a.protocol import A2ARemoteTaskError
from kedi.agent_adapter.adapters.a2a import A2AAdapter
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_profile import A2AAuthConfig, A2AConnection


async def probe(data):
    adapter = A2AAdapter(
        connection=A2AConnection(
            endpoint=data["endpoint"],
            auth=A2AAuthConfig(scheme="bearer", secret_env="KEDI_TEST_A2A_SECRET"),
            request_timeout=120,
            task_timeout=150,
        )
    )
    evidence = {}
    try:
        if data["operation"] == "invoke":
            evidence["output"] = await adapter.invoke(prompt=data["prompt"])
        else:
            first = AdapterConversationContext()
            try:
                with adapter.conversation_scope(first):
                    await adapter.invoke(
                        prompt="Start a receipt request and ask for confirmation first."
                    )
            except A2ARemoteTaskError as exc:
                evidence["suspended"] = {
                    "state": exc.state,
                    "task_id": exc.task_id,
                    "context_id": exc.context_id,
                }
                snapshot = await adapter.get_task(exc.task_id)
                evidence["snapshot_state"] = snapshot.state
                if data["operation"] == "explicit":
                    evidence["output"] = await adapter.continue_task(
                        task_id=exc.task_id, context_id=exc.context_id, prompt=data["prompt"]
                    )
                else:
                    resumed = AdapterConversationContext(resume_state=first.result_state)
                    with adapter.conversation_scope(resumed):
                        evidence["output"] = await adapter.invoke(prompt=data["prompt"])
                    evidence["resume_state"] = resumed.result_state
                final = await adapter.get_task(exc.task_id)
                evidence["final"] = {
                    "task_id": final.task_id,
                    "context_id": final.context_id,
                    "state": final.state,
                    "text": final.text,
                }
            else:
                raise AssertionError("Peer did not suspend before inference")
    except Exception as exc:
        chain = []
        current = exc
        while current is not None and len(chain) < 8:
            chain.append(
                {
                    "type": type(current).__name__,
                    "message": str(current).replace(
                        os.environ["KEDI_TEST_A2A_SECRET"], "[REDACTED]"
                    ),
                }
            )
            current = current.__cause__ or current.__context__
        evidence["error"] = chain
    return evidence


if __name__ == "__main__":
    print(json.dumps(asyncio.run(probe(json.loads(sys.stdin.read())))), flush=True)
