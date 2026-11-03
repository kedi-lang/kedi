Always read and understand docs/KEDI_DSL.md before working with the code. 
Before changing how async work runs from synchronous code (run_sync, event loops, operation scopes, Codex tool bridges), read docs/EVENT_LOOPS.md: its design and invariants are deliberate.
Look at the root directory for virtual environments, and activate them before running any code.
When running multi-line test scripts, always use python << 'PYTHON_EOF' ... PYTHON_EOF.
Add as many debug traces as possible. REMOVE ALL DEBUG TRACES AFTER FINALIZING YOUR WORK.
Keep the architecture clean and concise, if you encounter unnecessary logic, or unnecessary defenses, just refactor them. 
This project requires advanced reasoning abilities and extended thinking before taking action. 
Understand the object relationships and the context of the codebase before you proceed. 