# Skills from Python

Skills expose discoverable instruction files. They do not grant filesystem, tool, or approval permissions.

## Enable Skills

Enable scoped skills explicitly:

```python
kedi.configure(skills=True)
```

or for one callable:

```python
@kedi.query(skills=True)
def solve(task: str) -> str:
    """kedi
    >> Use an applicable project skill. The solution to <task> is [answer: str].
    = `answer`
    """
    ...
```

Kedi checks the user Kedi registry, project-local `.agents/skills`, then the
user-global `.agents/skills` directory and exposes two read-only tools:

- `list_skills(all=False, limit=20)`;
- `read_skill(name)`.

Enabling skills does not preload every `SKILL.md`; the agent discovers and
reads only relevant entries.

Pass `SkillsSettings` for the same policy controls as the expanded DSL
directive:

```python
from pathlib import Path

import kedi
from kedi import SkillsSettings

kedi.configure(
    skills=SkillsSettings(
        enabled=True,
        cwd=Path("workspace"),
        max_skills=40,
        include_registry=True,
        include_all=False,
        exclude_paths=(Path("~/.agents/skills"),),
    )
)
```
