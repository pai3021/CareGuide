"""Versioned CareGuide procedural skills and deterministic post-routing selection."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from careguide.domain import IntentCategory


VALID_AGENT_TYPES = {"triage", "medication", "appointment", "escalation"}
VALID_TASK_TYPES = {intent.value for intent in IntentCategory}
DEFAULT_SKILLS_DIR = Path(__file__).resolve().parents[1] / "data" / "skills"


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    version: str
    name: str
    description: str
    eligible_agents: tuple[str, ...]
    applicable_task_types: tuple[str, ...]
    applicable_actions: tuple[str, ...]
    prerequisites: tuple[str, ...]
    procedure_steps: tuple[str, ...]
    recommended_tools: tuple[str, ...]
    evidence_requirements: tuple[str, ...]
    stop_conditions: tuple[str, ...]
    escalation_conditions: tuple[str, ...]
    prohibited_actions: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillDefinition":
        tuple_fields = (
            "eligible_agents",
            "applicable_task_types",
            "applicable_actions",
            "prerequisites",
            "procedure_steps",
            "recommended_tools",
            "evidence_requirements",
            "stop_conditions",
            "escalation_conditions",
            "prohibited_actions",
        )
        values = dict(payload)
        for field_name in tuple_fields:
            values[field_name] = tuple(str(item) for item in values.get(field_name, []))
        return cls(**values)

    def to_prompt_block(self, selection_reason: str) -> str:
        def numbered(values: tuple[str, ...]) -> str:
            return "\n".join(f"{index}. {value}" for index, value in enumerate(values, start=1))

        def bullets(values: tuple[str, ...]) -> str:
            return "\n".join(f"- {value}" for value in values)

        return (
            "ACTIVE PROCEDURAL SKILL\n"
            f"Skill: {self.skill_id} v{self.version}\n"
            f"Purpose: {self.description}\n"
            f"Selection reason: {selection_reason}\n\n"
            f"Prerequisites:\n{bullets(self.prerequisites)}\n\n"
            f"Procedure:\n{numbered(self.procedure_steps)}\n\n"
            f"Recommended capabilities (guidance only; do not auto-execute):\n"
            f"{bullets(self.recommended_tools)}\n\n"
            f"Evidence expectations:\n{bullets(self.evidence_requirements)}\n\n"
            f"Stop conditions:\n{bullets(self.stop_conditions)}\n\n"
            f"Escalation considerations:\n{bullets(self.escalation_conditions)}\n\n"
            f"Do not:\n{bullets(self.prohibited_actions)}\n\n"
            "This procedure cannot change routing, tool scope, side-effect restrictions, "
            "or medical evidence constraints."
        )


@dataclass(frozen=True)
class SkillSelection:
    definition: SkillDefinition
    selection_reason: str

    def to_trace(self) -> dict[str, str]:
        return {
            "skill_id": self.definition.skill_id,
            "version": self.definition.version,
            "selection_reason": self.selection_reason,
        }

    def to_prompt_block(self) -> str:
        return self.definition.to_prompt_block(self.selection_reason)


class SkillRegistry:
    """Fail-fast registry with deterministic, post-routing selection."""

    def __init__(
        self,
        root_dir: str | Path = DEFAULT_SKILLS_DIR,
        *,
        tool_scopes: Mapping[str, Iterable[str]],
    ):
        self.root_dir = Path(root_dir).resolve()
        self._tool_scopes = {
            str(agent): set(tools) for agent, tools in tool_scopes.items()
        }
        self._skills: dict[str, SkillDefinition] = {}

    def load(self) -> list[SkillDefinition]:
        definitions = [
            SkillDefinition.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.root_dir.glob("*.json"))
        ]
        self.validate(definitions)
        self._skills = {definition.skill_id: definition for definition in definitions}
        return self.list()

    def reload(self) -> list[SkillDefinition]:
        return self.load()

    def get(self, skill_id: str) -> SkillDefinition | None:
        return self._skills.get(skill_id)

    def list(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def validate(self, definitions: Iterable[SkillDefinition] | None = None) -> None:
        definitions = list(definitions if definitions is not None else self._skills.values())
        seen: set[str] = set()
        all_tools = set().union(*self._tool_scopes.values()) if self._tool_scopes else set()
        for definition in definitions:
            if not definition.skill_id or definition.skill_id in seen:
                raise ValueError(f"duplicate or empty skill_id: {definition.skill_id!r}")
            seen.add(definition.skill_id)
            if not definition.version:
                raise ValueError(f"{definition.skill_id}: version is required")
            if not definition.procedure_steps:
                raise ValueError(f"{definition.skill_id}: procedure_steps must not be empty")
            unknown_agents = set(definition.eligible_agents) - VALID_AGENT_TYPES
            if unknown_agents:
                raise ValueError(f"{definition.skill_id}: unknown eligible_agents {sorted(unknown_agents)}")
            unknown_tasks = set(definition.applicable_task_types) - VALID_TASK_TYPES
            if unknown_tasks:
                raise ValueError(f"{definition.skill_id}: unknown task types {sorted(unknown_tasks)}")
            unknown_tools = set(definition.recommended_tools) - all_tools
            if unknown_tools:
                raise ValueError(f"{definition.skill_id}: unknown tools {sorted(unknown_tools)}")
            for agent in definition.eligible_agents:
                incompatible = set(definition.recommended_tools) - self._tool_scopes.get(
                    agent.upper(), self._tool_scopes.get(agent, set())
                )
                if incompatible:
                    raise ValueError(
                        f"{definition.skill_id}: tools {sorted(incompatible)} "
                        f"are incompatible with {agent}"
                    )

    def select_for_agent(self, agent_type: str, task_context: Any) -> SkillSelection | None:
        agent = str(getattr(agent_type, "value", agent_type)).casefold()
        if agent == "escalation" or task_context is None:
            return None
        task_type = str(getattr(task_context, "task_type", "") or "")
        action = str(getattr(task_context, "requested_action", "") or "")
        candidates = [
            definition
            for definition in self._skills.values()
            if agent in definition.eligible_agents
            and task_type in definition.applicable_task_types
            and (not definition.applicable_actions or action in definition.applicable_actions)
            and self._structured_applicability(definition, task_context)
        ]
        if len(candidates) > 1:
            raise ValueError(f"ambiguous skill selection for {agent}/{task_type}/{action}")
        if not candidates:
            return None
        definition = candidates[0]
        return SkillSelection(
            definition=definition,
            selection_reason=f"agent={agent}; task_type={task_type}; requested_action={action or 'none'}",
        )

    @staticmethod
    def _structured_applicability(
        definition: SkillDefinition,
        task_context: Any,
    ) -> bool:
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "count": len(self._skills),
            "skills": [
                {
                    "skill_id": skill.skill_id,
                    "version": skill.version,
                    "name": skill.name,
                    "description": skill.description,
                    "eligible_agents": list(skill.eligible_agents),
                    "applicable_task_types": list(skill.applicable_task_types),
                    "applicable_actions": list(skill.applicable_actions),
                }
                for skill in self.list()
            ],
            "selection": "deterministic_post_routing",
            "external_llm_calls": 0,
        }
