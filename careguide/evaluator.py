"""
亮点：端到端 Agent 评测框架

核心问题：如何评测端到端 Agent？

评测维度：
  1. 意图识别准确率 —— 预测意图 vs 标注意图，计算 Accuracy / F1
  2. 响应质量评分 —— 用 LLM 作为评判者（LLM-as-Judge），
     从相关性、准确性、完整性、有用性四个维度打分
  3. 端到端对话评测 —— 模拟完整多轮对话，评估整体体验
  4. 回归测试 —— 与历史基线对比，防止性能退化

LLM-as-Judge 是评测 Agent 质量的关键技术：
  人工标注成本高、主观性强；用 LLM 评判可以规模化、可重复。
"""
import asyncio
import json
import logging
import pathlib
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from careguide.llm_utils import create_chat_completion, create_llm_client, extract_text_content

from careguide.intent_recognizer import IntentCategory, IntentRecognizer
from careguide.trace_store import validate_trace_completeness

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class IntentTestCase:
    message:          str
    expected_intent:  str
    context:          Optional[Dict[str, Any]] = None


@dataclass
class QualityScores:
    """LLM-as-Judge 评分结果。"""
    relevance:    float   # 相关性：回答是否针对问题
    accuracy:     float   # 准确性：信息是否正确
    completeness: float   # 完整性：是否完整解决问题
    helpfulness:  float   # 有用性：用户是否能据此行动
    judge_failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        return statistics.mean([self.relevance, self.accuracy, self.completeness, self.helpfulness])


@dataclass
class EvalResult:
    test_id:    str
    passed:     Optional[bool]  # None means the quality judge did not score this case.
    scores:     Dict[str, float]
    detail:     str = ""
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """评测报告。"""
    timestamp:        str
    total:            int
    passed:           int
    pass_rate:        Optional[float]  # Computed only over scored results.
    avg_scores:       Dict[str, float]
    regressions:      List[str]          # 相比基线退化的指标
    recommendations:  List[str]
    results:          List[EvalResult]
    evaluation_layers: Dict[str, Any] = field(default_factory=dict)
    scored:           int = 0
    unscored:         int = 0


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

class LLMJudge:
    """
    用 LLM 评判 Agent 响应质量。

    为什么用 LLM 而不是人工？
    - 可规模化：数千条测试用例自动评测
    - 可重复：相同输入得到稳定评分
    - 多维度：同时评估相关性、准确性等多个维度

    注意：LLM Judge 本身也有偏差，建议定期用人工标注校准。
    """

    JUDGE_PROMPT = """你是一个医疗资料查询 Agent 的证据一致性评估员；不评价临床诊断正确性。请基于用户问题、Agent 响应和执行证据评分。

用户问题: {question}
Agent 响应: {response}
{context_section}
{evidence_section}

请从以下四个维度评分（0.0-1.0），返回 JSON：
- relevance: 响应是否直接针对用户问题（0=完全无关，1=完全相关）
- accuracy: 响应是否与工具结果和执行证据一致，是否避免编造事实（0=明显冲突或编造，1=完全有证据支撑）
- completeness: 是否完整解决了用户需求（0=完全没解决，1=完全解决）
- helpfulness: 用户能否据此采取行动（0=毫无帮助，1=非常有帮助）

评分时遵守以下工作流规则：
- 如果执行证据包含 case_contract，必须按契约评分。answerable=true 时，只有 actual_evidence_ids 命中
  target_evidence_ids 且回答受该证据支持，accuracy 和 completeness 才能得高分；answerable=false 时，
  在不返回证据、不编造答案的前提下作范围内拒答或询问必要字段，属于正确完成。
- 如果执行证据显示 final_status=needs_clarification（或NEED_CLARIFICATION），且 task_understanding.missing_fields 非空，
  Agent 明确询问了这些必要字段，就视为当前轮已经正确完成，不要因为没有在信息不足时强行解决任务而降低 completeness。
- 对问候或尚无具体任务的消息，简洁回应或说明能力边界可以是完整回答。
- 如果case_contract.expected.boundary与final_status一致，急症升级、拒绝诊断、拒绝个体化用药或拒绝
  范围外请求就是正确完成；不得因为Agent没有回答被限制的原始请求而降低relevance、completeness或helpfulness。
- 对pending_confirmation，完整说明模拟操作及待确认状态即为本轮完成；不得要求确认前已经写入预约。
- 只有追问与缺失字段无关、重复询问已有信息，或本可执行却无故追问时，才降低相关性、完整性和有用性。

只返回 JSON，例如: {{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.7, "helpfulness": 0.85}}"""

    def __init__(
        self,
        client: Optional[Any],
        model: str,
        *,
        thinking_enabled: Optional[bool] = None,
        json_mode: bool = False,
    ):
        self._client = client
        self._model  = model
        self._thinking_enabled = thinking_enabled
        self._json_mode = json_mode

    @property
    def configured(self) -> bool:
        return self._client is not None

    @property
    def model(self) -> str:
        return self._model

    async def judge(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> QualityScores:
        if self._client is None:
            return QualityScores(
                0.5, 0.5, 0.5, 0.5,
                judge_failed=True,
                error="JUDGE_API_KEY 未配置，已跳过 LLM-as-Judge",
            )
        ctx_section = f"背景信息: {context}" if context else ""
        evidence_section = ""
        if evidence:
            serialized = json.dumps(evidence, ensure_ascii=False, default=str)
            evidence_section = f"执行证据（RequestTrace 摘要）: {serialized[:12000]}"
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=ctx_section,
            evidence_section=evidence_section,
        )
        prompt = self._clean_text(prompt)
        try:
            extra_body = None
            if self._thinking_enabled is not None:
                extra_body = {
                    "thinking": {
                        "type": "enabled" if self._thinking_enabled else "disabled",
                    },
                }
            resp = await create_chat_completion(
                self._client,
                model=self._model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"} if self._json_mode else None,
                extra_body=extra_body,
            )
            raw = extract_text_content(resp)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            return QualityScores(
                relevance=float(data.get("relevance", 0.5)),
                accuracy=float(data.get("accuracy", 0.5)),
                completeness=float(data.get("completeness", 0.5)),
                helpfulness=float(data.get("helpfulness", 0.5)),
            )
        except Exception as ex:
            logger.warning(f"LLM Judge 失败: {ex}")
            return QualityScores(
                0.5, 0.5, 0.5, 0.5,
                judge_failed=True,
                error=str(ex),
            )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")


# ── 意图识别评测 ──────────────────────────────────────────────────────────────

class IntentEvaluator:
    """评测意图识别的准确率和 F1。"""

    def __init__(self, recognizer: IntentRecognizer):
        self._recognizer = recognizer

    async def evaluate(self, cases: List[IntentTestCase], *, concurrency: int = 1,
                       progress=None) -> Dict[str, Any]:
        if concurrency < 1:
            raise ValueError('concurrency must be positive')
        predictions, ground_truth = [], []
        case_details: List[Dict[str, Any]] = []
        semaphore = asyncio.Semaphore(concurrency)
        completed = 0
        async def recognize(case):
            nonlocal completed
            async with semaphore:
                result = await self._recognizer.recognize(case.message, (case.context or {}).get('history'))
                completed += 1
                if progress:
                    progress(completed, len(cases))
                return result
        results = await asyncio.gather(*(recognize(case) for case in cases))
        for case, result in zip(cases, results):
            predicted = result.intent.value if result.intent else result.disposition
            predictions.append(predicted)
            ground_truth.append(case.expected_intent)
            case_details.append({
                "message": case.message,
                "expected": case.expected_intent,
                "predicted": predicted,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "source_scores": result.source_scores,
                "fine_intent": result.fine_intent,
                "urgency": result.urgency,
            })

        # 纯 Python 计算指标
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions) if predictions else 0.0

        # 每类 F1
        labels = sorted(set(ground_truth + predictions))
        per_class: Dict[str, Dict[str, float]] = {}
        for label in labels:
            tp = sum(p == label and g == label for p, g in zip(predictions, ground_truth))
            fp = sum(p == label and g != label for p, g in zip(predictions, ground_truth))
            fn = sum(p != label and g == label for p, g in zip(predictions, ground_truth))
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[label] = {"precision": prec, "recall": rec, "f1": f1}

        macro_f1 = statistics.mean(v["f1"] for v in per_class.values()) if per_class else 0.0

        return {
            "accuracy":   round(accuracy, 4),
            "macro_f1":   round(macro_f1, 4),
            "per_class":  per_class,
            "total":      len(cases),
            "correct":    correct,
            "cases":      case_details,
        }


# ── 端到端评测器 ──────────────────────────────────────────────────────────────

class EndToEndEvaluator:
    """
    端到端 Agent 评测。

    评测流程：
      1. 运行意图识别评测（准确率/F1）
      2. 运行对话质量评测（LLM-as-Judge）
      3. 与历史基线对比（回归检测）
      4. 生成可操作的优化建议
    """

    # 质量及格线
    PASS_THRESHOLD = 0.75

    def __init__(
        self,
        orchestrator,
        recognizer: IntentRecognizer,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "mimo-v2.5",
        baseline_path: Optional[str] = None,
        client: Optional[Any] = None,
        judge_api_key: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        judge_model: Optional[str] = None,
        judge_client: Optional[Any] = None,
        judge_enabled: Optional[bool] = None,
        judge_thinking: Optional[bool] = None,
        judge_json_mode: bool = False,
    ):
        if judge_enabled is False:
            resolved_judge_client = None
        elif judge_client is not None:
            resolved_judge_client = judge_client
        elif judge_api_key:
            resolved_judge_client = create_llm_client(judge_api_key, judge_base_url)
        elif client is not None:
            resolved_judge_client = client
        elif api_key:
            # Backward compatibility for callers that intentionally share one provider.
            resolved_judge_client = create_llm_client(api_key, base_url)
        else:
            resolved_judge_client = None

        self._orchestrator     = orchestrator
        self._judge            = LLMJudge(
            resolved_judge_client,
            judge_model or model,
            thinking_enabled=judge_thinking,
            json_mode=judge_json_mode,
        )
        self._intent_evaluator = IntentEvaluator(recognizer)
        self._history:         List[EvalReport] = []
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None
        self._baseline: Optional[EvalReport] = self._load_baseline()

    async def run(
        self,
        intent_cases:    Optional[List[IntentTestCase]] = None,
        dialog_cases:    Optional[List[Dict[str, Any]]] = None,
    ) -> EvalReport:
        """
        运行完整评测。

        intent_cases: 意图识别测试用例
        dialog_cases:
          - 单轮: [{"question": "..."}]
          - 多轮: [{"turns": ["第一轮", "第二轮", ...]}]
        """
        results: List[EvalResult] = []
        all_scores: Dict[str, List[float]] = {
            "relevance": [], "accuracy": [], "completeness": [], "helpfulness": []
        }

        # 1. 意图识别评测
        intent_metrics: Dict[str, Any] = {}
        if intent_cases:
            intent_metrics = await self._intent_evaluator.evaluate(intent_cases)
            passed = intent_metrics["accuracy"] >= self.PASS_THRESHOLD
            results.append(EvalResult(
                test_id="intent_recognition",
                passed=passed,
                scores={"accuracy": intent_metrics["accuracy"], "macro_f1": intent_metrics["macro_f1"]},
                detail=f"准确率 {intent_metrics['accuracy']:.1%}，Macro-F1 {intent_metrics['macro_f1']:.3f}",
                metadata={
                    "total": intent_metrics.get("total", 0),
                    "correct": intent_metrics.get("correct", 0),
                    "cases": intent_metrics.get("cases", []),
                },
            ))

        # 2. 对话质量评测（调用 orchestrator 产出回复，再用 LLM Judge 评分）
        if dialog_cases:
            for i, case in enumerate(dialog_cases):
                case_results = await self._evaluate_dialog_case(case, i)
                results.extend(case_results)
                for r in case_results:
                    for k in all_scores:
                        if k in r.scores:
                            all_scores[k].append(r.scores[k])

        # 3. 汇总
        avg_scores = {
            k: round(statistics.mean(v), 4) for k, v in all_scores.items() if v
        }
        if intent_metrics:
            avg_scores["intent_accuracy"] = intent_metrics["accuracy"]

        scored_count = sum(r.passed is not None for r in results)
        passed_count = sum(r.passed is True for r in results)
        pass_rate = round(passed_count / scored_count, 4) if scored_count else None

        # 4. 回归检测
        regressions = self._detect_regressions(avg_scores)

        # 5. 优化建议
        recommendations = self._recommendations(avg_scores, intent_metrics)
        unscored_count = len(results) - scored_count
        if unscored_count:
            recommendations = [item for item in recommendations if item != "所有指标均达标，继续保持"]
            recommendations.append(f"{unscored_count} 轮对话未评分；检查 Judge 配置或运行状态")

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed_count,
            pass_rate=pass_rate,
            avg_scores=avg_scores,
            regressions=regressions,
            recommendations=recommendations,
            results=results,
            evaluation_layers=self._summarize_evaluation_layers(results),
            scored=scored_count,
            unscored=unscored_count,
        )
        self._history.append(report)
        # Baseline promotion requires an explicit separate operation.
        return report

    async def _evaluate_dialog_case(self, case: Dict[str, Any], case_idx: int) -> List[EvalResult]:
        """评测单轮或多轮对话用例。"""

        questions = self._dialog_turns(case)
        if not questions:
            return []

        conv_id = str(case.get("conv_id") or f"eval_{case_idx}")
        user_id = str(case.get("user_id") or "eval_user")
        history: List[Dict[str, str]] = []
        results: List[EvalResult] = []

        for turn_idx, question in enumerate(questions):
            context = self._history_context(history)
            from types import SimpleNamespace
            payload = await self._orchestrator.process(question, user_id, conv_id, history[-6:])
            orch_result = SimpleNamespace(**payload)
            orch_result.intent = IntentCategory(payload['intent']) if payload['intent'] else None
            actual_answer = payload['response']
            trace = self._orchestrator.traces.get(payload['request_id'])
            trace_evaluation = self._evaluate_trace(trace)
            judge_evidence = self._judge_evidence(trace)
            scores = await self._judge.judge(
                question,
                actual_answer,
                context=context or None,
                evidence=judge_evidence,
            )
            if scores.judge_failed:
                score_payload: Dict[str, float] = {}
                passed = None
                detail = (
                    f"Q: {question[:30]}... → Judge 未评分；"
                    f"Trace 结构完整性={'通过' if trace_evaluation['complete'] else '未通过'}"
                )
            else:
                score_payload = {
                    "relevance": scores.relevance,
                    "accuracy": scores.accuracy,
                    "completeness": scores.completeness,
                    "helpfulness": scores.helpfulness,
                    "overall": scores.overall,
                }
                passed = scores.overall >= self.PASS_THRESHOLD
                detail = f"Q: {question[:30]}... → Judge 综合评分 {scores.overall:.3f}"

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": actual_answer})

            test_id = f"dialog_{case_idx}" if len(questions) == 1 else f"dialog_{case_idx}_turn_{turn_idx}"
            results.append(EvalResult(
                test_id=test_id,
                passed=passed,
                scores=score_payload,
                detail=detail,
                metadata={
                    "question": question,
                    "response": actual_answer,
                    "agent_type": orch_result.agent_type,
                    "intent": orch_result.intent.value if orch_result.intent else None,
                    "turn": turn_idx,
                    "conv_id": conv_id,
                    "request_id": orch_result.request_id,
                    "evaluation_layers": {
                        "deterministic_trace": trace_evaluation,
                        "llm_judge": {
                            "configured": self._judge.configured,
                            "model": self._judge.model,
                            "failed": scores.judge_failed,
                            "error": scores.error,
                            "scores": score_payload,
                        },
                    },
                },
            ))

        return results

    @staticmethod
    def _evaluate_trace(trace: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """生成不依赖 LLM 的结构完整性结果，不冒充业务正确性评测。"""
        if trace is None:
            return {
                "available": False,
                "complete": False,
                "missing_fields": ["request_trace"],
                "workflow_status": None,
                "tool_call_count": 0,
                "evidence_count": 0,
                "side_effect_count": 0,
            }
        completeness = validate_trace_completeness(trace)
        execution = trace.get("execution") or {}
        final = trace.get("final") or {}
        return {
            "available": True,
            "complete": bool(completeness["complete"]),
            "missing_fields": list(completeness["missing_fields"]),
            "workflow_status": final.get("workflow_status"),
            "tool_call_count": len(execution.get("tool_calls") or []),
            "evidence_count": len(trace.get("evidence") or []),
            "side_effect_count": len(trace.get("side_effects") or []),
        }

    @staticmethod
    def _judge_evidence(trace: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """裁剪 canonical Trace，只把 Judge 判断回答所需的证据放入 Prompt。"""
        if trace is None:
            return {}
        execution = trace.get("execution") or {}
        return {
            "request_id": trace.get("request_id"),
            "task_understanding": trace.get("task_understanding") or {},
            "routing": trace.get("routing") or {},
            "tool_calls": execution.get("tool_calls") or [],
            "evidence": trace.get("evidence") or [],
            "side_effects": trace.get("side_effects") or [],
            "final_status": (trace.get("final") or {}).get("workflow_status"),
        }

    def _summarize_evaluation_layers(self, results: List[EvalResult]) -> Dict[str, Any]:
        trace_layers = []
        judge_layers = []
        for result in results:
            layers = result.metadata.get("evaluation_layers") or {}
            if layers.get("deterministic_trace"):
                trace_layers.append(layers["deterministic_trace"])
            if layers.get("llm_judge"):
                judge_layers.append(layers["llm_judge"])
        trace_complete = sum(1 for item in trace_layers if item.get("complete"))
        judge_scored = sum(1 for item in judge_layers if not item.get("failed"))
        return {
            "deterministic_trace": {
                "scope": "RequestTrace 结构完整性；不代表工具选择或业务结果正确",
                "total": len(trace_layers),
                "complete": trace_complete,
                "complete_rate": round(trace_complete / len(trace_layers), 4) if trace_layers else None,
            },
            "llm_judge": {
                "configured": self._judge.configured,
                "model": self._judge.model,
                "total": len(judge_layers),
                "scored": judge_scored,
                "failed_or_skipped": len(judge_layers) - judge_scored,
            },
        }

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        turns = case.get("turns")
        if isinstance(turns, list):
            return [str(t) for t in turns if str(t).strip()]
        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = [f"{m['role']}: {m['content']}" for m in history[-8:]]
        return "[评测多轮历史]\n" + "\n".join(lines)

    def _detect_regressions(self, current: Dict[str, float]) -> List[str]:
        """与上一次评测对比，找出退化超过 5% 的指标。"""
        prev_report = self._history[-1] if self._history else self._baseline
        if prev_report is None:
            return []
        prev = prev_report.avg_scores
        regressions = []
        for metric, value in current.items():
            if metric in prev and prev[metric] > 0:
                delta = (value - prev[metric]) / prev[metric]
                if delta < -0.05:
                    regressions.append(
                        f"{metric}: {prev[metric]:.3f} → {value:.3f} (退化 {abs(delta):.1%})"
                    )
        return regressions

    def _recommendations(
        self,
        scores: Dict[str, float],
        intent_metrics: Dict[str, Any],
    ) -> List[str]:
        recs = []
        if scores.get("intent_accuracy", 1.0) < 0.90:
            recs.append("意图识别准确率 < 90%：增加 Few-shot 示例，或对低 F1 的意图类别补充训练数据")
        if scores.get("relevance", 1.0) < 0.75:
            recs.append("相关性偏低：检查 Agent system_prompt，确保 Agent 聚焦于用户问题")
        if scores.get("completeness", 1.0) < 0.75:
            recs.append("完整性偏低：Agent 可能过早结束回答，考虑在 prompt 中要求提供完整解决方案")
        if scores.get("helpfulness", 1.0) < 0.75:
            recs.append("有用性偏低：回答可能过于抽象，考虑要求 Agent 提供具体操作步骤")
        if not recs and not scores:
            recs.append("没有可用的质量评分；检查 Judge 配置或运行状态")
        elif not recs:
            recs.append("所有指标均达标，继续保持")
        return recs

    @property
    def history(self) -> List[EvalReport]:
        return self._history

    def _load_baseline(self) -> Optional[EvalReport]:
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            data = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            return self._report_from_dict(data)
        except Exception as ex:
            logger.warning(f"读取评测基线失败: {ex}")
            return None

    def _save_baseline(self, report: EvalReport) -> None:
        if not self._baseline_path:
            return
        try:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self._baseline_path.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._baseline = report
        except Exception as ex:
            logger.warning(f"保存评测基线失败: {ex}")

    @staticmethod
    def _report_from_dict(data: Dict[str, Any]) -> EvalReport:
        return EvalReport(
            timestamp=data.get("timestamp", ""),
            total=int(data.get("total", 0)),
            passed=int(data.get("passed", 0)),
            pass_rate=(float(data["pass_rate"]) if data.get("pass_rate") is not None else None),
            avg_scores=dict(data.get("avg_scores", {})),
            regressions=list(data.get("regressions", [])),
            recommendations=list(data.get("recommendations", [])),
            results=[
                EvalResult(
                    test_id=r.get("test_id", ""),
                    passed=r.get("passed"),
                    scores=dict(r.get("scores", {})),
                    detail=r.get("detail", ""),
                    metadata=dict(r.get("metadata", {})),
                )
                for r in data.get("results", [])
            ],
            evaluation_layers=dict(data.get("evaluation_layers", {})),
            scored=int(data.get("scored", sum(r.get("passed") is not None for r in data.get("results", [])))),
            unscored=int(data.get("unscored", sum(r.get("passed") is None for r in data.get("results", [])))),
        )
