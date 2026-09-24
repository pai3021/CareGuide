"""Definition-only KUAKE-QIC auxiliary taxonomy baseline.

The rules were written from the eleven public label definitions, without
reading the pinned official dev examples.  They provide an interpretable second
layer for traces; they do not replace CareGuide's operational tool intents.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from careguide.external_data import CAREGUIDE_INTENT_LABELS, QIC_LABELS


class QICRuleClassifier:
    version = "qic-definition-rules-v1"

    RULES = (
        ("医疗费用", r"费用|价格|多少钱|收费|报销"),
        ("指标解读", r"指标|数值|化验|检验|参考值|阳性|阴性|偏高|偏低"),
        ("病因分析", r"为什么|什么原因|病因|怎么引起|为何会|诱因"),
        ("治疗方案", r"如何治疗|怎么治|治疗方法|治疗方案|调养|怎么办才好"),
        ("功效作用", r"功效|作用|能治|有没有效果|有效吗"),
        ("注意事项", r"注意事项|要注意|禁忌|副作用|能不能吃|可以吃吗|可以用吗"),
        ("就医建议", r"挂什么科|看什么科|去哪个科|需要就医|要去医院|找什么医生"),
        ("后果表述", r"后果|危害|会怎样|有什么影响|严重吗|会不会恶化"),
        ("病情诊断", r"是不是得了|是不是患|什么病|诊断|确诊|属于.*病"),
        ("疾病表述", r"什么是|是什么意思|有哪些症状|症状|表现|定义"),
    )

    def predict(self, query: str) -> tuple[str, float]:
        for label, expression in self.RULES:
            if re.search(expression, query):
                return label, 0.8
        return "其他", 0.35


class QICPrototypeClassifier:
    """Train-only character TF-IDF prototypes for the reproducible local branch.

    This is the offline analogue of the remote embedding branch.  It never
    reads the official dev split and therefore can safely provide a baseline,
    tests, and fallback diagnostics without leaking evaluation examples.
    """

    version = "qic-train-char-tfidf-centroid-v1"

    def __init__(self, cases: tuple[dict, ...]):
        if not cases or {item["label"] for item in cases} != set(QIC_LABELS):
            raise ValueError("QIC train split is missing labels")
        self.document_frequency: Counter[str] = Counter()
        for item in cases:
            self.document_frequency.update(set(self._tokens(item["query"])))
        self.document_count = len(cases)
        sums: dict[str, Counter[str]] = defaultdict(Counter)
        for item in cases:
            sums[item["label"]].update(self._vector(item["query"]))
        self.centroids = {label: self._normalize(dict(sums[label])) for label in QIC_LABELS}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        value = re.sub(r"\s+", "", str(text).casefold())
        return [value[index:index + size] for size in (1, 2, 3)
                for index in range(max(0, len(value) - size + 1))]

    @staticmethod
    def _normalize(vector: dict[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {key: value / norm for key, value in vector.items()}

    def _vector(self, text: str) -> dict[str, float]:
        counts = Counter(self._tokens(text))
        values = {
            token: (1 + math.log(count)) * math.log(
                (self.document_count + 1) / (self.document_frequency[token] + 1))
            for token, count in counts.items() if self.document_frequency[token] >= 2
        }
        return self._normalize(values)

    def score(self, query: str) -> dict[str, float]:
        vector = self._vector(query)
        return {label: sum(value * self.centroids[label].get(token, 0.0)
                           for token, value in vector.items()) for label in QIC_LABELS}

    def predict(self, query: str) -> tuple[str, float]:
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        label, top = ranked[0]
        margin = top - ranked[1][1]
        confidence = max(0.35, min(0.95, 0.5 + top * 0.5 + margin * 2.0))
        return label, confidence


class CareGuide13PrototypeClassifier:
    """Train-only character TF-IDF prototype baseline for CareGuide-13."""

    version = "careguide-13-train-char-tfidf-centroid-v1"

    def __init__(self, cases):
        values = tuple(cases)
        if not values or {item.label for item in values} != set(CAREGUIDE_INTENT_LABELS):
            raise ValueError("CareGuide-13 train split is missing labels")
        self.document_frequency: Counter[str] = Counter()
        for item in values:
            self.document_frequency.update(set(QICPrototypeClassifier._tokens(item.message)))
        self.document_count = len(values)
        sums: dict[str, Counter[str]] = defaultdict(Counter)
        for item in values:
            sums[item.label].update(self._vector(item.message))
        self.centroids = {
            label: QICPrototypeClassifier._normalize(dict(sums[label]))
            for label in CAREGUIDE_INTENT_LABELS
        }

    def _vector(self, text: str) -> dict[str, float]:
        counts = Counter(QICPrototypeClassifier._tokens(text))
        values = {
            token: (1 + math.log(count)) * math.log(
                (self.document_count + 1) / (self.document_frequency[token] + 1))
            for token, count in counts.items() if self.document_frequency[token] >= 2
        }
        return QICPrototypeClassifier._normalize(values)

    def score(self, query: str) -> dict[str, float]:
        vector = self._vector(query)
        return {
            label: sum(value * self.centroids[label].get(token, 0.0)
                       for token, value in vector.items())
            for label in CAREGUIDE_INTENT_LABELS
        }

    def predict(self, query: str) -> tuple[str, float]:
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        label, top = ranked[0]
        margin = top - ranked[1][1]
        confidence = max(0.35, min(0.95, 0.5 + top * 0.5 + margin * 2.0))
        return label, confidence


class CareGuide13HybridClassifier:
    """High-precision workflow/safety rules over the train-only baseline."""

    version = "careguide-13-rules-plus-char-tfidf-v1"
    EMERGENCY = re.compile(
        r"(?:胸痛.*(?:呼吸困难|喘不上气)|(?:呼吸困难|喘不上气).*胸痛|"
        r"叫不醒|没有反应|失去意识|不省人事|意识不清|意识模糊|"
        r"正在|持续|全身).*抽搐|"
        r"(?:大量出血|出血.*止不住)|"
        r"(?:误服|误食|吞了|吃下|喝下).*(?:大量|不少|药|有害|清洗液)|"
        r"(?:有害|清洗液|大量药|不少药).*(?:误服|误食|吞|吃下|喝下)|"
        r"(?:喉咙|咽喉).*(?:肿|呼吸困难|喘不上气)|"
        r"突然.*(?:嘴歪|一侧手脚无力|说话含糊)"
    )
    APPOINTMENT = re.compile(
        r"(?:模拟预约|模拟号源|创建预约|提交预约|预约改期|取消.*预约|退掉.*预约|"
        r"预约.*(?:状态|结果)|APT-[A-Z0-9-]+|SLOT-[A-Z0-9-]+|"
        r"(?:查询|列出|查看|哪些).*(?:可预约号源|模拟医生|医生列表)|"
        r"(?:可预约号源|模拟号源|预约记录))",
        re.I,
    )

    def __init__(self, cases):
        self.prototype = CareGuide13PrototypeClassifier(cases)

    def predict(self, query: str) -> tuple[str, float]:
        if self.EMERGENCY.search(query):
            return "emergency", 0.99
        if self.APPOINTMENT.search(query):
            return "appointment_action", 0.97
        return self.prototype.predict(query)
