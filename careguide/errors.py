"""Stable failures shared by workers, planning and deterministic replies."""
from dataclasses import asdict, dataclass
from enum import Enum


class ModelFormatError(ValueError):
    """A provider response cannot be interpreted as tool arguments."""


class ErrorCode(str, Enum):
    MODEL_FORMAT = 'model_format_error'
    MISSING_INPUT = 'missing_user_input'
    INVALID_PARAMETERS = 'invalid_parameters'
    PERMISSION_DENIED = 'permission_denied'
    NOT_COVERED = 'not_covered'
    SOURCE_UNAVAILABLE = 'source_unavailable'
    TOOL_UNAVAILABLE = 'tool_unavailable'
    TOOL_TIMEOUT = 'tool_timeout'
    MODEL_UNAVAILABLE = 'model_unavailable'
    BUDGET_EXHAUSTED = 'budget_exhausted'
    INTERNAL_ERROR = 'internal_error'
    DEPENDENCY_BLOCKED = 'dependency_blocked'


_GUIDANCE = {
    ErrorCode.MODEL_FORMAT: ('模型返回的工具参数格式不正确。', '可有限修正；失败后重新发起查询。'),
    ErrorCode.MISSING_INPUT: ('查询缺少必要信息。', '请补充列出的字段。'),
    ErrorCode.INVALID_PARAMETERS: ('工具参数未通过校验。', '请核对查询条件，不会猜测关键参数。'),
    ErrorCode.PERMISSION_DENIED: ('当前任务不允许执行此操作。', '请在允许的业务范围内重新提出请求。'),
    ErrorCode.NOT_COVERED: ('当前资料未覆盖该查询。', '请核对官方来源；未命中不代表安全或没有风险。'),
    ErrorCode.SOURCE_UNAVAILABLE: ('资料已过复核期限或当前来源不可用。', '请核对有效的官方来源。'),
    ErrorCode.TOOL_UNAVAILABLE: ('工具服务暂不可用。', '已完成的结果保留，可稍后重试失败项。'),
    ErrorCode.TOOL_TIMEOUT: ('工具未在期限内返回。', '保留已完成结果；超时不代表底层操作已停止。'),
    ErrorCode.MODEL_UNAVAILABLE: ('模型服务暂不可用。', '保留已完成结果，可稍后重试。'),
    ErrorCode.BUDGET_EXHAUSTED: ('累计执行预算已用尽。', '保留已完成结果，继续前需要检查剩余额度。'),
    ErrorCode.INTERNAL_ERROR: ('当前任务暂时无法完成。', '已完成结果保留，请稍后重试。'),
    ErrorCode.DEPENDENCY_BLOCKED: ('前序任务未完成，当前任务尚未执行。', '请先处理前序任务；不会使用失败或失效的数据继续。'),
}


@dataclass(frozen=True)
class Failure:
    code: ErrorCode
    stage: str
    task_id: str = ''
    call_id: str = ''
    retryable: bool = False
    missing_fields: tuple[str, ...] = ()

    def payload(self):
        value = asdict(self)
        value['code'] = self.code.value
        value['missing_fields'] = list(self.missing_fields)
        value['message'], value['next_step'] = _GUIDANCE[self.code]
        return value


def output_failure(output, *, task_id='', call_id=''):
    status = output.get('status')
    code = {'needs_clarification': ErrorCode.MISSING_INPUT,
            'not_covered': ErrorCode.NOT_COVERED,
            'source_unavailable': ErrorCode.SOURCE_UNAVAILABLE}.get(status)
    if code is None:
        return None
    return Failure(code, 'tool_result', task_id, call_id,
                   missing_fields=tuple(output.get('missing_fields', []))).payload()


def aggregate_status(calls, *, evidence=False, execution_errors=False):
    """Count effective outcomes; repaired attempts remain in the trace only."""
    effective = [call for call in calls if not call.get('resolved')]
    statuses = [call['output'].get('status') for call in effective]
    problems = any(call.get('failure') or not call.get('success', False) for call in effective)
    completed = bool(evidence or 'found' in statuses or 'completed' in statuses)
    if 'pending_confirmation' in statuses:
        return 'pending_confirmation'
    if any(status in {'executing', 'reconciling'} for status in statuses):
        return 'reconciling'
    if completed:
        return 'partial_completed' if problems or execution_errors else 'completed'
    if 'needs_clarification' in statuses:
        return 'needs_clarification'
    if any(s in statuses for s in ('model_format_error', 'invalid_parameters',
           'permission_denied', 'source_unavailable', 'failed', 'budget_exhausted')):
        return 'failed'
    if execution_errors:
        return 'failed'
    return 'not_covered'
