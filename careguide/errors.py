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
    MODEL_TIMEOUT = 'model_timeout'
    MODEL_AUTH_FAILED = 'model_auth_failed'
    MODEL_QUOTA_EXCEEDED = 'model_quota_exceeded'
    MODEL_RATE_LIMITED = 'model_rate_limited'
    MODEL_CONNECTION_FAILED = 'model_connection_failed'
    MODEL_ACCESS_DENIED = 'model_access_denied'
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
    ErrorCode.MODEL_TIMEOUT: ('模型响应超时。', '已保留完成的结果，请稍后重试。'),
    ErrorCode.MODEL_AUTH_FAILED: ('模型服务鉴权失败。', '请管理员检查模型 API 密钥及接口地址。'),
    ErrorCode.MODEL_QUOTA_EXCEEDED: ('模型服务额度不足。', '请管理员检查账户余额或套餐额度，恢复后重试。'),
    ErrorCode.MODEL_RATE_LIMITED: ('模型请求过于频繁，已被限流。', '请稍后重试；持续发生时请管理员检查并发和速率限制。'),
    ErrorCode.MODEL_CONNECTION_FAILED: ('无法连接模型服务。', '请管理员检查网络、代理及模型接口地址后重试。'),
    ErrorCode.MODEL_ACCESS_DENIED: ('模型服务拒绝访问。', '请管理员检查账户权限、模型授权及接口地址。'),
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


def model_failure(exc, stage, *, task_id='', call_id=''):
    """Classify provider failures without exposing response bodies or credentials.

    A generic 429 is a rate limit, not evidence of an exhausted balance.
    Only explicit provider quota codes/messages establish quota exhaustion.
    """
    from openai import APIConnectionError, APITimeoutError, APIStatusError

    code = ErrorCode.MODEL_UNAVAILABLE
    if isinstance(exc, (TimeoutError, APITimeoutError)):
        code = ErrorCode.MODEL_TIMEOUT
    elif isinstance(exc, APIStatusError):
        status = exc.status_code
        body = exc.body if isinstance(exc.body, dict) else {}
        detail = body.get('error', body)
        detail = detail if isinstance(detail, dict) else {}
        markers = ' '.join(str(detail.get(k, '')) for k in ('code', 'type', 'message')).casefold()
        if status == 401:
            code = ErrorCode.MODEL_AUTH_FAILED
        elif status in {402, 403, 429} and (status == 402 or any(marker in markers for marker in (
                'insufficient_quota', 'insufficient_balance', 'quota_exceeded',
                'quota exhausted', 'quota exceeded', 'insufficient balance',
                '余额不足', '额度不足', '额度已用尽', '套餐额度已耗尽'))):
            code = ErrorCode.MODEL_QUOTA_EXCEEDED
        elif status == 429:
            code = ErrorCode.MODEL_RATE_LIMITED
        elif status in {408, 504}:
            code = ErrorCode.MODEL_TIMEOUT
        elif status == 403:
            code = ErrorCode.MODEL_ACCESS_DENIED
    elif isinstance(exc, APIConnectionError):
        code = ErrorCode.MODEL_CONNECTION_FAILED
    return Failure(code, stage, task_id=task_id, call_id=call_id,
                   retryable=code in {ErrorCode.MODEL_TIMEOUT, ErrorCode.MODEL_RATE_LIMITED,
                                     ErrorCode.MODEL_CONNECTION_FAILED, ErrorCode.MODEL_UNAVAILABLE}).payload()


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
