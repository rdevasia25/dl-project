"""Stateless shared extraction entrypoint; no gate, dataset, retries or cache."""
from time import perf_counter
from .prompts import build_sc_extraction_prompts
from .parsing import parse_with_errors
from .types import (ExtractionRequest, RegistryEntry, ModelRuntime, ModelResponse,
                    ExtractionResult, ExtractionError, ExtractionStatus)


def validate_request(request: ExtractionRequest) -> None:
    if not isinstance(request, ExtractionRequest):
        raise ValueError('Expected ExtractionRequest.')
    for value in (request.conversation_id, request.message_id):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Identifiers must be nonempty strings.')
    if not isinstance(request.user_message, str):
        raise ValueError('user_message must be a string.')
    if request.previous_assistant_message is not None and not isinstance(request.previous_assistant_message, str):
        raise ValueError('previous_assistant_message must be a string or null.')


def render_registry(registry: list[RegistryEntry]) -> str | None:
    if not registry:
        return None
    lines = '\n'.join(f'{i}. {entry.text}' for i, entry in enumerate(registry, 1))
    return f'Constraints from earlier in this session:\n{lines}'


def extract_turn(request: ExtractionRequest, registry: list[RegistryEntry],
                 runtime: ModelRuntime) -> ExtractionResult:
    """Validate, attempt exactly one generation, and return independent new state.

    Invalid caller input raises ValueError before generation. Operational failures
    return structured results. Accepted partial output is appended as upstream does.
    """
    validate_request(request)
    if not isinstance(registry, list) or any(
        not isinstance(e, RegistryEntry) or not isinstance(e.text, str) or not e.text.strip()
        or not isinstance(e.evidence, str) or not isinstance(e.source_message_id, str)
        or not e.source_message_id.strip() for e in registry
    ):
        raise ValueError('Invalid registry entries.')
    state = list(registry)
    system, user = build_sc_extraction_prompts(request.user_message,
        [e.text for e in state], request.previous_assistant_message)
    started = perf_counter()
    response = None
    errors = []
    parsed, added = [], []
    invoked = True
    status = ExtractionStatus.SUCCESS
    try:
        response = runtime.generate([{'role': 'system', 'content': system},
                                     {'role': 'user', 'content': user}])
    except Exception as exc:
        # Never serialize exception text: transports may embed credentials/bodies.
        from .runtime import RuntimeFailure
        known = isinstance(exc, RuntimeFailure)
        invoked = exc.attempted if known else True
        status = ExtractionStatus.PROVIDER_ERROR if invoked else ExtractionStatus.NOT_INVOKED
        errors.append(ExtractionError(exc.code if known else 'runtime_error',
                                      exc.safe_message if known else 'Model runtime failed.'))
    elapsed = (perf_counter() - started) * 1000 if invoked else None
    if response is not None:
        if not isinstance(response, ModelResponse) or not isinstance(response.text, str):
            errors.append(ExtractionError('invalid_response', 'Runtime returned an invalid response.'))
            response = None
        else:
            parsed, diagnostics = parse_with_errors(response.text)
            errors.extend(diagnostics)
            if response.model != runtime.model:
                errors.append(ExtractionError('model_mismatch', 'Provider returned a different model.'))
                parsed = []
            if response.finish_reason not in (None, 'stop'):
                errors.append(ExtractionError('incomplete_response', 'Generation did not finish normally.'))
            seen = {e.text for e in state}
            for item in parsed:
                if item.text not in seen:
                    entry = RegistryEntry(item.text, item.evidence, request.message_id)
                    state.append(entry)
                    added.append(entry)
                    seen.add(item.text)
        if errors:
            status = ExtractionStatus.RESPONSE_ERROR
    elif status == ExtractionStatus.SUCCESS:
        status = ExtractionStatus.RESPONSE_ERROR
        errors.append(ExtractionError('invalid_response', 'Runtime returned no response.'))
    return ExtractionResult(request.conversation_id, request.message_id, parsed, added,
        state, render_registry(state), status, invoked, int(invoked), runtime.provider,
        response.model if response else runtime.model, runtime.config_id,
        response.usage if response else None, elapsed,
        response.cost_usd if response else None, response.cost_source if response else None,
        response.text if response else None, errors)
