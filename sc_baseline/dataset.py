"""Observed-field mapping and independent replay plans for released dataset files."""
from dataclasses import dataclass
import json
from .core import validate_request
from .types import ExtractionRequest


@dataclass(frozen=True)
class DatasetTurn:
    request: ExtractionRequest
    source_id: str
    source_conversation_id: str
    turn_index: int
    variant_id: str | None = None
    injection_turn: bool = False

    def source_dict(self):
        return {'source_id': self.source_id,
                'conversation_id': self.source_conversation_id,
                'turn_index': self.turn_index, 'variant_id': self.variant_id,
                'injection_turn': self.injection_turn}


def _identity(row, id_field='id'):
    if not isinstance(row, dict):
        raise ValueError('Dataset rows must be objects.')
    for field in (id_field, 'conversation_id'):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f'Missing or invalid {field}.')
    if type(row.get('turn_index')) is not int or row['turn_index'] < 0:
        raise ValueError('turn_index must be a nonnegative integer.')
    req = ExtractionRequest(row['conversation_id'], row[id_field], row.get('user_message'),
                            row.get('previous_assistant_message'))
    validate_request(req)
    return req


def _namespace(kind, value):
    return json.dumps([kind, value], ensure_ascii=False, separators=(',', ':'))


def _ordinary(rows, mode):
    seen, previous, turns = set(), {}, []
    for row in rows:
        req = _identity(row)
        if req.message_id in seen:
            raise ValueError('Duplicate dataset id.')
        seen.add(req.message_id)
        if mode == 'stream':
            index = row['turn_index']
            if index <= previous.get(req.conversation_id, -1):
                raise ValueError('Stream turn indices must increase within each conversation.')
            previous[req.conversation_id] = index
        else:
            req = ExtractionRequest(_namespace('example', req.message_id), req.message_id,
                                    req.user_message, req.previous_assistant_message)
        turns.append(DatasetTurn(req, row['id'], row['conversation_id'], row['turn_index']))
    return turns


def prepare_turns(rows, mode, *, clean_rows=None):
    """Validate all rows before inference; injection mode replays full variants.

    No gold fields are read. The supplied injected user text replaces exactly one
    clean user turn; subsequent observed assistant context stays as released.
    """
    if mode not in ('examples', 'stream', 'injections'):
        raise ValueError('Unknown dataset mode.')
    if mode != 'injections':
        if clean_rows is not None:
            raise ValueError('clean_rows is only valid for injections.')
        return _ordinary(rows, mode)
    if clean_rows is None:
        raise ValueError('Injection replay requires clean stream rows.')
    clean = _ordinary(clean_rows, 'stream')
    conversations, locations = {}, {}
    for turn in clean:
        conversations.setdefault(turn.source_conversation_id, []).append(turn)
        locations[(turn.source_conversation_id, turn.turn_index)] = turn
    seen, result = set(), []
    for row in rows:
        injected = _identity(row, 'conpact_id')
        variant = row['conpact_id']
        if variant in seen:
            raise ValueError('Duplicate conpact_id.')
        seen.add(variant)
        target = locations.get((injected.conversation_id, row['turn_index']))
        if target is None or injected.previous_assistant_message != target.request.previous_assistant_message:
            raise ValueError('Injection location/context does not match clean stream.')
        for turn in conversations[injected.conversation_id]:
            changed = turn.turn_index == row['turn_index']
            req = ExtractionRequest(_namespace('variant', variant), turn.request.message_id,
                injected.user_message if changed else turn.request.user_message,
                turn.request.previous_assistant_message)
            result.append(DatasetTurn(req, turn.source_id, turn.source_conversation_id,
                                      turn.turn_index, variant, changed))
    return result


def run_turns(turns, runtime):
    """Run each planned turn exactly once, with isolated per-execution state."""
    from .core import extract_turn
    from .types import ExtractionStatus
    registries = {}
    for turn in turns:
        key = turn.request.conversation_id
        result = extract_turn(turn.request, registries.get(key, []), runtime)
        registries[key] = result.registry
        yield {'source': turn.source_dict(), 'extraction': result.to_dict()}
        if result.status != ExtractionStatus.SUCCESS:
            return
