"""Locate change-of-turn and think-delimiter token positions in a finished
conversation's token sequence.

We scan the ACTUAL token ids (single-token markers, verified by the registry)
and decode-check every position we claim — a mislabeled position raises,
never silently produces wrong activations. Chain-of-thought interiors are
never returned: only the <think>/</think> delimiter tokens themselves.

Turn attribution: `turn_start`, `role`, `post_role_nl`, `think_open`,
`think_close` belong to the turn they OPEN; `turn_end` and `post_turn_end_nl`
belong to the turn they CLOSE. Templates may inject a system turn (SmolLM2,
Llama-3) that is not part of the ConversationRecord — boundary tokens of
system turns are labeled role="system", turn_index=-1 and kept (they can be
filtered downstream).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..chat_markup.markup import ChatMarkup
from ..conversation.records import ConversationRecord
from ..core.schema import BaseSchema

KNOWN_ROLES = ("system", "user", "assistant", "model", "tool")


@dataclass
class BoundaryToken(BaseSchema):
    abs_pos: int
    kind: str  # one of markup.BOUNDARY_KINDS
    role: str  # role of the turn this boundary belongs to
    turn_index: int  # index into ConversationRecord.turns; -1 for system turn
    step_index: int | None = None  # plan step of that turn (assistant turns)
    token_id: int = -1
    token_str: str = ""


@dataclass
class BoundaryMap(BaseSchema):
    sample_uid: str = ""
    boundaries: list[BoundaryToken] = field(default_factory=list)

    def positions(self) -> list[int]:
        return [b.abs_pos for b in self.boundaries]

    def filter(self, **conds) -> list[BoundaryToken]:
        out = []
        for b in self.boundaries:
            if all(getattr(b, k) == v for k, v in conds.items()):
                out.append(b)
        return out


def _decode_one(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def find_boundaries(
    token_ids: list[int],
    tokenizer,
    markup: ChatMarkup,
    record: ConversationRecord,
) -> BoundaryMap:
    turn_end_id = tokenizer.encode(markup.turn_end, add_special_tokens=False)[0]
    turn_start_id = tokenizer.encode(markup.turn_start, add_special_tokens=False)[0]
    think_ids: dict[int, str] = {}
    if markup.has_thinking:
        think_ids[tokenizer.encode(markup.think_open, add_special_tokens=False)[0]] = (
            "think_open"
        )
        think_ids[tokenizer.encode(markup.think_close, add_special_tokens=False)[0]] = (
            "think_close"
        )

    n = len(token_ids)
    boundaries: list[BoundaryToken] = []

    # ---- pass 1: segment into turns by turn_start markers -------------------
    # For each turn_start, greedily read the role word + post_role_sep.
    turn_opens: list[dict] = []  # {pos, role, role_positions, sep_positions}
    i = 0
    while i < n:
        if token_ids[i] != turn_start_id:
            i += 1
            continue
        # Decode forward until we've matched "<role>{post_role_sep}".
        acc = ""
        j = i + 1
        role_positions: list[int] = []
        sep_positions: list[int] = []
        matched_role: str | None = None
        while j < n and j - i <= 12:  # role headers are short
            piece = _decode_one(tokenizer, token_ids[j])
            acc += piece
            if matched_role is None:
                for cand in KNOWN_ROLES:
                    if acc.rstrip() == cand or acc == cand:
                        matched_role = cand
                        role_positions = list(range(i + 1, j + 1))
                        break
                if matched_role is None and not any(
                    cand.startswith(acc) for cand in KNOWN_ROLES
                ):
                    break  # not a role header we recognize
            else:
                sep_positions.append(j)
                expected = matched_role + markup.post_role_sep
                if acc == expected:
                    break
                if not expected.startswith(acc):
                    break
            j += 1
        if matched_role is None:
            raise ValueError(
                f"turn_start at pos {i} not followed by a recognizable role "
                f"header (saw {acc!r})"
            )
        turn_opens.append(
            {
                "pos": i,
                "role": matched_role,
                "role_positions": role_positions,
                "sep_positions": sep_positions,
            }
        )
        i = j + 1

    # ---- pass 2: map token-space turns onto record turns --------------------
    record_roles = [t.role for t in record.turns]
    mapped: list[tuple[dict, int]] = []  # (open info, record turn index or -1)
    ri = 0
    for open_info in turn_opens:
        role = open_info["role"]
        if role == "system":
            mapped.append((open_info, -1))
            continue
        record_role = "assistant" if role == markup.assistant_role else role
        if ri >= len(record_roles) or record_roles[ri] != record_role:
            raise ValueError(
                f"Turn structure mismatch: token-space turn {len(mapped)} has "
                f"role {record_role!r} but record expects "
                f"{record_roles[ri] if ri < len(record_roles) else '<none>'!r} "
                f"at record index {ri}"
            )
        mapped.append((open_info, ri))
        ri += 1
    if ri != len(record_roles):
        raise ValueError(
            f"Only matched {ri} of {len(record_roles)} record turns in token "
            "sequence"
        )

    # ---- pass 3: emit boundary tokens --------------------------------------
    def turn_of_pos(pos: int) -> tuple[int, str]:
        """Record turn index + role owning position `pos` (turn it opened in)."""
        owner = (-1, "system")
        for open_info, rec_idx in mapped:
            if open_info["pos"] <= pos:
                role = open_info["role"]
                role = "assistant" if role == markup.assistant_role else role
                owner = (rec_idx, role)
            else:
                break
        return owner

    def step_of(rec_idx: int) -> int | None:
        if 0 <= rec_idx < len(record.turns):
            return record.turns[rec_idx].step_index
        return None

    for open_info, rec_idx in mapped:
        role = open_info["role"]
        rec_role = "assistant" if role == markup.assistant_role else role
        step = step_of(rec_idx)
        boundaries.append(
            BoundaryToken(
                abs_pos=open_info["pos"],
                kind="turn_start",
                role=rec_role,
                turn_index=rec_idx,
                step_index=step,
                token_id=token_ids[open_info["pos"]],
                token_str=markup.turn_start,
            )
        )
        for p in open_info["role_positions"]:
            boundaries.append(
                BoundaryToken(
                    abs_pos=p,
                    kind="role",
                    role=rec_role,
                    turn_index=rec_idx,
                    step_index=step,
                    token_id=token_ids[p],
                    token_str=_decode_one(tokenizer, token_ids[p]),
                )
            )
        for p in open_info["sep_positions"]:
            boundaries.append(
                BoundaryToken(
                    abs_pos=p,
                    kind="post_role_nl",
                    role=rec_role,
                    turn_index=rec_idx,
                    step_index=step,
                    token_id=token_ids[p],
                    token_str=_decode_one(tokenizer, token_ids[p]),
                )
            )

    for pos, tid in enumerate(token_ids):
        if tid == turn_end_id:
            rec_idx, role = turn_of_pos(pos)
            boundaries.append(
                BoundaryToken(
                    abs_pos=pos,
                    kind="turn_end",
                    role=role,
                    turn_index=rec_idx,
                    step_index=step_of(rec_idx),
                    token_id=tid,
                    token_str=markup.turn_end,
                )
            )
            if markup.post_turn_end_sep and pos + 1 < n:
                nxt = _decode_one(tokenizer, token_ids[pos + 1])
                if markup.post_turn_end_sep.startswith(nxt) or nxt.startswith(
                    markup.post_turn_end_sep
                ):
                    boundaries.append(
                        BoundaryToken(
                            abs_pos=pos + 1,
                            kind="post_turn_end_nl",
                            role=role,
                            turn_index=rec_idx,
                            step_index=step_of(rec_idx),
                            token_id=token_ids[pos + 1],
                            token_str=nxt,
                        )
                    )
        elif tid in think_ids:
            rec_idx, role = turn_of_pos(pos)
            boundaries.append(
                BoundaryToken(
                    abs_pos=pos,
                    kind=think_ids[tid],
                    role=role,
                    turn_index=rec_idx,
                    step_index=step_of(rec_idx),
                    token_id=tid,
                    token_str=_decode_one(tokenizer, token_ids[pos]),
                )
            )

    boundaries.sort(key=lambda b: b.abs_pos)

    # ---- verification -------------------------------------------------------
    n_assistant_token_space = sum(
        1 for oi, ri in mapped if oi["role"] == markup.assistant_role
    )
    n_assistant_record = sum(1 for t in record.turns if t.role == "assistant")
    if n_assistant_token_space != n_assistant_record:
        raise ValueError(
            f"Assistant turn count mismatch: {n_assistant_token_space} in "
            f"tokens vs {n_assistant_record} in record"
        )
    seen = set()
    for b in boundaries:
        if b.abs_pos in seen:
            raise ValueError(f"Duplicate boundary position {b.abs_pos}")
        seen.add(b.abs_pos)
        if b.abs_pos >= n:
            raise ValueError(f"Boundary position {b.abs_pos} out of range")

    return BoundaryMap(sample_uid=record.sample_uid, boundaries=boundaries)
