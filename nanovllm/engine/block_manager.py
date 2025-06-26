from collections import deque
from dataclasses import dataclass, field
import xxhash
import numpy as np
from typing import Optional

from nanovllm.engine.sequence import Sequence, Turn


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0


@dataclass
class TurnData:
    token_ids: list[int]
    block_table: list[int]
    cache_group_ids: set[str] = field(default_factory=set)


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.turn_cache: dict[int, TurnData] = {}
        self.context_hashes: set[int] = set()

    @staticmethod
    def compute_context_hash(turn_hashes: list[int]) -> int:
        h = xxhash.xxh64()
        h.update(str(tuple(turn_hashes)).encode('utf-8'))
        return h.intdigest()

    def _allocate_physical_block(self) -> Optional[int]:
        if not self.free_block_ids:
            return None
        block_id = self.free_block_ids.popleft()
        assert self.blocks[block_id].ref_count == 0
        return block_id

    def _free_physical_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count > 0
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        required_blocks = 0
        for turn in seq.turns:
            required_blocks += turn.num_blocks
        return len(self.free_block_ids) >= required_blocks

    def match_and_allocate(self, seq: Sequence) -> bool:
        seq_turn_hashes = seq.turn_hashes
        matched_turns: dict[int, TurnData] = {}
        use_strict_prefix = True

        if seq.cache_group_id:
            found_match_in_group = False
            for turn_hash in seq_turn_hashes:
                if turn_hash in self.turn_cache and seq.cache_group_id in self.turn_cache[turn_hash].cache_group_ids:
                    matched_turns[turn_hash] = self.turn_cache[turn_hash]
                    found_match_in_group = True

            if found_match_in_group:
                use_strict_prefix = False

        if use_strict_prefix:
            matched_turns.clear()
            for i in range(len(seq_turn_hashes), 0, -1):
                prefix_hashes = seq_turn_hashes[:i]
                prefix_context_hash = self.compute_context_hash(prefix_hashes)
                if prefix_context_hash in self.context_hashes:
                    for turn_hash in prefix_hashes:
                        matched_turns[turn_hash] = self.turn_cache[turn_hash]
                    break

        blocks_to_allocate = 0
        allocation_plan = []
        for turn in seq.turns:
            if turn.hash in matched_turns:
                allocation_plan.append({'type': 'reuse', 'turn_hash': turn.hash})
            else:
                num_blocks_needed = turn.num_blocks
                blocks_to_allocate += num_blocks_needed
                allocation_plan.append({'type': 'alloc', 'turn': turn, 'num_blocks': num_blocks_needed})

        if len(self.free_block_ids) < blocks_to_allocate:
            return False

        final_block_table = []
        seq.num_cached_tokens = 0
        updated_group_ids_for_turns = set()

        for plan in allocation_plan:
            if plan['type'] == 'reuse':
                turn_data = matched_turns[plan['turn_hash']]
                final_block_table.extend(turn_data.block_table)
                seq.num_cached_tokens += len(turn_data.token_ids)

                for block_id in turn_data.block_table:
                    self.blocks[block_id].ref_count += 1

                if seq.cache_group_id and plan['turn_hash'] not in updated_group_ids_for_turns:
                    turn_data.cache_group_ids.add(seq.cache_group_id)
                    updated_group_ids_for_turns.add(plan['turn_hash'])

            elif plan['type'] == 'alloc':
                turn = plan['turn']
                num_blocks = plan['num_blocks']
                newly_allocated_blocks = []
                for _ in range(num_blocks):
                    block_id = self._allocate_physical_block()
                    self.blocks[block_id].ref_count = 1
                    newly_allocated_blocks.append(block_id)

                final_block_table.extend(newly_allocated_blocks)

                new_turn_data = TurnData(
                    token_ids=turn.token_ids,
                    block_table=newly_allocated_blocks,
                    cache_group_ids={seq.cache_group_id} if seq.cache_group_id else set(),
                )
                self.turn_cache[turn.hash] = new_turn_data

        for i in range(1, len(seq_turn_hashes) + 1):
             prefix_hashes = seq_turn_hashes[:i]
             context_hash = self.compute_context_hash(prefix_hashes)
             self.context_hashes.add(context_hash)

        seq.block_table = final_block_table
        return True

    def deallocate(self, seq: Sequence):
        for block_id in seq.block_table:
            self._free_physical_block(block_id)
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 0)

    def may_append(self, seq: Sequence):
        if len(seq) > 0 and (len(seq) - 1) % self.block_size == 0:
            block_id = self._allocate_physical_block()
            if block_id is None:
                raise MemoryError("Out of blocks for decoding")

            self.blocks[block_id].ref_count = 1
            seq.block_table.append(block_id)
