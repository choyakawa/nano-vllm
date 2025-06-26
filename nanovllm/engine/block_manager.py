from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Optional, Tuple, Any

from nanovllm.engine.sequence import Sequence, Turn


@dataclass
class TurnData:
    token_ids: list[int]
    block_table: list[int]
    node_id: int
    cache_group_ids: set[str] = field(default_factory=set)

class RadixNode:
    _node_counter = count()

    def __init__(self, key_fragment: Optional[list[int]] = None):
        self.key_fragment: list[int] = key_fragment if key_fragment is not None else []
        self.children: dict[int, 'RadixNode'] = {}
        self.data: Optional[TurnData] = None
        self.node_id = next(self._node_counter)

class RadixTree:
    def __init__(self):
        self.root = RadixNode()

    def find_longest_prefix(self, token_ids: list[int]) -> Tuple[Optional[TurnData], int]:
        node = self.root
        pos = 0
        last_match_data: Optional[TurnData] = node.data
        matched_len = 0

        while pos < len(token_ids):
            if not token_ids: break
            token = token_ids[pos]
            if token not in node.children:
                break

            child = node.children[token]
            fragment = child.key_fragment

            common_len = 0
            while (common_len < len(fragment) and
                   pos + common_len < len(token_ids) and
                   fragment[common_len] == token_ids[pos + common_len]):
                common_len += 1

            if common_len < len(fragment):
                break

            pos += common_len
            node = child
            if node.data:
                last_match_data = node.data
                matched_len = pos

        return last_match_data, matched_len

    def find_exact_match_node(self, token_ids: list[int]) -> Optional[RadixNode]:
        data, length = self.find_longest_prefix(token_ids)
        if length == len(token_ids) and data is not None and data.token_ids == token_ids:
            node = self.root
            pos = 0
            while pos < len(token_ids):
                token = token_ids[pos]
                if token not in node.children: return None
                child = node.children[token]
                pos += len(child.key_fragment)
                node = child
            return node
        return None

    def insert(self, token_ids: list[int], data_to_store: TurnData):
        node = self.root
        pos = 0
        while pos < len(token_ids):
            token = token_ids[pos]
            if token not in node.children:
                new_node = RadixNode(key_fragment=token_ids[pos:])
                new_node.data = data_to_store
                node.children[token] = new_node
                return

            child = node.children[token]
            fragment = child.key_fragment

            common_len = 0
            while (common_len < len(fragment) and
                   pos + common_len < len(token_ids) and
                   fragment[common_len] == token_ids[pos + common_len]):
                common_len += 1

            if common_len == len(fragment):
                pos += common_len
                node = child
                continue
            else:
                common_node = RadixNode(key_fragment=fragment[:common_len])
                child.key_fragment = fragment[common_len:]
                common_node.children[child.key_fragment[0]] = child
                node.children[token] = common_node
                remaining_tokens = token_ids[pos + common_len:]
                if remaining_tokens:
                    new_node = RadixNode(key_fragment=remaining_tokens)
                    new_node.data = data_to_store
                    common_node.children[new_node.key_fragment[0]] = new_node
                else:
                    common_node.data = data_to_store
                return

        node.data = data_to_store

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.turn_cache: RadixTree = RadixTree()
        self.context_cache: set[tuple[int]] = set()

    def _allocate_physical_block(self) -> Optional[int]:
        if not self.free_block_ids:
            return None
        block_id = self.free_block_ids.popleft()
        assert self.blocks[block_id].ref_count == 0
        return block_id

    def _free_physical_block(self, block_id: int):
        block = self.blocks[block_id]
        if block.ref_count > 0:
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        required_blocks = sum(turn.num_blocks for turn in seq.turns)
        return len(self.free_block_ids) >= required_blocks

    def match_and_allocate(self, seq: Sequence) -> bool:
        turn_matches: list[Tuple[Optional[TurnData], int]] = [(None, 0)] * len(seq.turns)

        if seq.cache_group_id:
            found_in_group = False
            for i, turn in enumerate(seq.turns):
                cached_data, matched_len = self.turn_cache.find_longest_prefix(turn.token_ids)
                if cached_data and seq.cache_group_id in cached_data.cache_group_ids:
                    turn_matches[i] = (cached_data, matched_len)
                    found_in_group = True
            if not found_in_group:
                self._find_strict_prefix_match(seq, turn_matches)
        else:
            self._find_strict_prefix_match(seq, turn_matches)

        blocks_to_allocate = 0
        for i, turn in enumerate(seq.turns):
            _, matched_len = turn_matches[i]
            num_reused_blocks = (matched_len - 1) // self.block_size + 1 if matched_len > 0 else 0
            total_blocks_needed = (len(turn.token_ids) - 1) // self.block_size + 1 if turn.token_ids else 0
            blocks_to_allocate += (total_blocks_needed - num_reused_blocks)

        if len(self.free_block_ids) < blocks_to_allocate:
            return False

        seq.num_cached_tokens = 0
        final_block_table = []
        turn_node_ids_for_context = []

        for i, turn in enumerate(seq.turns):
            turn_block_table = []
            cached_data, matched_len = turn_matches[i]

            if cached_data:
                num_blocks_to_reuse = (matched_len - 1) // self.block_size + 1 if matched_len > 0 else 0
                if num_blocks_to_reuse > 0:
                    reused_blocks = cached_data.block_table[:num_blocks_to_reuse]
                    turn_block_table.extend(reused_blocks)
                    for block_id in reused_blocks:
                        self.blocks[block_id].ref_count += 1
                seq.num_cached_tokens += matched_len

            total_blocks_needed = (len(turn.token_ids) - 1) // self.block_size + 1 if turn.token_ids else 0
            num_new_blocks_to_alloc = total_blocks_needed - len(turn_block_table)
            if num_new_blocks_to_alloc > 0:
                for _ in range(num_new_blocks_to_alloc):
                    block_id = self._allocate_physical_block()
                    self.blocks[block_id].ref_count = 1
                    turn_block_table.append(block_id)

            existing_node = self.turn_cache.find_exact_match_node(turn.token_ids)
            if existing_node:
                if seq.cache_group_id:
                    existing_node.data.cache_group_ids.add(seq.cache_group_id)
                turn_node_ids_for_context.append(existing_node.node_id)
            else:
                new_turn_data = TurnData(
                    token_ids=turn.token_ids,
                    block_table=turn_block_table,
                    node_id=-1,
                    cache_group_ids={seq.cache_group_id} if seq.cache_group_id else set(),
                )
                self.turn_cache.insert(turn.token_ids, new_turn_data)

                node = self.turn_cache.find_exact_match_node(turn.token_ids)
                node.data.node_id = node.node_id
                turn_node_ids_for_context.append(node.node_id)

            final_block_table.extend(turn_block_table)

        if turn_node_ids_for_context:
            for i in range(1, len(turn_node_ids_for_context) + 1):
                self.context_cache.add(tuple(turn_node_ids_for_context[:i]))

        seq.block_table = final_block_table
        return True

    def _find_strict_prefix_match(self, seq: Sequence, turn_matches: list):
        for i in range(len(seq.turns), 0, -1):
            prefix_turns = seq.turns[:i]
            prefix_node_ids = []
            is_perfect_match = True

            for turn in prefix_turns:
                node = self.turn_cache.find_exact_match_node(turn.token_ids)
                if node:
                    prefix_node_ids.append(node.node_id)
                else:
                    is_perfect_match = False
                    break

            if is_perfect_match and tuple(prefix_node_ids) in self.context_cache:
                for j in range(i):
                    matched_node = self.turn_cache.find_exact_match_node(prefix_turns[j].token_ids)
                    turn_matches[j] = (matched_node.data, len(prefix_turns[j].token_ids))
                return

    def deallocate(self, seq: Sequence):
        unique_blocks = set(seq.block_table)
        for block_id in unique_blocks:
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
