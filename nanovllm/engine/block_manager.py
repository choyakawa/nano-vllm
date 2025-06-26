from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Optional, Tuple, Any, List

from nanovllm.engine.sequence import Sequence, Turn

@dataclass
class CacheNode:
    node_id: int
    parent: Optional[CacheNode]
    block_table: List[int] = field(default_factory=list)
    token_count: int = 0
    ref_count: int = 0
    cache_group_ids: set[str] = field(default_factory=set)

@dataclass
class RadixNode:
    node_id: int = field(default_factory=lambda: next(RadixNode._node_counter))
    key_fragment: list[int] = field(default_factory=list)
    children: dict[int, 'RadixNode'] = field(default_factory=dict)
    data: Optional[CacheNode] = None
    sequential_children: set[int] = field(default_factory=set)
    _node_counter = count()

class RadixTree:
    def __init__(self):
        self.root = RadixNode(key_fragment=[])
        self.root.data = CacheNode(node_id=self.root.node_id, parent=None, ref_count=1)
        self.node_map: dict[int, RadixNode] = {self.root.node_id: self.root}

    def insert(self, token_ids: list[int], data_to_store: CacheNode) -> RadixNode:
        node = self.root
        pos = 0
        while pos < len(token_ids):
            if not token_ids: break
            token = token_ids[pos]
            if token not in node.children:
                new_node = RadixNode(key_fragment=token_ids[pos:])
                new_node.data = data_to_store
                node.children[token] = new_node
                self.node_map[new_node.node_id] = new_node
                return new_node
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
                self.node_map[common_node.node_id] = common_node
                child.key_fragment = fragment[common_len:]
                common_node.children[child.key_fragment[0]] = child
                node.children[token] = common_node
                remaining_tokens = token_ids[pos + common_len:]
                if remaining_tokens:
                    new_node = RadixNode(key_fragment=remaining_tokens)
                    new_node.data = data_to_store
                    common_node.children[new_node.key_fragment[0]] = new_node
                    self.node_map[new_node.node_id] = new_node
                    return new_node
                else:
                    common_node.data = data_to_store
                    return common_node
        node.data = data_to_store
        return node

    def find_longest_prefix_node(self, token_ids: list[int]) -> Tuple[Optional[RadixNode], int]:
        node = self.root
        pos = 0
        last_match_node: Optional[RadixNode] = self.root
        matched_len = 0
        while pos < len(token_ids):
            if not token_ids: break
            token = token_ids[pos]
            if token not in node.children: break
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
                if node.data:
                    last_match_node = node
                    matched_len = pos
            else: break
        if last_match_node == self.root and not self.root.data: return None, 0
        return last_match_node, matched_len

    def find_exact_match_node(self, token_ids: list[int]) -> Optional[RadixNode]:
        node, matched_len = self.find_longest_prefix_node(token_ids)
        if node and node.data and matched_len == len(token_ids): return node
        return None

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

    def _allocate_physical_block(self) -> int:
        if not self.free_block_ids: raise MemoryError("Out of free blocks")
        return self.free_block_ids.popleft()
    def _free_physical_block(self, block_id: int): self.free_block_ids.append(block_id)
    def _increase_ref_count(self, node: CacheNode):
        curr = node
        while curr is not None:
            if curr.ref_count == 0:
                for block_id in curr.block_table: self.blocks[block_id].ref_count += 1
            curr.ref_count += 1
            curr = curr.parent
    def _release_cache_node(self, node: Optional[CacheNode]):
        curr = node
        while curr is not None and curr.parent is not None:
            curr.ref_count -= 1
            if curr.ref_count == 0:
                for block_id in curr.block_table:
                    block = self.blocks[block_id]
                    block.ref_count -= 1
                    if block.ref_count == 0: self._free_physical_block(block_id)
            curr = curr.parent

    def _get_match_plan(self, seq: Sequence) -> tuple[list[dict], bool]:
        plan = [{"parent_radix_node": self.turn_cache.root, "matched_len": 0} for _ in seq.turns]

        if seq.cache_group_id:
            found_in_group = False
            for i, turn in enumerate(seq.turns):
                radix_node, matched_len = self.turn_cache.find_longest_prefix_node(turn.token_ids)
                if radix_node and radix_node.data:
                    curr = radix_node.data
                    while curr:
                        if seq.cache_group_id in curr.cache_group_ids:
                            plan[i] = {"parent_radix_node": radix_node, "matched_len": matched_len}
                            found_in_group = True
                            break
                        curr = curr.parent
            if found_in_group:
                return plan, False

        if not seq.turns:
            return plan, True

        current_seq_node = self.turn_cache.root
        sequence_matched_len = 0
        for i, turn in enumerate(seq.turns):
            turn_radix_node = self.turn_cache.find_exact_match_node(turn.token_ids)
            if turn_radix_node and turn_radix_node.node_id in current_seq_node.sequential_children:
                plan[i] = {"parent_radix_node": turn_radix_node, "matched_len": len(turn.token_ids)}
                current_seq_node = turn_radix_node
                sequence_matched_len += 1
            else:
                break

        if sequence_matched_len == 0:
            first_turn = seq.turns[0]
            radix_node, matched_len = self.turn_cache.find_longest_prefix_node(first_turn.token_ids)
            if matched_len > 0:
                plan[0] = {"parent_radix_node": radix_node, "matched_len": matched_len}

        return plan, True

    def _get_ancestors(self, cache_node: CacheNode) -> list[CacheNode]:
        ancestors = []
        curr = cache_node
        while curr:
            ancestors.append(curr)
            curr = curr.parent
        return ancestors

    def match_and_allocate(self, seq: Sequence) -> bool:
        match_plan, is_sequential_match = self._get_match_plan(seq)
        total_blocks_needed = 0
        allocation_details = []
        for i, turn in enumerate(seq.turns):
            parent_radix_node = match_plan[i]["parent_radix_node"]
            parent_cache_node = parent_radix_node.data
            matched_len = match_plan[i]["matched_len"]
            len_total = len(turn.token_ids)
            blocks_for_matched_prefix = (matched_len + self.block_size - 1) // self.block_size if matched_len > 0 else 0
            blocks_for_full_turn = (len_total + self.block_size - 1) // self.block_size if len_total > 0 else 0
            num_new_blocks = blocks_for_full_turn - blocks_for_matched_prefix
            total_blocks_needed += num_new_blocks
            cached_blocks_list = [b for n in reversed(self._get_ancestors(parent_cache_node)) for b in n.block_table]
            allocation_details.append({
                "parent_radix_node": parent_radix_node,
                "num_new_blocks": num_new_blocks,
                "tokens_to_cache": turn.token_ids[matched_len:],
                "cached_blocks_for_prefix": cached_blocks_list[:blocks_for_matched_prefix]
            })

        if len(self.free_block_ids) < total_blocks_needed: return False

        seq.num_cached_tokens = 0
        final_block_table = []
        final_radix_nodes = []
        for i, detail in enumerate(allocation_details):
            parent_radix_node = detail["parent_radix_node"]
            parent_cache_node = parent_radix_node.data
            tokens_to_cache = detail["tokens_to_cache"]
            turn_block_table = list(detail["cached_blocks_for_prefix"])
            seq.num_cached_tokens += len(seq.turns[i].token_ids) - len(tokens_to_cache)
            leaf_radix_node = parent_radix_node
            if tokens_to_cache:
                newly_allocated_blocks = [self._allocate_physical_block() for _ in range(detail["num_new_blocks"])]
                new_cache_node = CacheNode(
                    node_id=-1, parent=parent_cache_node,
                    block_table=newly_allocated_blocks, token_count=len(tokens_to_cache),
                    ref_count=0, cache_group_ids={seq.cache_group_id} if seq.cache_group_id else set()
                )
                leaf_radix_node = self.turn_cache.insert(seq.turns[i].token_ids, new_cache_node)
                leaf_radix_node.data.node_id = leaf_radix_node.node_id
                turn_block_table.extend(newly_allocated_blocks)

            self._increase_ref_count(leaf_radix_node.data)
            if seq.cache_group_id:
                for ancestor in self._get_ancestors(leaf_radix_node.data):
                    ancestor.cache_group_ids.add(seq.cache_group_id)

            seq.turn_cache_nodes[i] = leaf_radix_node.data
            final_radix_nodes.append(leaf_radix_node)
            final_block_table.extend(turn_block_table)

        if is_sequential_match:
            current_seq_node = self.turn_cache.root
            for node in final_radix_nodes:
                if node.node_id != self.turn_cache.root.node_id:
                    current_seq_node.sequential_children.add(node.node_id)
                    current_seq_node = node

        seq.block_table = final_block_table
        return True

    def deallocate(self, seq: Sequence):
        for node in seq.turn_cache_nodes: self._release_cache_node(node)
        seq.turn_cache_nodes = [None] * len(seq.turns)
        seq.block_table.clear()
    def can_append(self, seq: Sequence) -> bool: return len(self.free_block_ids) >= (len(seq) % self.block_size == 0)
    def may_append(self, seq: Sequence):
        if len(seq) > 0 and (len(seq) - 1) % self.block_size == 0:
            try: block_id = self._allocate_physical_block()
            except MemoryError: raise MemoryError("Out of blocks for decoding")
            self.blocks[block_id].ref_count = 1
            seq.block_table.append(block_id)
