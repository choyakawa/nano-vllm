from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        waiting_list = list(self.waiting)
        self.waiting.clear()

        while waiting_list:
            seq = waiting_list.pop(0)
            if num_seqs >= self.max_num_seqs:
                self.waiting.appendleft(seq)
                break

            if self.block_manager.match_and_allocate(seq):
                uncached_tokens = len(seq) - seq.num_cached_tokens
                if num_batched_tokens + uncached_tokens > self.max_num_batched_tokens:
                    self.block_manager.deallocate(seq)
                    self.waiting.appendleft(seq)
                    continue

                num_seqs += 1
                num_batched_tokens += uncached_tokens
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_seqs.append(seq)
            else:
                self.waiting.appendleft(seq)
                break

        self.waiting.extend(waiting_list)

        if scheduled_seqs:
            return scheduled_seqs, True

        scheduled_seqs.clear()
        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            try:
                self.block_manager.may_append(seq)
                num_seqs += 1
                scheduled_seqs.append(seq)
            except MemoryError:
                self.preempt(seq)
                if self.running:
                    self.preempt(self.running.pop())
                break

        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
