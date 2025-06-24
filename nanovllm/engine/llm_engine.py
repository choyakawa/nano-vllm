import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import uuid

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fileds = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fileds}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        self.im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        if sampling_params.cache_group_id is None:
            sampling_params.cache_group_id = str(uuid.uuid4())

        seq = Sequence(prompt, sampling_params, self.im_start_id)
        self.scheduler.add(seq)
        return sampling_params.cache_group_id

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0

        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

        outputs = [
            (seq.seq_id, seq.completion_token_ids, seq.cache_group_id)
            for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) - seq.num_cached_tokens for seq in seqs) if is_prefill else len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        request_map = {}
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            sp_copy = SamplingParams(**sp.__dict__)
            cache_group_id = self.add_request(prompt, sp_copy)
            request_map[self.scheduler.waiting[-1].seq_id] = (i, cache_group_id)

        outputs_by_index = [None] * len(prompts)

        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t) if (perf_counter() -t) > 0 else 0
                else:
                    decode_throughput = num_tokens / (perf_counter() - t) if (perf_counter() -t) > 0 else 0
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids, out_cache_group_id in output:
                original_index, _ = request_map[seq_id]
                outputs_by_index[original_index] = {
                    "text": self.tokenizer.decode(token_ids),
                    "token_ids": token_ids,
                    "cache_group_id": out_cache_group_id,
                }
                if use_tqdm:
                    pbar.update(1)
        if use_tqdm:
            pbar.close()

        for i in range(len(prompts)):
            if outputs_by_index[i] is None:
                _, cache_group_id = request_map[i]
                outputs_by_index[i] = {"text": "", "token_ids": [], "cache_group_id": cache_group_id, "error": "Generation failed"}

        return outputs_by_index
