"""Synchronous smoke plugin: retain sampler state across a Miles checkpoint load."""
from pathlib import Path

from bounded_rollout import BoundedRolloutFn
from qualification_checkpoint import CURSOR_KEYS, digest, restore_journal, save_journal

CONTRACT_KEYS = (
    'seed', 'rollout_seed', 'rollout_batch_size', 'n_samples_per_prompt',
    'global_batch_size', 'rollout_max_response_len', 'rollout_temperature',
    'rollout_top_p', 'rollout_top_k', 'rm_type', 'reward_key',
    'apply_chat_template', 'apply_chat_template_kwargs', 'advantage_estimator',
    'eps_clip', 'eps_clip_high', 'kl_loss_coef', 'entropy_coef', 'lr',
    'tensor_model_parallel_size', 'pipeline_model_parallel_size', 'context_parallel_size',
)


def recovery_contract(args):
    code = Path(args.prompt_data).parent
    names = ('smoke-prompts.jsonl', 'model-manifest.json', 'run_smoke.py',
             'qualification.py', 'rollout_gate.py', 'bounded_rollout.py',
             'qualification_checkpoint.py', 'recoverable_rollout.py', 'recovery_checks.py')
    return {'sources': {name: digest((code / name).read_bytes()) for name in names},
            'training': {key: getattr(args, key) for key in CONTRACT_KEYS}}


class CheckpointedBoundedRolloutFn(BoundedRolloutFn):
    def __init__(self, input):
        super().__init__(input)
        args = self.state.args
        if getattr(args, 'fully_async', False) or getattr(args, 'indep_dp', False):
            raise ValueError('Checkpointed smoke requires one synchronous learner cell')
        self.last_accepted_rollout = -1
        self.contract = recovery_contract(args)

    def _cursor(self):
        if self.data_source.get_buffer_length() not in (None, 0):
            raise ValueError('Checkpointed smoke cannot preserve partial pending groups')
        return {key: getattr(self.data_source, key) for key in CURSOR_KEYS}

    async def _call_train(self, input):
        if input.rollout_id != self.last_accepted_rollout + 1:
            raise ValueError('Restored sampler and requested rollout disagree')
        output = await super()._call_train(input)
        self.last_accepted_rollout = input.rollout_id
        return output

    def save(self, directory):
        root = Path(self.state.args.save_debug_rollout_data).parent
        save_journal(root / 'qualification', directory,
                     completed=self.last_accepted_rollout, cursor=self._cursor(),
                     seen=self.seen_indices, contract=self.contract)

    def load(self, directory):
        root = Path(self.state.args.save_debug_rollout_data).parent
        state, seen = restore_journal(directory, root, cursor=self._cursor(), contract=self.contract)
        self.last_accepted_rollout = state['completed_rollout_id']
        self.seen_indices = seen
