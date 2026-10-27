"""Miles plugin; keep the same rollout ID and policy version during qualification."""
from pathlib import Path

from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder

from qualification import qualify


class BoundedRolloutFn(InferenceRolloutFn):
    def __init__(self, input):
        super().__init__(input)
        self.seen_indices = set()
        args = self.state.args
        if (args.dynamic_sampling_filter_path or args.rollout_sample_filter_path
                or args.partial_rollout or args.over_sampling_batch_size != args.rollout_batch_size):
            raise ValueError("bounded qualification requires unfiltered complete synchronous batches")

    async def _call_train(self, input):
        args = self.state.args

        async def next_batch():
            issued = []
            original = self.data_source.get_samples

            def limited_get_samples(num_samples):
                if issued or num_samples != args.rollout_batch_size:
                    # Upstream retries swallowed task failures; prevent an unbounded inner scan.
                    raise RuntimeError("GRPO_CANDIDATE_ISSUANCE_LIMIT")
                groups = original(num_samples)
                issued.extend({"index": s.index, "group_index": s.group_index}
                              for group in groups for s in group)
                return groups

            self.data_source.get_samples = limited_get_samples
            try:
                output = await super(BoundedRolloutFn, self)._call_train(input)
            finally:
                self.data_source.get_samples = original
            samples = [s.to_dict() for group in output.samples for s in group]
            return output, samples, issued

        def on_drop(output):
            SampleOwnershipRecorder.log_dropped_groups(
                args=args, before=output.samples, after=[], reason="bounded_qualification",
            )

        return await qualify(
            next_batch, root=Path(args.save_debug_rollout_data).parent,
            rollout_id=input.rollout_id, weight_version=input.weight_version,
            seen=self.seen_indices, prompts=args.rollout_batch_size,
            answers=args.n_samples_per_prompt, max_response=args.rollout_max_response_len,
            on_drop=on_drop,
        )
