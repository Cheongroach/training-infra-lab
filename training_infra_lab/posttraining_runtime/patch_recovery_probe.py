"""Enable our rank-safe post-save probe on both TP ranks in a disposable worker."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

MILES_SHA = '2233d795dcf167be489d1859971c9233393f04c3'
RELATIVE_PATH = 'miles/backends/megatron_utils/actor.py'
ORIGINAL_SHA = 'f9cf84fb407a83a3ca277e03a8668b4848eaf110b99f7916d790cc4dcede0bd6'
BEFORE = 'if self.args.custom_megatron_post_save_hook_path is not None and dist.get_rank() == 0:'
AFTER = 'if self.args.custom_megatron_post_save_hook_path is not None:'
LOAD_BEFORE = '        self._init_training_state()\n'
LOAD_AFTER = '''        if (load_output.start_rollout_id == 2
                and self.args.custom_megatron_before_train_step_hook_path == "recovery_checks.before_step"):
            from recovery_checks import after_load
            after_load(self.args, load_output.start_rollout_id,
                       self.model, self.optimizer, self.opt_param_scheduler)
        self._init_training_state()
'''
TRAIN_SHA = '650eb449c44842a87c925f272b127d46f9b93b7a6498493bc326294ea0116571'
CHECK_BEFORE = '''    if args.check_weight_update_equal:
        await inference_controller.check_weights(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )
'''
CHECK_AFTER = '''    if args.check_weight_update_equal:
        if (args.start_rollout_id == 2
                and args.rollout_function_path == "recoverable_rollout.CheckpointedBoundedRolloutFn"):
            from resume_publication_guard import verify_initial_resume
            await verify_initial_resume(args, inference_controller)
        else:
            await inference_controller.check_weights(
                action="compare",
                allow_quant_error=args.check_weight_update_allow_quant_error,
                selector=args.check_weight_update_selector,
                skip_list=args.check_weight_update_skip_list,
            )
'''
EXIT_BEFORE = '''            break


if __name__ == "__main__":
'''
EXIT_AFTER = '''            if (rollout_id == 1 and args.start_rollout_id == 0
                    and args.rollout_function_path == "recoverable_rollout.CheckpointedBoundedRolloutFn"):
                from resume_publication_guard import capture_committed_engines
                await capture_committed_engines(args, inference_controller)
            break


if __name__ == "__main__":
'''


def patched_source(content):
    if hashlib.sha256(content.encode()).hexdigest() != ORIGINAL_SHA or content.count(BEFORE) != 1:
        raise ValueError('Unexpected Miles actor source; recovery probe patch refused')
    if content.count(LOAD_BEFORE) != 1:
        raise ValueError('Unexpected post-load fingerprint location')
    return content.replace(BEFORE, AFTER, 1).replace(LOAD_BEFORE, LOAD_AFTER, 1)


def patched_driver(content):
    if (hashlib.sha256(content.encode()).hexdigest() != TRAIN_SHA
            or content.count(CHECK_BEFORE) != 1 or content.count(EXIT_BEFORE) != 1):
        raise ValueError('Unexpected Miles train driver; recovery guard patch refused')
    return content.replace(CHECK_BEFORE, CHECK_AFTER, 1).replace(EXIT_BEFORE, EXIT_AFTER, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/root/miles'))
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    sha = subprocess.check_output(['git', '-C', str(a.root), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != MILES_SHA:
        raise ValueError('Unexpected Miles commit')
    path = a.root / RELATIVE_PATH
    original = path.read_text()
    modified = patched_source(original)
    driver = a.root / 'train.py'
    driver_source = driver.read_text()
    driver_modified = patched_driver(driver_source)
    path.write_text(modified)
    path.write_text(modified)
    driver.write_text(driver_modified)
    a.report.write_text(json.dumps({'base_commit': sha, 'path': RELATIVE_PATH,
        'before_sha256': ORIGINAL_SHA, 'after_sha256': hashlib.sha256(modified.encode()).hexdigest(),
        'driver_path': 'train.py', 'driver_before_sha256': TRAIN_SHA,
        'driver_after_sha256': hashlib.sha256(driver_modified.encode()).hexdigest(),
        'purpose': 'Rank-safe load/save fingerprints and checkpoint-bound raw engine SHA256 guard',
        'gpu_verified': False}, indent=2))


if __name__ == '__main__':
    main()
