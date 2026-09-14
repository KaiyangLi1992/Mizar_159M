import os
from datetime import datetime
import logging
import sys
from pathlib import Path

import distributed
from utils.launch_utils import parse_args, multiprocessing_init, parse_mode
from training.log import configure_logging
from training.trainer import Trainer, TrainerMode

def _checkpoint_allowed(path):
    min_optstep = int(os.environ.get("MELLOW_AUTO_RESUME_MIN_OPTSTEP", "0") or 0)
    if min_optstep <= 0:
        return True
    name = Path(path).name
    if "-complete.ckpt" in name:
        return True
    marker = "optstep-"
    if marker not in name:
        return False
    step_text = name.split(marker, 1)[1].split(".", 1)[0]
    try:
        return int(step_text) >= min_optstep
    except ValueError:
        return False


def _find_latest_training_state_checkpoint(run_save_dir):
    run_save_dir = Path(run_save_dir)
    pointer_path = run_save_dir / "latest_training_state.txt"
    if pointer_path.is_file():
        try:
            pointed = Path(pointer_path.read_text().splitlines()[0])
            if pointed.is_file() and _checkpoint_allowed(pointed):
                return str(pointed)
        except Exception:
            pass

    if not run_save_dir.is_dir():
        return None

    candidates = [
        path for path in run_save_dir.glob("*/training-state-*.ckpt")
        if _checkpoint_allowed(path)
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return str(latest)

def _has_explicit_resume_arg():
    for arg in sys.argv[1:]:
        if arg == "--resume_checkpoint" or arg.startswith("--resume_checkpoint="):
            return True
    return False

def _maybe_auto_resume_latest(args):
    if args.mode is not TrainerMode.Train:
        return
    if os.environ.get("MELLOW_AUTO_RESUME_LATEST", "1").lower() in {"0", "false", "off", "no"}:
        return
    if _has_explicit_resume_arg():
        return

    latest = _find_latest_training_state_checkpoint(args.save_dir)
    if latest is None:
        return

    args.resume_checkpoint = latest
    if isinstance(args.model, dict):
        args.model["resume_checkpoint"] = latest
    print(f"Auto-resume checkpoint: {latest}", flush=True)

def main():
    args, conf = parse_args()
    args = parse_mode(args)
    _maybe_auto_resume_latest(args)
    args.job_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.save_dir = os.path.join(args.save_dir, args.job_id)
    args.save_adir = os.path.join(args.save_dir, "audio")
    # create output folder
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_adir, exist_ok=True)

    configure_logging()
    multiprocessing_init()

    if args.distributed_backend is None:
        distributed_ctx = distributed.get_local_context()
    else:
        import distributed.torch as impl
        distributed_ctx = impl.TorchDistributedContext(args.distributed_backend)

    with distributed_ctx:
        # Suppress logging from non-zero ranks IMMEDIATELY after distributed context is initialized
        # This prevents duplicate "Proceeding with config" and other messages
        if distributed_ctx.rank() > 0:
            logging.getLogger().setLevel(logging.ERROR)  # Changed to ERROR to suppress warnings too
        
        # copy configs to output folder (only rank 0)
        if distributed_ctx.rank() == 0:
            import shutil
            shutil.copy(conf, os.path.join(args.save_dir, "conf.yaml"))

        logging.info(f"Proceeding with config {vars(args)}")

        with Trainer(vars(args), distributed_ctx=distributed_ctx) as trainer:
            # noinspection PyBroadException
            try:
                return run_command(trainer, args)
            except Exception:
                # make sure to print error to per-process log file as well, not only to console output
                logging.error("error running command", exc_info=sys.exc_info())
                if args.reraise_exceptions:
                    raise

                return -1

def run_command(trainer: Trainer, args):
    if args.mode is TrainerMode.Train:
        trainer.train()
    elif args.mode is TrainerMode.EvaluateCheckpoint:
        trainer.evaluate_checkpoint()
    else:
        assert False, f"Unknown operation mode {args.mode}"

if __name__ == "__main__":
    exit(main())
