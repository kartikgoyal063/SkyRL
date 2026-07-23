import math
import os
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ray
import torch
from jaxtyping import Float
from loguru import logger
from ray.util.placement_group import placement_group
from tqdm import tqdm
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.distributed.dispatch import (
    ActorInfo,
    MeshRank,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.training_batch import (
    TensorList,
    TrainingInputBatch,
    pad_training_input_batch,
)
from skyrl.backends.skyrl_train.utils import ppo_utils
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    AdaptiveKLController,
    FixedKLController,
    apply_loss_reduction_to_advantages_minibatch,
    compute_approx_kl,
    get_kl_controller,
)
from skyrl.backends.skyrl_train.utils.sdpo import (
    _left_pad_concat,
    build_self_distillation_tensors,
    build_sdpo_feedback,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.backends.skyrl_train.workers.worker_utils import reduce_metrics
from skyrl.env_vars import SKYRL_RAY_PG_TIMEOUT_IN_S
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset import PromptDataset
from skyrl.train.dataset.preprocess import (
    compute_prompt_boundaries,
    compute_prompt_mini_batch_boundaries,
    convert_prompts_responses_to_batch_tensors,
)
from skyrl.train.evaluate import evaluate, evaluate_step_wise
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
)
from skyrl.train.generators.utils import (
    get_metrics_from_generator_output,
    merge_stepwise_output,
    prepare_generator_input,
)
from skyrl.train.utils import (
    Timer,
    get_ray_pg_ready_with_timeout,
    trainer_utils,
)
from skyrl.train.utils.callbacks import (
    CallbackHandler,
    CallbackInput,
    TrainingCallback,
    TrainingControl,
)
from skyrl.train.utils.logging_utils import log_example
from skyrl.train.utils.ray_gpu_monitor import RayGpuMonitor
from skyrl.train.utils.tracking import Tracking
from skyrl.train.utils.trainer_utils import (
    GLOBAL_STEP_PREFIX,
    DynamicSamplingState,
    ResumeMode,
    build_dataloader,
    cleanup_old_checkpoints,
    dump_train_rollouts,
    extract_step_from_path,
    run_on_each_node,
    validate_consistency_for_latest_checkpoint,
    validate_generator_output,
    zero_variance_filter,
)
from skyrl.train.utils.utils import ResolvedPlacementGroup, configure_ray_worker_logging
from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper


class RayPPOTrainer:
    def __init__(
        self,
        cfg: SkyRLTrainConfig,
        tracker: Tracking,
        tokenizer: AutoTokenizer,
        train_dataset: Optional[PromptDataset],
        inference_engine_client: InferenceEngineClient,
        generator: GeneratorInterface,
        colocate_pg: Optional[ResolvedPlacementGroup] = None,
        eval_dataset: Optional[PromptDataset] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
    ):
        self.cfg = cfg
        self.colocate_all = cfg.trainer.placement.colocate_all
        self.tracker = tracker
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.inference_engine_client = inference_engine_client
        self.generator = generator
        self.train_dataloader = None
        self.total_training_steps = None
        self._build_train_dataloader_and_compute_training_steps()

        self.eval_dataloader = (
            build_dataloader(self.cfg, eval_dataset, is_train=False) if eval_dataset is not None else None
        )
        self.colocate_pg = colocate_pg

        self.resume_mode = ResumeMode(cfg.trainer.resume_mode)

        self.all_metrics = {}
        self.all_timings = {}
        self.global_step = 0

        self._vllm_metrics_scraper: Optional[VLLMMetricsScraper] = (
            VLLMMetricsScraper() if cfg.generator.inference_engine.enable_ray_prometheus_stats else None
        )

        self._ray_gpu_monitor = RayGpuMonitor() if cfg.trainer.enable_ray_gpu_monitor else None

        # initialized in `build_models`
        self.policy_model: PPORayActorGroup = None
        self.critic_model: Optional[PPORayActorGroup] = None
        self.ref_model: Optional[PPORayActorGroup] = None
        # used for checkpoint cleanup
        self._node_ids: Optional[List[str]] = None

        self.dynamic_sampling_state: Optional[DynamicSamplingState] = None

        self.reward_kl_controller: Optional[Union[FixedKLController, AdaptiveKLController]] = None
        self.dispatch: WorkerDispatch = None

        self._callback_handler = CallbackHandler(callbacks)
        self._training_control = TrainingControl()
        self._current_epoch: int = 0

        configure_ray_worker_logging()

        self._num_training_gpus = (
            cfg.trainer.placement.policy_num_gpus_per_node * cfg.trainer.placement.policy_num_nodes
        )

    def add_callback(self, callback: TrainingCallback) -> None:
        """Register a callback. Events fired after this call reach the new callback."""
        self._callback_handler.add(callback)

    def _build_callback_input(self, **fields) -> CallbackInput:
        """Snapshot loop counters + per-event fields into a CallbackInput."""
        steps_per_epoch = len(self.train_dataloader) if self.train_dataloader is not None else 0
        total_steps = self.total_training_steps or 0
        return CallbackInput(
            global_step=self.global_step,
            epoch=self._current_epoch,
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
            **fields,
        )

    def _fire(self, event_name: str, **fields) -> None:
        """Build a CallbackInput and dispatch the given event to all callbacks."""
        cb_input = self._build_callback_input(**fields)
        getattr(self._callback_handler, event_name)(self, cb_input, self._training_control)

    @property
    def has_critic(self) -> bool:
        """Check if critic model is configured."""
        return bool(self.cfg.trainer.critic.model.path)

    def _build_train_dataloader_and_compute_training_steps(self):
        """
        Hook for constructing the training dataloader. Subclasses can override
        this to customize dataloader behavior. For instance, fully async training
        needs a batch size of 1, among other features.
        Defaults to `trainer_utils.build_dataloader` with `is_train=True`.
        When train_dataset is None (e.g. Tinker backend provides data externally),
        the dataloader is not built.
        """
        if self.train_dataset is not None:
            self.train_dataloader = build_dataloader(self.cfg, self.train_dataset, is_train=True)
            self.total_training_steps = len(self.train_dataloader) * self.cfg.trainer.epochs

    @torch.no_grad()
    async def eval(self) -> Dict[str, float]:
        """
        Run generation and scoring on the evaluation dataset.

        The eval metrics are recorded after having finished training `self.global_step` steps.
        Metrics recorded in global_step 0 corresponds to evaluations before training.

        Returns:
            A dictionary of evaluation metrics.
        """
        if self.cfg.generator.step_wise_trajectories:
            eval_metrics = await evaluate_step_wise(
                eval_dataloader=self.eval_dataloader,
                generator=self.generator,
                cfg=self.cfg,
                global_step=self.global_step,
                tokenizer=self.tokenizer,
            )
        else:
            eval_metrics = await evaluate(
                eval_dataloader=self.eval_dataloader,
                generator=self.generator,
                cfg=self.cfg,
                global_step=self.global_step,
                tokenizer=self.tokenizer,
            )
        return eval_metrics

    async def train(self):
        """
        Main training loop for PPO
        """
        if self._ray_gpu_monitor is not None:
            self._ray_gpu_monitor.start()

        # Initialize weight sync state between policy model and inference engines.
        with Timer("init_weight_sync_state"):
            self.init_weight_sync_state()

        # Load checkpoint state if resumption is enabled.
        if self.resume_mode != ResumeMode.NONE:
            with Timer("load_checkpoints"):
                self.global_step, _ = self.load_checkpoints()

        # Prepare weights for sampling
        with Timer("sync_weights"):
            await self.dispatch.save_weights_for_sampler()

        # Compute start_epoch up-front so callback metadata is ready before
        # any event fires (including the baseline eval below).
        start_epoch = self.global_step // len(self.train_dataloader)
        self._current_epoch = start_epoch
        self._training_control.reset()

        self._fire("on_train_start")

        # Eval before training. Wrapped in eval callbacks + on_log so that e.g.
        # a best-checkpoint callback sees the baseline reading.
        if self.cfg.trainer.eval_interval > 0 and (self.cfg.trainer.eval_before_train or self.cfg.trainer.eval_only):
            self._fire("on_eval_start")
            with Timer("eval", self.all_timings):
                eval_metrics = await self.eval()
            self._fire("on_eval_end", metrics=eval_metrics)
            self._fire("on_log", logs=eval_metrics)
            self.tracker.log(eval_metrics, step=self.global_step, commit=True)

        # eval-only: the baseline eval above already dumped trajectories + metrics; stop here so a
        # no-training pass doesn't spend rollout/user-sim budget on gradient steps we'd discard.
        if self.cfg.trainer.eval_only:
            logger.info("trainer.eval_only=True: baseline eval complete; exiting before training.")
            if self.colocate_all:
                await self.inference_engine_client.sleep()
            return

        # initialize kl controller
        if self.cfg.trainer.algorithm.use_kl_in_reward:
            self.reward_kl_controller = get_kl_controller(self.cfg.trainer.algorithm)

        # main training loop
        pbar = tqdm(total=self.total_training_steps, initial=self.global_step, desc="Training Batches Processed")
        self.global_step += 1  # start training at global_step 1

        # booleans tracking whether we save ckpts
        # as well as hf model at step end
        will_save_ckpts = False
        hf_model_save = False
        for epoch in range(start_epoch, self.cfg.trainer.epochs):
            self._current_epoch = epoch
            self._fire("on_epoch_start")
            # ``step_started`` tracks the on_step_start/on_step_end pairing taking
            # dynamic-sampling into account (which span multiple inner iterations
            # before completing a logical step).
            step_started = False
            for _, rand_prompts in enumerate(self.train_dataloader):
                if not step_started:
                    self._fire("on_step_start")
                    step_started = True
                with Timer("step", self.all_timings):
                    # for colocate_all=true, inference engine is always on GPU when starting the training step

                    # 0. truncate data to have even shards
                    rand_prompts = self._remove_tail_data(rand_prompts)
                    generator_input, uids = prepare_generator_input(
                        rand_prompts,
                        self.cfg.generator.n_samples_per_prompt,
                        get_sampling_params_for_backend(
                            self.cfg.generator.inference_engine.backend, self.cfg.generator.sampling_params
                        ),
                        self.cfg.environment.env_class,
                        "train",
                        self.global_step,
                    )

                    # 1.1. generation phase
                    with Timer("generate", self.all_timings):
                        generator_output: GeneratorOutput = await self.generate(generator_input)

                    if self.cfg.generator.step_wise_trajectories:
                        # NOTE: We use instance_ids from `trajectory_ids` here instead of re-using `uids`
                        # this is because in step-wise training, len(uids) != len(generator_output["response_ids"])
                        uids = [trajectory_id.instance_id for trajectory_id in generator_output["trajectory_ids"]]

                    # dynamic sampling
                    if self.cfg.trainer.algorithm.dynamic_sampling.type is not None:
                        generator_output, uids, keep_sampling = self.handle_dynamic_sampling(generator_output, uids)
                        if keep_sampling:  # continue sampling
                            # update progress bar for current batch (but not global step)
                            pbar.update(1)
                            continue

                    # HERO: run the reviewer (LOCATE/reflect) while the engine is still awake (must
                    # precede the colocate sleep below). Skipped for gold+top_level (no located turns
                    # needed). Produces per-turn hints consumed by the HERO teacher-data pipeline.
                    self._hero_hints = None
                    if self._hero_enabled() and self._hero_needs_reviewer():
                        with Timer("hero_reflect", self.all_timings):
                            self._hero_hints = await self._generate_hero_reflections(
                                generator_input, generator_output
                            )

                    if self.colocate_all:
                        # if we are not continuing sampling, we sleep the inference engine
                        await self.inference_engine_client.sleep()

                    # 1.2 postprocess rewards (and merge step-wise turns if enabled)
                    with Timer("postprocess_generator_output", self.all_timings):
                        generator_output, uids = self.postprocess_generator_output(generator_output, uids)

                    # 1.3 dump readable TRAIN rollouts (same row shape as the eval dumps) — persists the
                    # env_metrics.messages that ride every rollout's generator_output. For HERO the
                    # training_input is later replaced by per-turn fragments, so we dump here (full
                    # trajectories) before that transform.
                    if self.cfg.trainer.dump_train_rollouts:
                        with Timer("dump_train_rollouts", self.all_timings):
                            dump_train_rollouts(
                                Path(self.cfg.trainer.export_path) / "dumped_train_rollouts",
                                self.tokenizer,
                                generator_output,
                                uids,
                                self.global_step,
                            )

                    # 2. print example just for debugging
                    log_interval = self.cfg.trainer.log_example_interval
                    if log_interval > 0 and self.global_step % log_interval == 0:
                        vis = self.tokenizer.decode(generator_output["response_ids"][0])
                        log_example(
                            logger,
                            prompt=generator_input["prompts"][0],
                            response=vis,
                            reward=generator_output["rewards"][0],
                        )

                    # 3. Convert GeneratorOutput to TrainingInputBatch
                    with Timer("convert_to_training_input", self.all_timings):
                        training_input: TrainingInputBatch = self.convert_to_training_input(generator_output, uids)

                    # 3b. HERO: REPLACE the batch with the modular teacher-data pipeline's rows (per_turn
                    # or top_level); else the non-HERO SDPO path appends sibling teacher inputs.
                    if self._hero_enabled():
                        with Timer("build_hero_perturn_inputs", self.all_timings):
                            hero_ti = self._build_hero_training_input(
                                generator_output, generator_input["prompts"], generator_input.get("env_extras")
                            )
                        if hero_ti is None:
                            # no problematic turns this step -> nothing to distill; skip the update
                            # (mirror dynamic-sampling's skip; the logical step completes on a later batch).
                            pbar.update(1)
                            continue
                        training_input = hero_ti
                    elif self.cfg.trainer.algorithm.policy_loss_type == "sdpo":
                        with Timer("build_self_distillation_inputs", self.all_timings):
                            training_input = self._add_self_distillation_inputs(
                                training_input,
                                generator_output,
                                generator_input["prompts"],
                                uids,
                                generator_input.get("env_extras"),
                            )

                    # 4. Inference and calculate values, log probs, rewards, kl divergence
                    with Timer("fwd_logprobs_values_reward", self.all_timings):
                        training_input = self.fwd_logprobs_values_reward(training_input)

                    # 5. apply kl divergence penalty to rewards
                    if self.cfg.trainer.algorithm.use_kl_in_reward:
                        with Timer("apply_reward_kl_penalty", self.all_timings):
                            training_input = self.apply_reward_kl_penalty(training_input)

                    # 6. calculate advantages and returns
                    with Timer("compute_advantages_and_returns", self.all_timings):
                        training_input = self.compute_advantages_and_returns(training_input)
                        # remove some unwanted keys
                        for key in ["rewards"]:
                            training_input.pop(key)
                        training_input.metadata.pop("uids")
                        training_input.metadata.pop("is_last_step", None)

                    if self.cfg.trainer.dump_data_batch:
                        # dump data to file
                        with Timer("dump_data_batch"):
                            self.dump_data(training_input, file_name=f"global_step_{self.global_step}_training_input")

                    # 7. train policy/critic model
                    # Policy model is backloaded to GPU during training
                    with Timer("train_critic_and_policy", self.all_timings):
                        status = self.train_critic_and_policy(training_input)

                    self._fire("on_step_end", batch=training_input, metrics=status)
                    step_started = False

                    # Capture callback-driven triggers, then reset.
                    force_save = self._training_control.should_save
                    force_eval = self._training_control.should_evaluate
                    self._training_control.should_save = False
                    self._training_control.should_evaluate = False

                    # 8. conditionally save checkpoints and hf model
                    is_epoch_end = self.global_step % len(self.train_dataloader) == 0
                    # Only force a save on epoch boundaries if explicitly enabled. With short epochs
                    # (e.g. 2 steps/epoch) the unconditional epoch-end save piles up 30GB ckpts every
                    # 2 steps; gating it lets ckpt_interval/hf_save_interval be the sole trigger.
                    save_on_epoch_end = is_epoch_end and self.cfg.trainer.save_ckpt_on_epoch_end
                    hf_model_save = self.cfg.trainer.hf_save_interval > 0 and (
                        save_on_epoch_end or self.global_step % self.cfg.trainer.hf_save_interval == 0
                    )
                    ckpt_interval_save = self.cfg.trainer.ckpt_interval > 0 and (
                        save_on_epoch_end or self.global_step % self.cfg.trainer.ckpt_interval == 0
                    )
                    will_save_ckpts = force_save or ckpt_interval_save
                    if will_save_ckpts:
                        with Timer("save_checkpoints", self.all_timings):
                            ckpt_path = self.save_checkpoints()
                        self._fire("on_save", ckpt_path=ckpt_path)
                    if hf_model_save:
                        with Timer("save_hf_model", self.all_timings):
                            self.save_models()

                    # 9. conditionally sync policy and ref at the end of the epoch
                    if (
                        self.cfg.trainer.update_ref_every_epoch
                        and self.ref_model is not None
                        and is_epoch_end
                        and epoch != self.cfg.trainer.epochs - 1  # skip updating ref at the end of the last epoch
                    ):
                        with Timer("update_ref_with_policy", self.all_timings):
                            self.update_ref_with_policy()

                    # 10. Prepare weights for sampling
                    with Timer("sync_weights", self.all_timings):
                        await self.dispatch.save_weights_for_sampler()

                # 11. set logs
                logger.info(status)
                # Throughput metrics
                train_time = self.all_timings.get("train_critic_and_policy", 0.0)
                if train_time > 0 and training_input.get("attention_mask") is not None:
                    total_tokens = int(training_input["attention_mask"].sum().item())
                    self.all_metrics["trainer/tokens_per_second_per_gpu"] = total_tokens / (
                        train_time * self._num_training_gpus
                    )
                # log epoch info
                self.all_metrics.update({"trainer/epoch": epoch, "trainer/global_step": self.global_step})
                # Cumulative rollouts (trajectories) generated so far. This is the sample-efficiency
                # x-axis: it is comparable across runs with different batch sizes (step count is not).
                # Exact because the dataloader uses drop_last (every step is batch_size * n_samples
                # trajectories). Logged every step; W&B can use it as a custom x-axis. For runs that
                # predate this metric, the same value is recomputed from config at plot time (see
                # scripts/plot_rollouts.py) since train_batch_size and n_samples_per_prompt are in config.
                self.all_metrics["trainer/cumulative_rollouts"] = (
                    self.global_step
                    * self.cfg.trainer.train_batch_size
                    * self.cfg.generator.n_samples_per_prompt
                )
                interval_eval = self.cfg.trainer.eval_interval > 0 and (
                    self.global_step % self.cfg.trainer.eval_interval == 0
                    or self.global_step == self.total_training_steps
                )
                if force_eval or interval_eval:
                    self._fire("on_eval_start")
                    with Timer("eval", self.all_timings):
                        eval_metrics = await self.eval()
                        self.all_metrics.update(eval_metrics)
                    self._fire("on_eval_end", metrics=eval_metrics)

                log_payload = {
                    **self.all_metrics,
                    **{f"timing/{k}": v for k, v in self.all_timings.items()},
                }
                if self._vllm_metrics_scraper is not None:
                    log_payload.update(await self._vllm_metrics_scraper.sample())

                if self._ray_gpu_monitor is not None:
                    log_payload.update(self._ray_gpu_monitor.flush())

                self._fire("on_log", logs=log_payload)

                self.tracker.log(log_payload, step=self.global_step, commit=True)
                self.all_metrics = {}
                self.all_timings = {}

                # update progress bar after logging
                pbar.update(1)

                self.global_step += 1

                del training_input, generator_output

            self._fire("on_epoch_end")

        pbar.close()
        if self.colocate_all:
            await self.inference_engine_client.sleep()

        # Decrement global step by 1 to stop at the last global step
        # We use the global step value in callbacks when training finishes,
        # as well as for a final checkpoint save
        self.global_step -= 1

        # Safety net: always save final checkpoint at end of training.
        # Skip if we already saved at the last step
        if self.cfg.trainer.ckpt_interval > 0 and not will_save_ckpts:
            with Timer("save_checkpoints", self.all_timings):
                ckpt_path = self.save_checkpoints()
                logger.info("Saved final checkpoint.")
            self._fire("on_save", ckpt_path=ckpt_path)
        if self.cfg.trainer.hf_save_interval > 0 and not hf_model_save:
            with Timer("save_hf_model", self.all_timings):
                self.save_models()
                logger.info("Saved final model.")
        if self._vllm_metrics_scraper is not None:
            await self._vllm_metrics_scraper.aclose()

        if self._ray_gpu_monitor is not None:
            self._ray_gpu_monitor.stop()

        self._fire("on_train_end")
        self.tracker.finish()
        logger.info("Training done!")

    def _remove_tail_data(self, entries: List[Any]) -> List[Any]:
        """Remove tail data to have even shards in terms of *effective* samples.

        Each prompt produces `n_samples_per_prompt` samples. For data-parallel
        training we care that the total number of samples is nicely splittable
        across the (combined) data-parallel size of all enabled models.
        """
        lcm_dp_size = self.dispatch.get_lcm_dp_size()

        n_samples_per_prompt = self.cfg.generator.n_samples_per_prompt

        # We want the largest m <= len(entries) such that:
        #   (m * n_samples_per_prompt) % lcm_dp_size == 0
        #
        # Let g = gcd(lcm_dp_size, n_samples_per_prompt). Then this is equivalent
        # to requiring m to be a multiple of (lcm_dp_size / g).
        stride = lcm_dp_size // math.gcd(lcm_dp_size, n_samples_per_prompt)
        if stride <= 1:
            # Every prompt count is valid, keep all entries.
            return entries

        kept_prompts = (len(entries) // stride) * stride
        return entries[:kept_prompts]

    def build_models(self, PolicyWorker, CriticWorker, RefWorker):
        """
        Initialize the actors for training, and handle colocation logic
        """
        cfg = self.cfg
        pg = None

        use_ref_model = cfg.trainer.algorithm.use_kl_loss or cfg.trainer.algorithm.use_kl_in_reward

        if cfg.trainer.placement.colocate_all:
            num_policy_gpus = cfg.trainer.placement.policy_num_gpus_per_node * cfg.trainer.placement.policy_num_nodes
            num_critic_gpus = cfg.trainer.placement.critic_num_gpus_per_node * cfg.trainer.placement.critic_num_nodes
            num_ref_gpus = cfg.trainer.placement.ref_num_gpus_per_node * cfg.trainer.placement.ref_num_nodes
            ie_cfg = cfg.generator.inference_engine
            num_rollout_gpus = (
                ie_cfg.num_engines
                * ie_cfg.tensor_parallel_size
                * ie_cfg.pipeline_parallel_size
                * ie_cfg.data_parallel_size
            )
            assert (
                num_policy_gpus == num_rollout_gpus
            ), "num_policy_gpus and num_rollout_gpus must be the same when colocating all models"
            pg = self.colocate_pg

            policy_model = PPORayActorGroup(
                cfg.trainer,
                cfg.trainer.placement.policy_num_nodes,
                cfg.trainer.placement.policy_num_gpus_per_node,
                PolicyWorker,
                pg=pg,
                num_gpus_per_actor=0.2 if pg else 1,
                colocate_all=True,
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
                record_memory=cfg.trainer.policy.record_memory,
            )
            if use_ref_model:
                assert (
                    num_policy_gpus == num_ref_gpus
                ), "num_policy_gpus and num_ref_gpus must be the same when colocating policy and ref model"
                ref_model = PPORayActorGroup(
                    cfg.trainer,
                    cfg.trainer.placement.ref_num_nodes,
                    cfg.trainer.placement.ref_num_gpus_per_node,
                    RefWorker,
                    pg=pg,
                    num_gpus_per_actor=0.2 if pg else 1,
                    colocate_all=True,
                    sequence_parallel_size=cfg.trainer.ref.sequence_parallel_size,
                )
            else:
                ref_model = None

            if cfg.trainer.critic.model.path:
                assert (
                    num_policy_gpus == num_critic_gpus
                ), "num_policy_gpus and num_critic_gpus must be the same when colocating policy and critic model"
                critic_model = PPORayActorGroup(
                    cfg.trainer,
                    cfg.trainer.placement.critic_num_nodes,
                    cfg.trainer.placement.critic_num_gpus_per_node,
                    CriticWorker,
                    pg=pg,
                    num_gpus_per_actor=0.2,
                    colocate_all=True,
                    sequence_parallel_size=cfg.trainer.critic.sequence_parallel_size,
                )
            else:
                critic_model = None

        else:
            if cfg.trainer.placement.colocate_policy_ref and use_ref_model:
                assert (
                    cfg.trainer.placement.policy_num_nodes == cfg.trainer.placement.ref_num_nodes
                    and cfg.trainer.placement.policy_num_gpus_per_node == cfg.trainer.placement.ref_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate policy and ref model."

                bundles = [
                    {
                        "GPU": cfg.trainer.placement.policy_num_gpus_per_node,
                        "CPU": cfg.trainer.placement.policy_num_gpus_per_node,
                    }
                    for _ in range(cfg.trainer.placement.policy_num_nodes)
                ]
                raw_pg = placement_group(bundles, strategy="PACK")
                get_ray_pg_ready_with_timeout(raw_pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
                pg = ResolvedPlacementGroup(raw_pg)

            policy_model = PPORayActorGroup(
                cfg.trainer,
                cfg.trainer.placement.policy_num_nodes,
                cfg.trainer.placement.policy_num_gpus_per_node,
                PolicyWorker,
                pg=pg,
                num_gpus_per_actor=0.75 if pg else 1,
                colocate_all=False,
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
            )
            if use_ref_model:
                ref_model = PPORayActorGroup(
                    cfg.trainer,
                    cfg.trainer.placement.ref_num_nodes,
                    cfg.trainer.placement.ref_num_gpus_per_node,
                    RefWorker,
                    pg=pg,
                    num_gpus_per_actor=0.25 if pg else 1,
                    colocate_all=False,
                    sequence_parallel_size=cfg.trainer.ref.sequence_parallel_size,
                )
                if pg is not None:
                    # The shared policy/ref placement group `pg` is set only when colocate_policy_ref is enabled
                    logger.info(
                        "Colocating policy and ref on the same GPUs across "
                        f"{cfg.trainer.placement.policy_num_nodes} node(s)."
                    )
            else:
                ref_model = None

            if cfg.trainer.critic.model.path:
                critic_model = PPORayActorGroup(
                    cfg.trainer,
                    cfg.trainer.placement.critic_num_nodes,
                    cfg.trainer.placement.critic_num_gpus_per_node,
                    CriticWorker,
                    num_gpus_per_actor=1,
                    colocate_all=False,
                    sequence_parallel_size=cfg.trainer.critic.sequence_parallel_size,
                )
            else:
                critic_model = None

        policy_steps_per_train_batch = (
            cfg.trainer.train_batch_size // cfg.trainer.policy_mini_batch_size * cfg.trainer.update_epochs_per_batch
        )
        critic_steps_per_train_batch = 0
        if cfg.trainer.critic.model.path:
            critic_steps_per_train_batch = (
                cfg.trainer.train_batch_size // cfg.trainer.critic_mini_batch_size * cfg.trainer.update_epochs_per_batch
            )
        policy_num_training_steps = (
            self.total_training_steps * policy_steps_per_train_batch if self.total_training_steps is not None else None
        )
        critic_num_training_steps = (
            self.total_training_steps * critic_steps_per_train_batch if self.total_training_steps is not None else None
        )
        if not cfg.trainer.placement.colocate_all:
            refs = []
            if ref_model is not None:
                refs.extend(ref_model.async_init_model(cfg.trainer.ref.model.path))
            refs.extend(
                policy_model.async_init_model(
                    cfg.trainer.policy.model.path,
                    num_training_steps=policy_num_training_steps,
                )
            )
            if cfg.trainer.critic.model.path:
                refs.extend(
                    critic_model.async_init_model(
                        cfg.trainer.critic.model.path,
                        num_training_steps=critic_num_training_steps,
                    )
                )
            ray.get(refs)
            ray.get(policy_model.async_run_ray_method("pass_through", "_set_pad_token_id", self.tokenizer.pad_token_id))
        else:
            if ref_model is not None:
                ray.get(ref_model.async_init_model(cfg.trainer.ref.model.path))
                ref_model.offload_to_cpu()
            ray.get(
                policy_model.async_init_model(
                    cfg.trainer.policy.model.path,
                    num_training_steps=policy_num_training_steps,
                )
            )
            ray.get(policy_model.async_run_ray_method("pass_through", "_set_pad_token_id", self.tokenizer.pad_token_id))
            policy_model.offload_to_cpu()
            if cfg.trainer.critic.model.path:
                ray.get(
                    critic_model.async_init_model(
                        cfg.trainer.critic.model.path,
                        num_training_steps=critic_num_training_steps,
                    )
                )
                critic_model.offload_to_cpu()

        self.policy_model: PPORayActorGroup = policy_model
        self.critic_model: Optional[PPORayActorGroup] = critic_model
        self.ref_model: Optional[PPORayActorGroup] = ref_model

        # Create unified dispatch that manages all actor groups
        self.dispatch = WorkerDispatch(
            cfg=self.cfg,
            policy_actor_group=policy_model,
            critic_actor_group=critic_model,
            ref_actor_group=ref_model,
            inference_engine_client=self.inference_engine_client,
        )

        # Mark all models as offloaded if colocate_all (they were offloaded above)
        if self.colocate_all:
            self.dispatch.mark_all_offloaded()

        logger.info("init policy/ref/critic models done")

    def init_weight_sync_state(self):
        """
        Setup the connection between policy model and inference engine for weight syncing.
        """
        self.dispatch.init_weight_sync_state(self.inference_engine_client)
        logger.info("Initialized weight sync state for policy model and inference engines.")

    def convert_to_training_input(self, generator_output: GeneratorOutput, uids: List[str]) -> TrainingInputBatch:
        """Converts lists to a padded batch of tensors for training

        Args:
            generator_output (GeneratorOutput): Generated rollouts and associated data.
            uids (List[str]): List of prompt-unique identifiers for each generator ouput in the same
                order as `generator_output`. Used to identify which prompt each generated rollout belongs to.
        Returns:
            training_input (TrainingInputBatch): Padded batch of tensors for training. It preserves the
                order of `generator_output` and hence `uids`.
        """
        # 1. Extract generator output fields.
        prompt_ids: List[List[int]] = generator_output["prompt_token_ids"]
        response_ids: List[List[int]] = generator_output["response_ids"]
        rewards: List[List[float]] = generator_output["rewards"]
        loss_masks: List[List[int]] = generator_output["loss_masks"]

        logprobs: Optional[List[List[float]]] = generator_output.get("rollout_logprobs", None)
        rollout_expert_indices: Optional[List[List[List[List[int]]]]] = generator_output.get(
            "rollout_expert_indices", None
        )

        pixel_values = generator_output.get("pixel_values", None)
        image_grid_thw = generator_output.get("image_grid_thw", None)
        if pixel_values is not None:
            assert (
                pixel_values is not None and image_grid_thw is not None
            ), "Both pixel_values and image_grid_thw must exist for multi-modal inputs"
            assert len(pixel_values) == len(
                image_grid_thw
            ), "Number of pixel values should match number of image grid thw"
            pixel_values = TensorList(pixel_values)
            image_grid_thw = TensorList(image_grid_thw)

        # 2. Convert to tensors.
        (
            sequences_tensor,
            attention_masks_tensor,
            response_masks_tensor,
            rewards_tensor,
            loss_masks_tensor,
            rollout_logprobs_tensor,
            rollout_expert_indices_tensor,
        ) = convert_prompts_responses_to_batch_tensors(
            self.tokenizer,
            prompt_ids,
            response_ids,
            rewards,
            loss_masks,
            logprobs,
            rollout_expert_indices,
            max_seq_len=self.cfg.trainer.algorithm.max_seq_len,
        )

        # sanity check for off_policy_correction
        off_policy_correction = self.cfg.trainer.algorithm.off_policy_correction
        tis_ratio_type = off_policy_correction.tis_ratio_type
        sequence_mask_metric = off_policy_correction.sequence_mask_metric
        if tis_ratio_type is not None or sequence_mask_metric is not None:
            assert (
                rollout_logprobs_tensor is not None
            ), "expected non-null rollout logprobs tensor when off_policy_correction is enabled"
            assert rollout_logprobs_tensor.shape == loss_masks_tensor.shape, "Logprobs should look like responses"

        # 3. Create training input batch.
        training_input = TrainingInputBatch(
            {
                "sequences": sequences_tensor,  # Full trajectories (padded and concatenated prompts and responses)
                "attention_mask": attention_masks_tensor,
                "response_mask": response_masks_tensor,
                "rewards": rewards_tensor,
                "loss_mask": loss_masks_tensor,
                "rollout_logprobs": rollout_logprobs_tensor,
                "rollout_expert_indices": rollout_expert_indices_tensor,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
        )
        training_input.metadata = {"uids": uids}
        if generator_output.get("is_last_step", None) is not None:
            training_input.metadata["is_last_step"] = generator_output["is_last_step"]

        # 4. Compute mini-batch boundaries for train_critic_and_policy(). It excludes the ones
        # we will add in pad_training_input_batch().
        train_batch_size = self.cfg.trainer.train_batch_size
        n_samples_per_prompt = self.cfg.generator.n_samples_per_prompt
        is_stepwise = self.cfg.generator.step_wise_trajectories
        training_input.metadata["policy_mini_batch_boundaries"] = compute_prompt_mini_batch_boundaries(
            uids, self.cfg.trainer.policy_mini_batch_size, train_batch_size, is_stepwise, n_samples_per_prompt
        )
        # Per-prompt boundaries (used by the `prompt_mean` loss reduction). Policy-only,
        # since advantage normalization only applies to the policy.
        training_input.metadata["policy_prompt_boundaries"] = compute_prompt_boundaries(uids)
        if self.cfg.trainer.critic.model.path is not None:
            training_input.metadata["critic_mini_batch_boundaries"] = compute_prompt_mini_batch_boundaries(
                uids, self.cfg.trainer.critic_mini_batch_size, train_batch_size, is_stepwise, n_samples_per_prompt
            )

        # 5. Record metadata and metrics.
        training_input.metadata["response_length"] = response_masks_tensor.shape[1]
        batch_num_seq, batch_padded_seq_len = sequences_tensor.shape
        logger.info(f"batch_num_seq: {batch_num_seq}, batch_padded_seq_len: {batch_padded_seq_len}")
        self.all_metrics.update(
            {
                "generate/batch_num_seq": batch_num_seq,
                "generate/batch_padded_seq_len": batch_padded_seq_len,
            }
        )
        training_input.metadata["avg_response_length"] = sum(
            len(sample_response_ids) for sample_response_ids in response_ids
        ) / len(response_ids)

        # 6. Pad the batch, only needed for step-wise training's `fwd_logprobs_values_reward()`.
        logger.info(f"Number of sequences before padding: {len(training_input['sequences'])}")
        dp_size = self.dispatch.get_lcm_dp_size()
        pad_size = math.ceil(training_input.batch_size / dp_size) * dp_size - training_input.batch_size
        training_input = pad_training_input_batch(training_input, pad_size)
        logger.info(f"Number of sequences after padding: {len(training_input['sequences'])}")

        return training_input

    @torch.no_grad()
    async def generate(
        self,
        input_batch: GeneratorInput,
    ) -> GeneratorOutput:
        """
        Generate rollouts.

        If colocate_all is enabled:
        - before calling this method, the policy model should be on CPU and inference engine should
            be awake (i.e. on GPU).
        - after calling this method, the same model placement still holds.
        """
        # NOTE: we assume that .generate returns samples in the same order as passed in
        generator_output: GeneratorOutput = await self.generator.generate(input_batch)

        # add rollout metrics to self.all_metrics
        if generator_output["rollout_metrics"] is not None:
            self.all_metrics.update(generator_output["rollout_metrics"])
        generator_output.pop("rollout_metrics", None)

        validate_generator_output(
            len(input_batch["prompts"]),
            generator_output,
            step_wise=self.cfg.generator.step_wise_trajectories,
        )

        return generator_output

    @torch.no_grad()
    def postprocess_generator_output(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str]]:
        """
        Converts to per token rewards and computes pass@N.

        For step-wise training with ``merge_stepwise_output=true``, also collapses
        consecutive turns sharing a common prefix into a single sequence; ``uids``
        is shortened to match.

        In the future algorithm specific reward or loss mask post processing should be done here.

        Returns:
            (generator_output, uids) — uids may be shorter than the input when merging.
        """
        generator_output_for_metrics = generator_output
        uids_for_metrics = uids
        if self.cfg.generator.step_wise_trajectories:
            generator_output_for_metrics = defaultdict(list)
            for key in generator_output:
                if isinstance(generator_output[key], list):
                    generator_output_for_metrics[key] = [
                        generator_output[key][i]
                        for i in range(len(generator_output[key]))
                        if generator_output["is_last_step"][i]
                    ]
            uids_for_metrics = [
                uid for uid, is_last_step in zip(uids, generator_output["is_last_step"]) if is_last_step
            ]

        # only use `generator_output_for_metrics` for metrics calculation
        # For step-wise training, we only calculate metrics for the last step of each trajectory
        overall_metrics = get_metrics_from_generator_output(
            generator_output_for_metrics,
            uids_for_metrics,
        )

        # Prefix-aware merging of step-wise turns.
        if self.cfg.generator.merge_stepwise_output:
            assert self.cfg.generator.step_wise_trajectories, "merge_stepwise_output requires step-wise training"
            num_seq_before_merge = len(generator_output["response_ids"])
            generator_output = merge_stepwise_output(generator_output)
            num_seq_after_merge = len(generator_output["response_ids"])
            logger.info(f"Merged step wise: {num_seq_before_merge} sequences -> {num_seq_after_merge} sequences")
            self.all_metrics.update(
                {
                    "generate/num_seq_before_merge": num_seq_before_merge,
                    "generate/num_seq_after_merge": num_seq_after_merge,
                }
            )
            uids = [tid.instance_id for tid in generator_output["trajectory_ids"]]

        # these use the full generator output
        rewards: Union[List[float], List[List[float]]] = generator_output["rewards"]
        responses: List[List[int]] = generator_output["response_ids"]
        per_token_rewards: List[List[float]] = []

        # Check if rewards are already token-level (List[List[float]]) or response-level (List[float])
        if rewards and isinstance(rewards[0], list):
            # Token-level rewards: rewards is List[List[float]]
            per_token_rewards = rewards
        else:
            if self.cfg.trainer.algorithm.zero_variance_filter:
                kept_indices_set = set(zero_variance_filter(rewards, uids))
                generator_output["loss_masks"] = [
                    [0] * len(mask) if i not in kept_indices_set else mask
                    for i, mask in enumerate(generator_output["loss_masks"])
                ]
            # Response-level rewards: rewards is List[float], convert to per-token rewards
            for reward, response in zip(rewards, responses):
                per_token_reward = [0.0] * len(response)
                per_token_reward[-1] = float(reward)
                per_token_rewards.append(per_token_reward)

        n_samples_per_prompt = self.cfg.generator.n_samples_per_prompt

        reward_metrics = {
            f"reward/avg_pass_at_{n_samples_per_prompt}": overall_metrics["pass_at_n"],
            "reward/avg_raw_reward": overall_metrics["avg_score"],
            "reward/mean_positive_reward": overall_metrics["mean_positive_reward"],
        }
        self.all_metrics.update(reward_metrics)
        logger.info(
            f"reward/avg_pass_at_{n_samples_per_prompt}: {overall_metrics['pass_at_n']}, reward/avg_raw_reward: {overall_metrics['avg_score']}, reward/mean_positive_reward: {overall_metrics['mean_positive_reward']}"
        )
        # re-assign reward but now it's per token rewards
        generator_output["rewards"] = per_token_rewards
        return generator_output, uids

    @torch.no_grad()
    def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """Calculate advantages and returns for the data batch.

        Expects:
            - `["sequences"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["response_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["loss_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["values"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["rewards"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `.metadata["uids"]`: List[str]
            - `.metadata["is_last_step"]`: List[bool] for step-wise training

        Adds:
            - `["advantages"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["returns"]`: Float[torch.Tensor, "batch_size seqlen"]
        """
        token_level_rewards = data["rewards"]

        if self.cfg.generator.step_wise_trajectories:
            is_last_step = torch.tensor(data.metadata["is_last_step"], dtype=torch.bool)
            index = np.array(data.metadata["uids"])
            values = data["values"]
            # Step-wise only supports outcome-based estimators (GRPO, RLOO, MAXRL); ensured by `validate_cfg`.
            # We use the last step of each trajectory to compute advantages and broadcast them to
            # all steps of that trajectory, so we ignore per-step rewards in step-wise training.
            # We pass an all-ones mask here so the estimator returns the scalar advantage at every
            # position. The real per-step `response_mask` is re-applied on broadcast below.
            # Shapes:
            #   traj_ids, (batch_size,):         trajectory id per step (cumsum of shifted is_last_step)
            #   last_step_advantages/returns,
            #       (num_traj, seqlen):          scalar advantage/return per trajectory at every position
            #   last_step_advantages/returns[traj_ids],
            #       (batch_size, seqlen):        broadcast to every step of the owning trajectory
            #   response_mask_float,
            #       (batch_size, seqlen):        per-step response mask
            last_step_response_mask = data["response_mask"][is_last_step]
            last_step_advantages, last_step_returns = ppo_utils.compute_advantages_and_returns(
                token_level_rewards=token_level_rewards[is_last_step],
                response_mask=torch.ones_like(last_step_response_mask, dtype=torch.float),
                index=index[is_last_step.cpu().numpy()],
                adv_estimator=self.cfg.trainer.algorithm.advantage_estimator,
                values=values[is_last_step] if values is not None else None,
                config=self.cfg.trainer.algorithm,
                gamma=self.cfg.trainer.algorithm.gamma,
                lambd=self.cfg.trainer.algorithm.lambd,
                grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
            )
            traj_ids = (
                torch.cat([torch.tensor([False], device=is_last_step.device), is_last_step[:-1]]).int().cumsum(dim=0)
            )
            num_traj = traj_ids[-1].item() + 1
            assert num_traj == len(
                last_step_advantages
            ), f"num_traj {num_traj} doesn't match the number of trajectories as given by `is_last_step` {len(last_step_advantages)}. The `is_last_step` tensor is likely malformed"
            response_mask_float = data["response_mask"].to(last_step_advantages.dtype)
            advantages = last_step_advantages[traj_ids] * response_mask_float
            returns = last_step_returns[traj_ids] * response_mask_float
        else:
            advantages, returns = ppo_utils.compute_advantages_and_returns(
                token_level_rewards=token_level_rewards,
                response_mask=data["response_mask"],
                index=data.metadata["uids"],
                adv_estimator=self.cfg.trainer.algorithm.advantage_estimator,
                config=self.cfg.trainer.algorithm,
                values=data["values"],
                gamma=self.cfg.trainer.algorithm.gamma,
                lambd=self.cfg.trainer.algorithm.lambd,
                grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
            )
        data["returns"] = returns
        data["advantages"] = advantages

        # remove padding while calculating metrics
        pad_size = data.metadata.get("pad_size", 0)
        num_samples = len(token_level_rewards)

        return_sums = token_level_rewards.sum(dim=-1)[: num_samples - pad_size]
        if self.cfg.generator.step_wise_trajectories:
            avg_rewards: float = return_sums[is_last_step[: num_samples - pad_size]].mean().item()
        else:
            avg_rewards: float = return_sums.mean().item()

        avg_response_length = data.metadata["avg_response_length"]
        data = data.to("cpu")

        valid_advantages = torch.masked_select(
            data["advantages"][: num_samples - pad_size, ...], data["response_mask"][: num_samples - pad_size].bool()
        )
        avg_advantages: float = valid_advantages.mean().item()
        avg_advantages_abs: float = valid_advantages.abs().mean().item()

        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}
        data.metadata["metrics"].update(
            {
                "avg_final_rewards": avg_rewards,
                "avg_response_length": avg_response_length,
                "avg_advantages": avg_advantages,
                "avg_advantages_abs": avg_advantages_abs,
            }
        )

        logger.info(f"avg_final_rewards: {avg_rewards}, avg_response_length: {avg_response_length}")
        self.all_metrics.update(
            {
                "loss/avg_final_rewards": avg_rewards,
                "loss/avg_raw_advantages": avg_advantages,
                "loss/avg_raw_advantages_abs": avg_advantages_abs,
            }
        )
        return data

    def dump_data(self, data: TrainingInputBatch, file_name: str):
        """
        Dump data to pickle file
        """
        data_save_dir = Path(self.cfg.trainer.export_path) / "dumped_data"
        data_save_dir.mkdir(parents=True, exist_ok=True)
        data.save(data_save_dir / f"{file_name}.pkl")

    @torch.no_grad()
    def fwd_logprobs_values_reward(
        self,
        training_input: TrainingInputBatch,
    ):
        """
        Calculate values from the critic, log probs from the policy and ref model.

        Dispatch handles offload/backload automatically for all colocation configurations.

        Expects:
            - `["sequences"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["attention_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `.metadata["response_length"]`: Int

        Adds:
            - `["base_action_log_probs"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["action_log_probs"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["values"]`: Float[torch.Tensor, "batch_size seqlen"]
        """
        fwd_keys = ["sequences", "attention_mask"]
        if training_input.get("rollout_expert_indices") is not None:
            fwd_keys.append("rollout_expert_indices")
        if training_input.get("pixel_values") is not None:
            fwd_keys.append("pixel_values")
        if training_input.get("image_grid_thw") is not None:
            fwd_keys.append("image_grid_thw")
        data_fwd_pass = training_input.select(keys=fwd_keys, metadata_keys=["response_length"])

        values = None
        base_log_probs = None
        action_log_probs = None

        # Critic forward (dispatch handles offload/backload automatically)
        if self.has_critic:
            critic_output = self.dispatch.forward("critic", data_fwd_pass)
            values = loss_fn_outputs_to_tensor(critic_output.loss_fn_outputs, key="values")

        # Ref forward
        if self.ref_model is not None:
            ref_output = self.dispatch.forward("ref", data_fwd_pass)
            base_log_probs = loss_fn_outputs_to_tensor(ref_output.loss_fn_outputs, key="logprobs")
            self.dispatch.empty_cache("ref")

        # Policy forward
        policy_output = self.dispatch.forward("policy", data_fwd_pass)
        action_log_probs = loss_fn_outputs_to_tensor(policy_output.loss_fn_outputs, key="logprobs")

        # Empty cache after all forward passes
        self.dispatch.empty_cache()

        sequences_all: torch.Tensor = training_input["sequences"]
        # NOTE (sumanthrh): The slicing is needed to make sure that the batch dimension doesn't change for the tensordict.
        base_log_probs = base_log_probs[: len(sequences_all)] if base_log_probs is not None else None
        action_log_probs = action_log_probs[: len(sequences_all)]
        values = values[: len(sequences_all)] if values is not None else None

        training_input["base_action_log_probs"] = base_log_probs
        training_input["action_log_probs"] = action_log_probs
        training_input["values"] = values

        if training_input.get("rollout_logprobs", None) is not None:
            # calculates the difference in probs between inference and trainer components
            # only consider response tokens
            logprobs_diff = (
                training_input["rollout_logprobs"][training_input["loss_mask"] > 0]
                - action_log_probs[training_input["loss_mask"] > 0]
            ).abs()

            logprobs_diff_max = logprobs_diff.max().item()
            logprobs_diff_min = logprobs_diff.min().item()
            logprobs_diff_mean = logprobs_diff.mean().item()
            logprobs_diff_std = logprobs_diff.std().item()
            self.all_metrics.update(
                {
                    "policy/rollout_train_logprobs_abs_diff_max": logprobs_diff_max,
                    "policy/rollout_train_logprobs_abs_diff_min": logprobs_diff_min,
                    "policy/rollout_train_logprobs_abs_diff_mean": logprobs_diff_mean,
                    "policy/rollout_train_logprobs_abs_diff_std": logprobs_diff_std,
                }
            )
        return training_input

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Applies a penalty for KL divergence between the policy log probs and the base model log probs to the rewards."""
        loss_masks_all: torch.Tensor = data["loss_mask"]
        rewards: torch.Tensor = data["rewards"]
        base_action_log_probs: torch.Tensor = data["base_action_log_probs"]
        action_log_probs: torch.Tensor = data["action_log_probs"]

        # single batched computation
        with torch.no_grad():
            kl: Float[torch.Tensor, "batch_size seqlen"] = compute_approx_kl(  # type: ignore
                action_log_probs,
                base_action_log_probs,
                loss_mask=loss_masks_all,
                kl_estimator_type=self.cfg.trainer.algorithm.kl_estimator_type,
            )
        kl_max: Float[torch.Tensor, "batch_size"] = torch.max(kl.abs(), dim=-1)[0]  # noqa: F821
        kl_mean: Float[torch.Tensor, "batch_size"] = masked_mean(kl, loss_masks_all, dim=-1)  # noqa: F821

        # NOTE (erictang000): only supporting custom rewards currently
        kl_loss_coef = (
            self.reward_kl_controller.value
            if self.reward_kl_controller is not None
            else self.cfg.trainer.algorithm.kl_loss_coef
        )
        rewards = rewards - kl * max(0, kl_loss_coef)
        data["rewards"] = rewards

        avg_kl: float = kl_mean.mean().item()
        avg_kl_max: float = kl_max.mean().item()

        # update the kl controller
        if self.reward_kl_controller is not None:
            self.reward_kl_controller.update(current=avg_kl, n_steps=kl.shape[0])  # n_steps is just the batch size
        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}

        data.metadata["metrics"].update(
            {
                "avg_kl": avg_kl,
                "avg_kl_max": avg_kl_max,
                "kl_loss_coef": kl_loss_coef,
            }
        )

        self.all_metrics.update(
            {
                "loss/avg_kl": avg_kl,
                "loss/avg_kl_max": avg_kl_max,
                "loss/kl_loss_coef": kl_loss_coef,
            }
        )

        return data

    def _add_self_distillation_inputs(
        self,
        training_input: TrainingInputBatch,
        generator_output: GeneratorOutput,
        prompts: List,
        uids: List[str],
        env_extras: Optional[List] = None,
    ) -> TrainingInputBatch:
        """Attach SDPO teacher (hindsight-conditioned) inputs to the training batch.

        For each sample, picks a successful sibling from the same prompt group, builds a
        hindsight-augmented teacher prompt, and stores left-padded ``teacher_sequences`` +
        ``teacher_attention_mask`` + ``self_distillation_mask`` aligned with the (already padded)
        ``training_input``. Task-agnostic — all behavior is driven by ``algorithm.sdpo`` config.
        """
        response_ids = generator_output["response_ids"]
        num_real = len(response_ids)
        # Rebuilding the teacher prompt from chat messages needs per-sample prompt/response alignment,
        # which holds for non-step-wise, non-merged rollouts (e.g. SQL). Skip otherwise.
        if len(prompts) != num_real:
            logger.warning(
                f"[sdpo] prompts ({len(prompts)}) != responses ({num_real}); skipping self-distillation "
                "for this step (unsupported rollout mode, e.g. step-wise / merged multi-turn)."
            )
            return training_input

        # Per-sample scalar sequence reward (matches verl's reward_tensor.sum(-1)).
        seq_rewards = training_input["rewards"][:num_real].sum(dim=-1).tolist()

        sdpo_cfg = self.cfg.trainer.algorithm.sdpo
        # Environment feedback: tell each FAILED rollout why it failed (format / turn-exhaustion /
        # wrong-result), embedded top-level in the teacher's hindsight prompt. Failed samples with
        # no successful sibling still get mask=1 here (feedback-only conditioning), so no group is
        # discarded. Classification is on the response text (excludes the prompt's one-shot example).
        feedback = None
        if sdpo_cfg.include_environment_feedback:
            # Prefer env-rendered feedback carried per-sample in generator_output["env_metrics"]
            # under "sdpo_feedback" (multi-turn envs that have rich terminal state, e.g. tau2's
            # reward_info diagnosis). The string survives metric aggregation (non-numerics are
            # dropped from the aggregate, kept in the per-row dict). Falls back to the reward-sign /
            # response-text classifier (build_sdpo_feedback) for envs that don't supply it (e.g. SQL).
            env_metrics = generator_output.get("env_metrics")
            env_fb = None
            if env_metrics is not None and len(env_metrics) >= num_real:
                env_fb = [
                    (em.get("sdpo_feedback") if isinstance(em, dict) else None)
                    for em in env_metrics[:num_real]
                ]
            if env_fb is not None and any(f is not None for f in env_fb):
                feedback = env_fb
            else:
                completions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in response_ids]
                feedback = build_sdpo_feedback(seq_rewards, completions, sdpo_cfg)

        # OPSD-style gold conditioning: pull each sample's gold answer from env_extras
        # (reward_spec.ground_truth) so it can replace the sibling demonstration. Only consumed when
        # sdpo.use_ground_truth_demonstration is set; otherwise the builder ignores it.
        ground_truth = None
        if getattr(sdpo_cfg, "use_ground_truth_demonstration", False):
            if env_extras is not None and len(env_extras) >= num_real:
                ground_truth = [
                    (ee.get("reward_spec", {}) or {}).get("ground_truth") if isinstance(ee, dict) else None
                    for ee in env_extras[:num_real]
                ]
            else:
                logger.warning(
                    "[sdpo] use_ground_truth_demonstration=true but env_extras unavailable/misaligned "
                    f"(have {0 if env_extras is None else len(env_extras)}, need {num_real}); "
                    "no gold will be injected this step."
                )

        built = build_self_distillation_tensors(
            tokenizer=self.tokenizer,
            prompt_messages=prompts,
            response_ids=response_ids,
            seq_rewards=seq_rewards,
            uids=list(uids[:num_real]),
            cfg=sdpo_cfg,
            feedback=feedback,
            ground_truth=ground_truth,
        )

        # Pad teacher tensors up to the (already padded) batch size. Padded rows are masked out via
        # loss_mask, so their values are irrelevant; self_distillation_mask is padded with 0.
        pad_size = training_input.batch_size - num_real
        teacher_sequences = built["teacher_sequences"]
        teacher_attention_mask = built["teacher_attention_mask"]
        self_distillation_mask = built["self_distillation_mask"]
        if pad_size > 0:
            teacher_sequences = torch.cat([teacher_sequences, teacher_sequences[:1].repeat(pad_size, 1)], dim=0)
            teacher_attention_mask = torch.cat(
                [teacher_attention_mask, teacher_attention_mask[:1].repeat(pad_size, 1)], dim=0
            )
            self_distillation_mask = torch.cat(
                [self_distillation_mask, torch.zeros(pad_size, 1, dtype=self_distillation_mask.dtype)], dim=0
            )

        training_input["teacher_sequences"] = teacher_sequences
        training_input["teacher_attention_mask"] = teacher_attention_mask
        training_input["self_distillation_mask"] = self_distillation_mask

        for k, v in built["metrics"].items():
            self.all_metrics[k] = v
        return training_input

    # ── HERO faithful per-turn self-distillation (Axis-B) ──────────────────────

    def _hero_enabled(self) -> bool:
        """Modular HERO pipeline (reverie.hero) is active when policy_loss_type=='sdpo' and BOTH axis
        flags are set. Non-HERO sdpo (feedback_type/placement None) uses the sibling path."""
        algo = self.cfg.trainer.algorithm
        s = algo.sdpo
        return (
            algo.policy_loss_type == "sdpo"
            and getattr(s, "feedback_type", None) is not None
            and getattr(s, "placement", None) is not None
        )

    def _hero_needs_reviewer(self) -> bool:
        """Reviewer pass (LOCATE turns / WRITE diagnoses) is needed for everything except gold+top_level,
        which injects the whole-task gold with no located turns — so that combo skips the reviewer."""
        s = self.cfg.trainer.algorithm.sdpo
        return not (getattr(s, "feedback_type", None) == "gold" and getattr(s, "placement", None) == "top_level")

    async def _generate_hero_reflections(self, generator_input, generator_output) -> List[dict]:
        """In-loop self-reflection (HERO Chunk 3b): the CURRENT policy reflects on each fresh rollout →
        per-turn hints. Runs while the inference engine is awake (before the colocate sleep). Returns a
        list of ``{turnN: hint}`` dicts aligned to ``generator_output['response_ids']`` (``{}`` where
        the rollout has no parseable reflection)."""
        from reverie.offline.hero_perturn import (
            build_reflector_prompts,
            parse_reflector_hints,
            reflect_indices,
        )

        prompts = generator_input["prompts"]
        response_ids = generator_output["response_ids"]
        n = len(response_ids)
        if len(prompts) != n:
            logger.warning(f"[hero] prompts ({len(prompts)}) != responses ({n}); skipping reflection.")
            return [{} for _ in range(n)]

        hero_cfg = self.cfg.trainer.algorithm.sdpo
        response_texts = [self.tokenizer.decode(r, skip_special_tokens=False) for r in response_ids]
        raw_rewards = generator_output.get("rewards") or []
        seq_rewards = []
        for i in range(n):
            r = raw_rewards[i] if i < len(raw_rewards) else 0.0
            seq_rewards.append(float(sum(r)) if isinstance(r, (list, tuple)) else float(r))

        # HERO reflects on UNSUCCESSFUL attempts: by default skip successful trajectories entirely — no
        # reflector call (saves API cost) and no per-turn rows. `reflect_idx` maps subset -> full index.
        reflect_idx = reflect_indices(
            seq_rewards,
            success_threshold=float(getattr(hero_cfg, "success_reward_threshold", 1.0)),
            reflect_on_success=bool(getattr(hero_cfg, "hero_reflect_on_success", False)),
        )
        self.all_metrics["hero/n_reflected"] = len(reflect_idx)
        self.all_metrics["hero/frac_reflected"] = len(reflect_idx) / max(n, 1)

        hints: List[dict] = [{} for _ in range(n)]
        responses_full = ["(skipped: successful trajectory)"] * n
        if not reflect_idx:
            self.all_metrics["hero/reflector_parse_ok_frac"] = 0.0
            if getattr(hero_cfg, "hero_dump_reflections", False):
                self._dump_reflections(responses_full, hints, seq_rewards, response_texts, prompts)
            return hints

        sub_prompts = build_reflector_prompts(
            [prompts[i] for i in reflect_idx],
            [response_texts[i] for i in reflect_idx],
            [seq_rewards[i] for i in reflect_idx],
            include_think=hero_cfg.hero_reflector_thinking,
        )
        if getattr(hero_cfg, "hero_reflector_backend", "self_vllm") == "gemini":
            # External-critic reflector: generate hints with a strong API model (litellm) instead of the
            # in-loop policy. NOT HERO self-distillation. Downstream (parse / is_problematic / teacher) unchanged.
            sub_responses = await self._gemini_reflect(sub_prompts, hero_cfg)
        else:
            sub_responses = await self._vllm_reflect(sub_prompts, hero_cfg)

        sub_hints = parse_reflector_hints(sub_responses)
        for j, i in enumerate(reflect_idx):
            hints[i] = sub_hints[j]
            responses_full[i] = sub_responses[j]
        self.all_metrics["hero/reflector_parse_ok_frac"] = sum(1 for h in sub_hints if h) / len(reflect_idx)
        if getattr(hero_cfg, "hero_dump_reflections", False):
            self._dump_reflections(responses_full, hints, seq_rewards, response_texts, prompts)
        return hints

    async def _vllm_reflect(self, user_prompts, hero_cfg) -> list:
        """Self-vllm reflector: run the HERO reflector prompts through the colocated inference engine
        (the in-loop policy = HERO self-distillation). Returns raw response texts aligned to user_prompts."""
        chat_kwargs = {"enable_thinking": hero_cfg.hero_reflector_thinking}
        # Two-step (render text -> tokenize): apply_chat_template(tokenize=True) can return a
        # BatchEncoding (version-dependent), which isn't JSON-serializable for the engine call.
        prompt_token_ids = []
        for p in user_prompts:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, **chat_kwargs
            )
            prompt_token_ids.append(self.tokenizer(text, add_special_tokens=False)["input_ids"])

        from omegaconf import OmegaConf

        # Override temp/top_p/max-gen while PRESERVING the structured SamplingParams type (merge keeps
        # the first arg's type; to_container+create would demote it to a plain DictConfig and
        # get_sampling_params_for_backend rejects that).
        sp_cfg = OmegaConf.merge(
            self.cfg.generator.sampling_params,
            {
                "temperature": hero_cfg.hero_reflector_temperature,
                "top_p": hero_cfg.hero_reflector_top_p,
                "max_generate_length": hero_cfg.hero_reflector_max_tokens,
            },
        )
        reflector_sp = get_sampling_params_for_backend(
            self.cfg.generator.inference_engine.backend, sp_cfg
        )
        out = await self.inference_engine_client.generate(
            {"prompt_token_ids": prompt_token_ids, "sampling_params": reflector_sp}
        )
        return out["responses"]

    async def _gemini_reflect(self, user_prompts, hero_cfg) -> list:
        """External-critic reflector: run the same HERO reflector prompts through an API model (litellm),
        concurrently. temperature / max_tokens come from hero_cfg; reasoning is left at the model default.
        A failed call returns "" (-> empty hint -> that trajectory drops out of I(tau)). Reads GEMINI_API_KEY
        from env, loading it from the tau2 .env if absent (the trainer process may not have it set)."""
        import asyncio
        import litellm

        if not os.environ.get("GEMINI_API_KEY"):
            envp = os.path.expanduser("~/reverie/benchmarks/tau2-bench/.env")
            if os.path.exists(envp):
                for line in open(envp):
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

        sem = asyncio.Semaphore(int(getattr(hero_cfg, "hero_reflector_api_max_concurrency", 8)))

        async def _one(prompt: str) -> str:
            async with sem:
                try:
                    r = await asyncio.wait_for(
                        litellm.acompletion(
                            model=hero_cfg.hero_reflector_api_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=hero_cfg.hero_reflector_temperature,
                            max_tokens=hero_cfg.hero_reflector_max_tokens,
                        ),
                        timeout=240,
                    )
                    return r.choices[0].message.content or ""
                except Exception as e:  # noqa: BLE001 (incl. asyncio.TimeoutError -> drop this hint)
                    logger.warning(f"[hero] gemini reflector call failed/timeout: {e}")
                    return ""

        return await asyncio.gather(*[_one(p) for p in user_prompts])

    def _dump_reflections(self, responses, hints, seq_rewards, transcripts=None, prompts=None) -> None:
        """Append one JSONL line per trajectory to {export_path}/reflections/step_{step}.jsonl. Each line
        carries the WHOLE rollout so a reader can see what the reflector saw: the task context, the full
        decoded transcript (rollout), the raw reflector text, the parsed per-turn hints, and which turns
        pass is_problematic. Gated by hero_dump_reflections; CPU-only, best-effort (never fails the step)."""
        import json as _json
        from reverie.offline.reflection import is_problematic
        from reverie.offline.hero_perturn import _prompt_to_task_context

        step = int(getattr(self, "global_step", 0) or 0)
        out_dir = os.path.join(self.cfg.trainer.export_path, "reflections")
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"step_{step:04d}.jsonl"), "w") as f:
                for i, (raw, hint) in enumerate(zip(responses, hints)):
                    rew = float(seq_rewards[i]) if i < len(seq_rewards) else None
                    probs = [k for k, h in (hint or {}).items() if is_problematic(h)]
                    rec = {"step": step, "idx": i, "reward": rew}
                    if prompts is not None and i < len(prompts):
                        try:
                            rec["task"] = _prompt_to_task_context(prompts[i])
                        except Exception:  # noqa: BLE001
                            rec["task"] = None
                    if transcripts is not None and i < len(transcripts):
                        rec["transcript"] = transcripts[i]
                    rec.update({"raw": raw, "hints": hint, "problematic_turns": probs})
                    f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[hero] reflection dump failed: {e}")

    def _build_hero_training_input(self, generator_output, prompts, env_extras=None) -> Optional[TrainingInputBatch]:
        """HERO: run the modular teacher-data pipeline (reverie.hero) for this step and return a fresh
        :class:`TrainingInputBatch` — one row per placement unit (a problematic turn for ``per_turn``, a
        whole failed trajectory for ``top_level``), each scoring ``y`` under the real student context vs
        the feedback-reframed teacher context. The SDPO loss runs unchanged. Returns ``None`` when the
        pipeline yields no rows (caller skips the update).

        AXES: ``sdpo.feedback_type`` (reflection|review_raw|gold) x ``sdpo.placement`` (per_turn|top_level)
        x G. Reviewer hints (``self._hero_hints``) were produced upstream while the engine was awake;
        ``gold`` pulls its pre-rendered target from ``env_extras[i].reward_spec.ground_truth``."""
        from reverie.hero.pipeline import build_teacher_batch

        response_ids = generator_output["response_ids"]
        n = len(response_ids)
        if len(prompts) != n:
            logger.warning(f"[hero] prompts ({len(prompts)}) != responses ({n}); skipping HERO SD.")
            return None
        hints = getattr(self, "_hero_hints", None) or [{} for _ in range(n)]
        raw_rewards = generator_output.get("rewards") or [0.0] * n
        seq_rewards = [
            float(sum(r)) if isinstance(r, (list, tuple)) else float(r) for r in raw_rewards[:n]
        ]
        sdpo_cfg = self.cfg.trainer.algorithm.sdpo

        gold_texts = None
        if getattr(sdpo_cfg, "feedback_type", None) == "gold":
            if env_extras is not None and len(env_extras) >= n:
                gold_texts = [
                    (e.get("reward_spec", {}) or {}).get("ground_truth") if isinstance(e, dict) else None
                    for e in env_extras[:n]
                ]
            else:
                logger.warning(
                    "[hero] feedback_type=gold but env_extras unavailable/misaligned "
                    f"(have {0 if env_extras is None else len(env_extras)}, need {n}); no gold this step."
                )

        ex = build_teacher_batch(
            self.tokenizer,
            feedback_type=sdpo_cfg.feedback_type,
            placement=sdpo_cfg.placement,
            prompt_messages=prompts,
            prompt_token_ids=generator_output["prompt_token_ids"],
            response_ids=response_ids,
            response_texts=[self.tokenizer.decode(r, skip_special_tokens=False) for r in response_ids],
            loss_masks=generator_output["loss_masks"],
            hints=hints[:n],
            seq_rewards=seq_rewards,
            gold_texts=gold_texts,
            success_threshold=sdpo_cfg.success_reward_threshold,
            scrub_args=sdpo_cfg.hero_scrub_args,
            max_reprompt_len=sdpo_cfg.max_reprompt_len,
            apply_chat_template_kwargs=dict(getattr(self.cfg.generator, "chat_template_kwargs", {}) or {}),
        )
        for k, v in ex["stats"].items():
            self.all_metrics[k] = v

        m = len(ex["y_ids"])
        if m == 0:
            logger.warning("[hero] no problematic turns this step; skipping update.")
            return None

        # Student batch: build tensors from per-turn (prompt = real H_t, response = y_t). We do NOT
        # call convert_to_training_input because its boundary computation asserts num_prompts ==
        # train_batch_size — untrue for the M per-turn rows. Each row is its own (stepwise) prompt,
        # n_samples=1; rewards are dummy (pure SD ignores advantages), loss is over all y_t tokens.
        uids = ex["uids"]
        resp = ex["y_ids"]
        dummy_rewards = [[0.0] * len(y) for y in resp]
        # per_turn: y is a pure agent span -> all-ones. top_level: y is the FULL trajectory, so the
        # placement supplies the ORIGINAL per-token loss mask (agent tokens only).
        loss_masks = ex.get("loss_masks") or [[1] * len(y) for y in resp]
        (seq_t, attn_t, resp_mask_t, rew_t, lm_t, rollout_lp_t, _) = convert_prompts_responses_to_batch_tensors(
            self.tokenizer, ex["student_prefix"], resp, dummy_rewards, loss_masks, None, None,
            max_seq_len=self.cfg.trainer.algorithm.max_seq_len,
        )
        ti = TrainingInputBatch({
            "sequences": seq_t, "attention_mask": attn_t, "response_mask": resp_mask_t,
            "rewards": rew_t, "loss_mask": lm_t, "rollout_logprobs": rollout_lp_t,
            "rollout_expert_indices": None, "pixel_values": None, "image_grid_thw": None,
        })
        # Treat the M rows as M stepwise prompts in a single mini-batch (reuse the boundary helper with
        # train_batch_size=mini_batch_size=M, is_stepwise=True → [(0, M)]).
        ti.metadata = {"uids": uids}
        ti.metadata["policy_mini_batch_boundaries"] = compute_prompt_mini_batch_boundaries(
            uids, m, m, True, 1
        )
        ti.metadata["policy_prompt_boundaries"] = compute_prompt_boundaries(uids)
        ti.metadata["response_length"] = resp_mask_t.shape[1]
        ti.metadata["avg_response_length"] = sum(len(y) for y in resp) / m

        # Pad the student batch to an even multiple of dp_size (zeros loss_mask on padded rows).
        dp_size = self.dispatch.get_lcm_dp_size()
        pad_size = math.ceil(ti.batch_size / dp_size) * dp_size - ti.batch_size
        ti = pad_training_input_batch(ti, pad_size)

        # Teacher rows: reframed H̃_t + the SAME y_t ids, row-aligned + padded like the standard SDPO
        # path (padded rows get self_distillation_mask=0 so they never contribute to the loss).
        teacher_sequences, teacher_attention_mask = _left_pad_concat(
            ex["teacher_prefix"], resp, self.tokenizer.pad_token_id
        )
        if pad_size > 0:
            teacher_sequences = torch.cat([teacher_sequences, teacher_sequences[:1].repeat(pad_size, 1)], dim=0)
            teacher_attention_mask = torch.cat(
                [teacher_attention_mask, teacher_attention_mask[:1].repeat(pad_size, 1)], dim=0
            )
        ti["teacher_sequences"] = teacher_sequences
        ti["teacher_attention_mask"] = teacher_attention_mask
        ti["self_distillation_mask"] = torch.cat(
            [torch.ones(m, 1, dtype=torch.float32), torch.zeros(pad_size, 1, dtype=torch.float32)], dim=0
        )
        return ti

    @torch.no_grad()
    def _compute_sdpo_loss_scale(
        self,
        data: TrainingInputBatch,
        mini_batch_boundaries: List[Tuple[int, int]],
    ) -> TrainingInputBatch:
        """Per-row loss-reduction pre-scale for SDPO (the counterpart of ``_normalize_advantages``).

        SDPO has no advantage tensor to carry SkyRL's token-mean normalization, so we precompute a
        per-row ``1/N`` factor that ``compute_sdpo_loss`` multiplies in. ``N`` is the count of
        *distilled* tokens (``loss_mask * self_distillation_mask``) in the mini-batch, so the effective
        learning rate stays stable as the fraction of groups with a successful sibling changes over
        training (rather than shrinking early when few groups succeed).
        """
        loss_mask = data["loss_mask"]
        self_distillation_mask = data["self_distillation_mask"]
        reduction = self.cfg.trainer.algorithm.loss_reduction
        if reduction != "token_mean":
            logger.warning(
                f"[sdpo] loss_reduction={reduction!r} is not specialized for SDPO; using token_mean scaling."
            )
        scale = torch.zeros((loss_mask.shape[0], 1), dtype=torch.float32, device=loss_mask.device)
        for start_idx, end_idx in mini_batch_boundaries:
            distilled = loss_mask[start_idx:end_idx] * self_distillation_mask[start_idx:end_idx]
            denom = distilled.sum().clamp(min=1.0)
            scale[start_idx:end_idx] = 1.0 / denom
        data["sdpo_loss_scale"] = scale
        return data

    @torch.no_grad()
    def _normalize_advantages(
        self,
        data: TrainingInputBatch,
        mini_batch_boundaries: List[Tuple[int, int]],
        prompt_boundaries: Optional[List[Tuple[int, int]]] = None,
    ) -> TrainingInputBatch:
        advantages = data["advantages"]
        response_mask = data["response_mask"]

        # Step 1: Z-score normalization (if enabled)
        if self.cfg.trainer.algorithm.advantage_batch_normalize:
            num_actions = response_mask.sum()
            mean = advantages.mean()
            std = ((advantages - mean).pow(2) * response_mask).sum()
            rstd = (std / num_actions).clamp(min=1e-8).rsqrt()
            data["advantages"] = (advantages - mean) * rstd

        # Step 2: Loss reduction normalization per mini-batch
        normalized_advantages = torch.zeros_like(advantages)
        for start_idx, end_idx in mini_batch_boundaries:
            mini_batch = data[start_idx:end_idx]
            # For prompt_mean, select the prompt boundaries falling within this mini-batch
            # and rebase them to mini-batch-relative indices.
            mb_prompt_boundaries = None
            if prompt_boundaries is not None:
                mb_prompt_boundaries = [
                    (p_start - start_idx, p_end - start_idx)
                    for p_start, p_end in prompt_boundaries
                    if start_idx <= p_start < end_idx
                ]
            normalized_advantages[start_idx:end_idx] = apply_loss_reduction_to_advantages_minibatch(
                advantages=mini_batch["advantages"],
                loss_mask=mini_batch["loss_mask"],
                loss_reduction=self.cfg.trainer.algorithm.loss_reduction,
                micro_batch_size=self.cfg.trainer.micro_train_batch_size_per_gpu,
                max_seq_len=self.cfg.trainer.algorithm.max_seq_len,
                prompt_boundaries=mb_prompt_boundaries,
            )

        data["advantages"] = normalized_advantages
        return data

    def _execute_training_step(self, model: str, data: TrainingInputBatch) -> Dict[str, float]:
        """
        Execute training step using forward_backward + optim_step.

        The trainer loops over epochs and mini-batches. Workers handle micro-batching
        internally for gradient accumulation (memory efficiency).

        All per-DP mini-batch chunks are pre-staged in the Ray object store before
        the training loop so serialization stays off the GPU critical path.

        Args:
            model: Model name ("policy" or "critic")
            data: Training data batch

        Returns:
            Dict of reduced metrics from training
        """
        boundaries = data.metadata[f"{model}_mini_batch_boundaries"]

        if model == "policy":
            # Normalize advantages for policy training; critic training does not need this
            prompt_boundaries = data.metadata.get("policy_prompt_boundaries")
            data = self._normalize_advantages(data, boundaries, prompt_boundaries)
            # SDPO carries its token-mean normalization in a separate per-row scale (no advantages).
            if self.cfg.trainer.algorithm.policy_loss_type == "sdpo":
                data = self._compute_sdpo_loss_scale(data, boundaries)

        all_metrics: Dict[str, List[float]] = defaultdict(list)

        # Pre-stage all per-DP mini-batch chunks in the object store so that
        # serialization is fully off the critical path during training.
        all_chunk_refs = self.dispatch.stage_data(model, data, boundaries)

        # Training loop over epochs and mini-batches
        for _epoch in range(self.cfg.trainer.update_epochs_per_batch):
            for chunk_refs in all_chunk_refs:
                status = self.dispatch.forward_backward_from_staged(model, chunk_refs)
                for k, v in status.metrics.items():
                    all_metrics[k].append(v)

                # Optimizer step after each mini batch
                grad_norm = self.dispatch.optim_step(model)
                if grad_norm is not None:
                    all_metrics["grad_norm"].append(grad_norm)

        # Reduce metrics across all mini-batches and epochs
        reduced_metrics = reduce_metrics(all_metrics, sum_loss_metrics=False)
        return reduced_metrics

    def train_critic_and_policy(self, data: TrainingInputBatch):
        """
        Run the training step for the policy and critic models.

        Uses forward_backward + optim_step for both FSDP and Megatron strategies.
        """
        data.metadata["global_step"] = self.global_step
        critic_status = None

        # Unified training interface for both FSDP and Megatron
        if self.has_critic:
            with Timer("critic_train", self.all_timings):
                critic_status = self._execute_training_step("critic", data)
        with Timer("policy_train", self.all_timings):
            policy_status = self._execute_training_step("policy", data)

        # Update metrics
        if critic_status is not None:
            for k, v in critic_status.items():
                self.all_metrics.update({f"critic/{k}": v})

        for k, v in policy_status.items():
            self.all_metrics.update({f"policy/{k}": v})

        self.dispatch.empty_cache()

        return policy_status

    def handle_dynamic_sampling(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str], bool]:
        """
        Handle dynamic sampling for the current batch.

        Accumulates the generator output and UIDs across batches if we are sampling repeatedly
        and applies the dynamic sampling strategy (i.e. filter, replace) to the current batch.
        If we hit the limit of max sample batches, we raise an error.

        Args:
            generator_output: Current batch generator output
            uids: Current batch UIDs

        Returns:
            processed_output: Filtered generator output
            processed_uids: Filtered UIDs
            keep_sampling: Whether to keep sampling
        """
        # Prepare sampling configuration
        max_sample_batches = self.cfg.trainer.algorithm.dynamic_sampling.max_sample_batches
        dynamic_sampling_config = {
            "type": self.cfg.trainer.algorithm.dynamic_sampling.type,
            "max_sample_batches": max_sample_batches,
            "min_replace_ratio": self.cfg.trainer.algorithm.dynamic_sampling.min_replace_ratio,
            "train_batch_size": self.cfg.trainer.train_batch_size,
            "n_samples_per_prompt": self.cfg.generator.n_samples_per_prompt,
        }

        if self.dynamic_sampling_state is None:
            self.dynamic_sampling_state: DynamicSamplingState = {
                "sample_batch_count": 1,
            }
        else:
            self.dynamic_sampling_state["sample_batch_count"] += 1

        # Handle dynamic sampling using utilities
        processed_output, processed_uids, keep_sampling, updated_state = trainer_utils.handle_dynamic_sampling(
            generator_output, uids, dynamic_sampling_config, self.dynamic_sampling_state
        )

        # Check max resample limit, and if we hit it, raise an error
        if (
            keep_sampling
            and max_sample_batches > 0
            and self.dynamic_sampling_state["sample_batch_count"] >= max_sample_batches
        ):
            raise RuntimeError(
                f"Exiting training loop due to hitting dynamic sampling limit for "
                f"{self.cfg.trainer.algorithm.dynamic_sampling.type} strategy with "
                f"{self.cfg.trainer.algorithm.dynamic_sampling.max_sample_batches} max sample batches. "
                f"Please check your data difficulty distribution."
            )
        # Update state
        self.dynamic_sampling_state = updated_state

        if not keep_sampling:
            # Reset state when sampling is complete
            self.dynamic_sampling_state = None

        return processed_output, processed_uids, keep_sampling

    def _get_dp_group_models(self, rank: int, model_type: str = ""):
        model = getattr(self, model_type)
        return model._actor_handlers[rank]

    def _get_mesh_rank(self, rank: int, model_type: str = "") -> MeshRank:
        model: PPORayActorGroup = getattr(self, model_type)
        actor_info: ActorInfo = model.actor_infos[rank]
        return actor_info.rank

    def save_checkpoints(self) -> str:
        """
        Save the model, optimizer, and training states to disk. Returns the
        checkpoint folder path.

        Dispatch handles offload/backload automatically for all colocation configurations.
        """
        # Create global step folder structure
        global_step_folder = os.path.join(self.cfg.trainer.ckpt_path, f"global_step_{self.global_step}")
        policy_save_dir = os.path.join(global_step_folder, "policy")
        critic_save_dir = os.path.join(global_step_folder, "critic")

        io.makedirs(global_step_folder, exist_ok=True)

        # Save policy checkpoint (dispatch handles offload/backload)
        self.dispatch.save_checkpoint("policy", policy_save_dir, self.tokenizer)

        # Save critic checkpoint (if it exists)
        if self.has_critic:
            self.dispatch.save_checkpoint("critic", critic_save_dir, self.tokenizer)

        # Save dataloader state
        dataloader_save_path = os.path.join(global_step_folder, "data.pt")
        try:
            dataloader_state_dict = self.train_dataloader.state_dict()
            with io.open_file(dataloader_save_path, "wb") as f:
                torch.save(dataloader_state_dict, f)
            logger.info(f"Saved dataloader state to {dataloader_save_path}")
        except Exception as e:
            logger.warning(f"Failed to save dataloader state: {e}")

        # Save additional trainer state
        trainer_state = {
            "global_step": self.global_step,
            "config": asdict(self.cfg),
        }
        trainer_state_path = os.path.join(global_step_folder, "trainer_state.pt")
        with io.open_file(trainer_state_path, "wb") as f:
            torch.save(trainer_state, f)
        logger.info(f"Saved trainer state to {trainer_state_path}")

        # Atomic tracking - write this last after all saves succeed
        latest_checkpoint_file = os.path.join(self.cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt")
        with io.open_file(latest_checkpoint_file, "w") as f:
            f.write(str(self.global_step))

        logger.info(f"Successfully saved checkpoint for global_step_{self.global_step} to: {global_step_folder}")

        # Clean up old checkpoints after successful save
        with Timer("cleanup_old_checkpoints", self.all_timings):
            self._cleanup_old_checkpoints()

        return global_step_folder

    def _cleanup_old_checkpoints(self):
        if not self._node_ids:
            self._node_ids = self.dispatch.get_node_ids()
        run_on_each_node(
            self._node_ids,
            cleanup_old_checkpoints,
            self.cfg.trainer.ckpt_path,
            self.cfg.trainer.max_ckpts_to_keep,
        )
        # run on driver as well
        # NOTE (sumanthrh): the function will get called twice on the node with driver process, but it's ok because it's idempotent
        cleanup_old_checkpoints(self.cfg.trainer.ckpt_path, self.cfg.trainer.max_ckpts_to_keep)

    def load_checkpoints(self) -> Tuple[int, str]:
        """
        Load complete checkpoint state and return the global_step to resume from.
        Returns 0 if no checkpoint is loaded.

        If colocate_all is True, assumes that the policy model is currently on GPU.

        Returns:
            global_step: The global step to resume from.
            checkpoint_path: The path to the checkpoint.
        """
        checkpoint_path = None
        # Check if resumption is enabled
        if self.resume_mode == ResumeMode.NONE:
            logger.info("Checkpoint resumption disabled, starting training from scratch")
            return 0, None
        # first, let's get resume_path
        elif self.resume_mode == ResumeMode.LATEST:
            latest_checkpoint_file = os.path.join(self.cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt")
            if not io.exists(latest_checkpoint_file):
                logger.info("No checkpoint found, starting training from scratch")
                return 0, None
            with io.open_file(latest_checkpoint_file, "r") as f:
                ckpt_iteration = int(f.read().strip())
            checkpoint_path = os.path.join(self.cfg.trainer.ckpt_path, f"{GLOBAL_STEP_PREFIX}{ckpt_iteration}")
            # Run validation: Make sure ckpt folder is consistent with latest_ckpt_global_step.txt
            validate_consistency_for_latest_checkpoint(
                self.cfg.trainer.ckpt_path,
                ckpt_iteration,
                checkpoint_path,
                latest_checkpoint_file,
                self.cfg.trainer.ckpt_interval,
            )
        else:
            # Get and validate resume path
            checkpoint_path = Path(self.cfg.trainer.resume_path)
            if not checkpoint_path:
                raise ValueError("`trainer.resume_path` must be specified when resume_mode is 'from_path'")

            # Validate that it's a global_step directory
            if GLOBAL_STEP_PREFIX not in checkpoint_path.name:
                raise ValueError(
                    f"`trainer.resume_path` must point to a directory whose name starting with {GLOBAL_STEP_PREFIX}, got: {checkpoint_path}"
                )

        # Validate that the path exists
        if not io.exists(str(checkpoint_path)):
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        # Extract global step from checkpoint path
        global_step = extract_step_from_path(Path(checkpoint_path))
        if global_step == -1:
            raise ValueError(f"Checkpoint path {checkpoint_path} is not a valid checkpoint path")
        logger.info(f"Resuming from global_step: {global_step}")

        # Define paths for different checkpoint components
        policy_ckpt_dir = os.path.join(checkpoint_path, "policy")
        critic_ckpt_dir = os.path.join(checkpoint_path, "critic")
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        dataloader_state_path = os.path.join(checkpoint_path, "data.pt")

        # Validate that required checkpoint files exist
        if not io.exists(trainer_state_path):
            raise FileNotFoundError(f"Trainer state file not found: {trainer_state_path}")

        # 1. Load and validate trainer state
        with io.open_file(trainer_state_path, "rb") as f:
            trainer_state = torch.load(f, map_location="cpu", weights_only=False)
        saved_global_step = trainer_state.get("global_step", global_step)
        logger.info("Successfully loaded trainer state")
        if saved_global_step != global_step:
            logger.warning(f"Global step mismatch: path={global_step}, saved={saved_global_step}. Using path value.")

        # 2. Load dataloader state if available
        if io.exists(dataloader_state_path):
            try:
                with io.open_file(dataloader_state_path, "rb") as f:
                    dataloader_state = torch.load(f, map_location="cpu", weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state)
                logger.info("Successfully loaded dataloader state")
            except Exception as e:
                logger.warning(f"Failed to load dataloader state: {e}. Dataloader will start from beginning.")
        else:
            logger.warning(
                f"No dataloader state found at {dataloader_state_path}. Dataloader will start from beginning."
            )

        # 3. Load policy checkpoint (dispatch handles offload/backload)
        logger.info(f"Loading policy checkpoint from {policy_ckpt_dir}")
        self.dispatch.load_checkpoint(
            "policy",
            policy_ckpt_dir,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        logger.info("Successfully loaded policy checkpoint")

        # 4. Load critic checkpoint if it exists and we have a critic model
        if self.has_critic:
            logger.info(f"Loading critic checkpoint from {critic_ckpt_dir}")
            self.dispatch.load_checkpoint(
                "critic",
                critic_ckpt_dir,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
            logger.info("Successfully loaded critic checkpoint")

        logger.info(f"Successfully loaded complete checkpoint state from global_step_{global_step}")
        return global_step, str(checkpoint_path)

    def save_models(self):
        """
        Save the model parameters in HF format at `cfg.trainer.export_path`.

        Dispatch handles offload/backload automatically for all colocation configurations.
        """
        policy_export_dir = os.path.join(self.cfg.trainer.export_path, f"global_step_{self.global_step}", "policy")
        self.dispatch.save_hf_model("policy", policy_export_dir, self.tokenizer)

        if self.has_critic:
            critic_export_dir = os.path.join(self.cfg.trainer.export_path, f"global_step_{self.global_step}", "critic")
            self.dispatch.save_hf_model("critic", critic_export_dir, self.tokenizer)

        logger.info("Successfully saved model weights.")

    def update_ref_with_policy(self):
        """
        Update the reference model with the policy model weights (required by some algorithms).

        Dispatch handles offload/backload automatically for all colocation configurations.
        After this method, save_weights_for_sampler() should be called to sync weights.
        """
        # TODO(tgriggs): Make policy-to-ref sync faster.
        policy_export_dir = os.path.join(self.cfg.trainer.export_path, f"global_step_{self.global_step}", "policy")

        # Save policy model (dispatch handles GPU state)
        self.dispatch.save_hf_model("policy", policy_export_dir, self.tokenizer)

        # Re-initialize ref model from saved policy (dispatch handles offloading policy first)
        self.dispatch.init_model("ref", policy_export_dir)

        # Clean up temporary saved model files
        try:
            shutil.rmtree(policy_export_dir)
            logger.info(f"Cleaned up temporary policy export directory: {policy_export_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary policy export directory {policy_export_dir}: {e}")

        logger.info("Successfully updated ref model with policy model, training continues.")
