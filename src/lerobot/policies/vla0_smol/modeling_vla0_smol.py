#!/usr/bin/env python

import logging
import random
from collections import deque
from typing import Optional
import copy
import torch
import xgrammar as xgr
from torch import Tensor, nn
from torch.profiler import record_function
from torchvision.transforms import CenterCrop, RandomCrop
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.smolvlm.image_processing_smolvlm_fast import SmolVLMImageProcessorFast

from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vla0_smol.configuration_vla0_smol import VLA0SmolConfig
from lerobot.policies.vla0_smol.monkey_patch import patch_SmolVLM_amp, patch_SmolVLMProcessor
from lerobot.utils.constants import ACTION, OBS_STATE

PRECISION = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

EPS = 1e-6


class VLA0SmolPolicy(PreTrainedPolicy):
    """Wrapper class around VLA0 model to train and run inference within LeRobot."""

    config_class = VLA0SmolConfig
    name = "vla0"

    def __init__(
        self,
        config: VLA0SmolConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = VLA0(config)

        self.use_ensembling = self.config.ensemble_size > 1
        if self.use_ensembling:
            self.temporal_ensembler = VLA0TemporalEnsembler(
                ensemble_prediction_count=self.config.ensemble_size
            )
            logging.info("Ensemble mode for token prediction is enabled.")
            assert config.n_action_steps == 0, (
                "When ensemble mode is enabled, n_action_steps param should be zero."
            )
        else:
            self.temporal_ensembler = None
            logging.info("N actions step mode for token prediction is enabled.")
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

        if self.use_ensembling:
            self.temporal_ensembler.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("Currently not implemented for VLA0")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        if self.use_ensembling:
            actions = self.model.generate_actions(batch)

            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            return self.temporal_ensembler.update(actions)
        elif self.config.action_streaming:
            next_action = self.model.generate_one_action(batch).squeeze(1)
            return next_action 
        else:
            # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
            # querying the policy.
            if len(self._action_queue) == 0:
                actions = self.model.generate_actions(batch)
                actions = actions[:, :self.config.n_action_steps] # torch.Size([batch_size, self.config.n_action_steps, action_dim])
                # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
                # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
                self._action_queue.extend(actions.transpose(0, 1))
            next_action = self._action_queue.popleft() # torch.Size([batch_size, action_dim])
            return next_action

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        loss_dict = self.model.forward(batch)
        loss = loss_dict.pop("loss")
        return loss, loss_dict


def build_exact_n_numbers_grammar(n_numbers: int) -> str:
    """
    Constructs an EBNF grammar that enforces exactly `n_numbers` integers.
    """
    # integer ::= "-"? [0-9]+
    base_rules = """
    integer ::= "-"? [0-9]+
    space ::= " "
    """

    # Build the exact sequence string: integer space integer space integer ...
    # We construct "integer " * (N-1) + "integer"
    sequence_parts = ["integer"] * n_numbers
    sequence_rule = "root ::= space " + " space ".join(sequence_parts)

    return base_rules + sequence_rule


class VLA0TemporalEnsembler:
    def __init__(self, ensemble_prediction_count: int) -> None:
        """
        Implements the specific ensembling logic used in VLA0 Libero evaluation.

        Args:
            ensemble_prediction_count (int): Corresponds to ensemble_prediction param.
                This limits how many overlapping schedules are averaged.
        """
        self.max_schedules = ensemble_prediction_count
        self.reset()

    def reset(self):
        self.schedules = deque(maxlen=self.max_schedules)

    def update(self, new_action_chunk: Tensor) -> Tensor:
        """
        Args:
            new_action_chunk: Tensor of shape (batch, horizon, action_dim).
                Note: This implementation assumes batch_size=1 for simplicity
                as per standard eval loops, but can be adapted.
        """
        self.schedules.append(new_action_chunk)

        current_actions = []
        for i, schedule in enumerate(reversed(self.schedules)):
            # schedule shape: (Batch, Horizon, Action_Dim)
            horizon_len = schedule.shape[1]

            if i < horizon_len:
                action_at_step_i = schedule[:, i, :]
                current_actions.append(action_at_step_i)
            else:
                break

        if not current_actions:
            return new_action_chunk[:, 0, :]

        stacked_actions = torch.stack(current_actions, dim=0)
        action_to_execute = stacked_actions.mean(dim=0)

        return action_to_execute

@torch.no_grad()
def shift_right(tensor):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


class EagleModel(nn.Module):
    def __init__(self, 
                 num_heads: int,
                 config: LlamaConfig,
                 input_embedding: nn.Module,
                 output_embedding: nn.Module):
        super().__init__()
        self.num_heads = num_heads
        self.cfg = copy.deepcopy(config)
        self.cfg.num_hidden_layers=1

        self.embed_tokens = input_embedding
        self.lm_head = output_embedding

        self.norm = LlamaRMSNorm(self.cfg.hidden_size,
                                 eps=self.cfg.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.cfg)

        self.fuse_hidden_and_embed = nn.Linear(2 * self.cfg.hidden_size,
                                 self.cfg.hidden_size,
                                 bias=False)

        self.fuse_3_hidden = nn.Linear(3 * self.cfg.hidden_size,
                                               self.cfg.hidden_size)

        self.decoder_layer = LlamaDecoderLayer(self.cfg,
                                               layer_idx=0)

    def fuse_base_model_hidden_states(self, hidden_states: list):
        """Fuse a list of 3 (batch_size, seq_len, hidden_size) tensors along last dimension"""
        if len(hidden_states) != 3:
            raise ValueError(f"Expected 3 hidden-state tensors, got {len(hidden_states)}")
        
        hidden_states = torch.cat(hidden_states, dim = -1)
        return self.fuse_3_hidden(hidden_states)

    def forward(self,
                input_ids: torch.LongTensor,
                hidden_states: torch.FloatTensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                ):
        inputs_embeds = self.embed_tokens(input_ids) # [B, seq_len, hidden_size]
        inputs_embeds = inputs_embeds.to(hidden_states.device)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.cfg)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.cfg,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states = torch.cat((hidden_states, inputs_embeds), dim = -1)
        hidden_states = self.fuse_hidden_and_embed(hidden_states)

        hidden_states = self.decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
    
    def calculate_mtp_loss(self,
                           input_ids: torch.LongTensor,
                           hidden_states: torch.FloatTensor,
                           loss_mask: torch.Tensor,
                           ):
        """
        input_ids: [B, seq_len]
        hidden_state: [B, seq_len, hidden_size]
        loss_mask: [B, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = hidden_states.device

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        past_key_values = DynamicCache(config=self.cfg)

        losses = []
        for head_idx in range(0, self.num_heads):

            inputs_embeds = self.embed_tokens(input_ids) # [B, seq_len, hidden_size]
            hidden_states = torch.cat((hidden_states, inputs_embeds), dim = -1) # [B, seq_len, 2*hidden_size]
            hidden_states = self.fuse_hidden_and_embed(hidden_states) # [B, seq_len, hidden_size]

            if head_idx == 0:
                block_attention_shape = (batch_size, 1, seq_len, seq_len)
                causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
                attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(*block_attention_shape)
            else:
                next_attention_block = torch.full(block_attention_shape, False, device=device, dtype=torch.bool)
                diag = torch.arange(seq_len, device=device)
                next_attention_block[:, :, diag, diag] = True
                attention_mask = torch.cat([attention_mask, next_attention_block], dim = -1)

            position_ids = torch.arange(head_idx, input_ids.shape[1] + head_idx, device=device).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            hidden_states_out = self.decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = self.lm_head(self.norm(hidden_states_out))[:,:-1,:] # [B, seq_len - 1, vocab_size]
            targets = input_ids[:, 1:].to(device) # [B, seq_len - 1]

            head_loss = loss_fct(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            
            loss_mask_eagle = loss_mask[:, 1:].to(device) # [B, seq_len - 1]

            head_loss = head_loss * loss_mask_eagle.reshape(-1)
            head_loss = head_loss.sum() / torch.clamp(loss_mask_eagle.sum(), min=1)

            losses.append(head_loss)

            input_ids = shift_right(input_ids)
            loss_mask = shift_right(loss_mask)

            hidden_states = hidden_states_out

        return losses
    
    def generate_next_token(self,
                            input_ids: torch.LongTensor,
                            hidden_states: torch.FloatTensor,
                            attention_mask: Optional[torch.Tensor] = None,
                            position_ids: Optional[torch.LongTensor] = None,
                            past_key_values: Optional[Cache] = None,
                            cache_position: Optional[torch.LongTensor] = None,
                            use_cache: Optional[bool] = None,
                ):
        with torch.inference_mode():
            output = self.forward(input_ids,
                                hidden_states,
                                attention_mask,
                                position_ids,
                                past_key_values,
                                cache_position,
                                use_cache)
            logits = self.lm_head(self.norm(output.last_hidden_state)) # [batch_size, seq_len - 1, vocab_size]
        generated_token = logits[:, -1, :].argmax(-1, keepdim=True)

        return generated_token, output

class VLA0(nn.Module):
    def __init__(self, config: VLA0SmolConfig):
        super().__init__()
        self.config = config

        self.precision = PRECISION.get(config.precision, torch.float32)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            self.config.vlm_checkpoint, dtype=self.precision
        )

        # Patch SmolVLMProcessor to enable using SmolVLMImageProcessorFast
        patch_SmolVLMProcessor()

        # Patch SmolVLM to enable AMP training
        patch_SmolVLM_amp(False)

        image_processor = SmolVLMImageProcessorFast.from_pretrained(
            self.config.vlm_checkpoint,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.config.vlm_checkpoint,
            image_processor=image_processor,
            use_fast=True,
        )

        self.action_horizon = self.config.chunk_size
        self.action_dim = self.config.action_feature.shape[0]

        if config.freeze_vision_encoder:
            for param in self.vlm.model.vision_model.parameters():
                param.requires_grad = False

        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.eos_token_id = self.processor.tokenizer.eos_token_id

        self.image_keys = self.config.image_features.keys()

        self.do_crop = config.crop_shape is not None
        if self.do_crop:
            self.random_crop_fn = RandomCrop(config.crop_shape)
            self.center_crop_fn = CenterCrop(config.crop_shape)

        self.actions_mask_symbol = "<MASK_ACT>"
        assert self.actions_mask_symbol not in self.processor.tokenizer.get_vocab(), (
            f"Replace {self.actions_mask_symbol} token with a different token."
        )
        self.processor.tokenizer.add_tokens([self.actions_mask_symbol], special_tokens=True)
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer), mean_resizing=False)
        self.mask_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.actions_mask_symbol)

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.processor.tokenizer)
        self.grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        total_actions = self.config.chunk_size * self.config.action_feature.shape[0]
        ebnf_string = build_exact_n_numbers_grammar(total_actions)
        self.compiled_grammar = self.grammar_compiler.compile_grammar(ebnf_string)

        # stream generation
        self.new_obs = True

        # multi-token prediction
        self.train_mtp = True if config.num_train_eagle_heads > 0 else False
        self.inference_mtp = True if config.num_inference_eagle_heads > 0 else False

        if self.train_mtp:
            self.train_mtp = True
            self.eagle_model = EagleModel(num_heads=self.config.num_train_eagle_heads,
                                          config=self.vlm.model.text_model.config,
                                          input_embedding=self.vlm.get_input_embeddings(),
                                          output_embedding=self.vlm.get_output_embeddings())

    def apply_action_masking(self, actions: list[list[str]]):
        if not self.training:
            return actions

        if random.random() < self.config.action_mask_skip_per:
            return actions

        num_actions = len(actions)

        aug_per = random.uniform(0.0, self.config.action_mask_aug_per)
        num_actions_to_mask = int(num_actions * aug_per)

        if num_actions_to_mask > 0:
            indices = random.sample(range(num_actions), num_actions_to_mask)

            for idx in indices:
                actions[idx] = self.actions_mask_symbol

        return actions

    def create_prefix_tokens(
        self,
        states: torch.Tensor,
        images: torch.Tensor,
        lang_text: str,
        actions: torch.Tensor | None,
    ):
        device = states.device
        batch_size = states.shape[0]

        # Precompute bin edges on GPU
        bins = torch.linspace(-1.0 - EPS, 1.0 + EPS, self.config.n_state_bins + 1, device=device)[:-1]

        # Discretize directly on GPU
        discretized_states = torch.bucketize(states, bins) - 1  # shape: [B, state_dim]

        # Move the batched results to CPU only once for string formatting
        disc_states_cpu = discretized_states.detach().cpu().numpy()

        if actions is None:
            disc_actions_cpu = [""] * batch_size
        else:
            if self.config.relative_actions:
                actions = actions - states.unsqueeze(1)
            discretized_actions = torch.bucketize(actions, bins) - 1  # shape: [B, state_dim]
            disc_actions_cpu = discretized_actions.detach().cpu().numpy()

        # Build strings in batch
        prompts = []
        for txt, disc_st, act in zip(lang_text, disc_states_cpu, disc_actions_cpu, strict=False):
            task_cleaned = txt.lower().strip().replace("_", " ")
            state_str = " ".join(map(str, disc_st.tolist()))

            if self.config.use_state:
                prefix = f"Task: {task_cleaned}, State: {state_str}, Actions: "
            else:
                prefix = f"Task: {task_cleaned}, Actions: "

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image"} for _ in range(len(images))],
                        {
                            "type": "text",
                            "text": prefix,
                        },
                    ],
                }
            ]

            if actions is not None:
                action_list = list(map(str, act.flatten().tolist()))
                action_list = self.apply_action_masking(action_list)
                action_str = " ".join(action_list)
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{action_str}",
                            },
                        ],
                    }
                )
            prompts.append(
                self.processor.apply_chat_template(messages, add_generation_prompt=actions is None)
            )

        images = {
            camera_name: list(torch.unbind(camera_images, dim=0))
            for camera_name, camera_images in images.items()
        }

        images_reshaped = []
        for imgs in zip(*images.values(), strict=True):
            if self.do_crop:
                crop_fn = self.random_crop_fn if self.training else self.center_crop_fn
                images_reshaped.append([crop_fn(img) for img in imgs])
            else:
                images_reshaped.append(list(imgs))

        prefix_out = self.processor(
            images=images_reshaped,
            text=prompts,
            do_resize=self.config.do_image_splitting,
            do_rescale=False,
            return_tensors="pt",
            padding=True,
            padding_side="right" if actions is not None else "left",
        )
        return prefix_out

    def create_input_tokens(
        self,
        states: torch.Tensor,
        images: torch.Tensor,
        lang_text: str,
        actions: torch.Tensor | None = None,
    ):
        device = states.device

        prefix_out = self.create_prefix_tokens(
            states=states, images=images, lang_text=lang_text, actions=actions
        )
        prefix_out = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in prefix_out.items()}

        if actions is None:
            loss_mask = None
        else:
            split_mask = torch.where(prefix_out["input_ids"] == self.config.start_actions_token, 1, 0)
            loss_mask = torch.cumsum(split_mask, dim=-1).clamp(0, 1) & prefix_out["attention_mask"]
            is_masked_token = prefix_out["input_ids"] == self.mask_token_id
            loss_mask = loss_mask & (~is_masked_token)

        return prefix_out, loss_mask

    def prepare_images(self, batch: torch.Tensor):
        """Preprocess LeRobot batch into inputs"""
        images = {}
        present_img_keys = [key for key in self.image_keys if key in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        for key in self.image_keys:
            if key in present_img_keys:
                img = batch[key]

            images[key] = img
        return images

    def forward(self, batch: dict[str, Tensor]):
        device = batch[OBS_STATE].device

        with record_function("create_input_tokens"):
            images = self.prepare_images(batch)

            padded_outs, loss_mask = self.create_input_tokens(
                states=batch[OBS_STATE],
                images=images,
                lang_text=batch.get("task", ""),
                actions=batch[ACTION],
            )

        with record_function("forward"):
            outputs = self.vlm.forward(
                input_ids=padded_outs["input_ids"],
                attention_mask=padded_outs["attention_mask"],
                pixel_values=padded_outs["pixel_values"],
                pixel_attention_mask=padded_outs["pixel_attention_mask"],
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        with record_function("loss"):
            logits = outputs.logits
            logits = logits.to(torch.float32)

            loss_fct = nn.CrossEntropyLoss(reduction="none")

            # Shift left for next-step prediction
            logits = logits[:, :-1, :]
            targets = padded_outs["input_ids"][:, 1:].to(device)  # Shift targets
            loss_mask_vlm = loss_mask[:, 1:].to(device)  # Ensure correct shape

            # Compute per-token loss
            token_loss = loss_fct(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            # Apply loss mask
            token_loss = token_loss * loss_mask_vlm.reshape(-1)

            # Compute final loss
            vlm_loss = token_loss.sum() / torch.clamp(loss_mask_vlm.sum(), min=1)

        if not self.train_mtp:
            loss_dict = {
                "vlm_loss": vlm_loss.item(),
                "loss": vlm_loss,
                "sequence_len": padded_outs["input_ids"].shape[-1],
            }
            return loss_dict

        with record_function("mtp_loss"):

            base_hidden_states = [outputs.hidden_states[id] for id in self.config.eagle_layers_ids]
            fused_hidden_state = self.eagle_model.fuse_base_model_hidden_states(base_hidden_states)[:, :-1, :]

            mtp_losses = self.eagle_model.calculate_mtp_loss(input_ids=padded_outs["input_ids"][:, 1:],
                                                             hidden_states=fused_hidden_state,
                                                             loss_mask=loss_mask[:, 1:])
            mtp_loss = sum(mtp_losses)

            loss = vlm_loss + mtp_loss

            loss_dict = {
                "vlm_loss": vlm_loss.item(),
                "eagle_loss": mtp_loss.item(),
                "loss": loss,
                "sequence_len": padded_outs["input_ids"].shape[-1],
            }

        return loss_dict
    
    def generate_next_token(self,
                            input_ids,
                            past_key_values):
        with torch.inference_mode():
            out = self.vlm(input_ids=input_ids,
                           past_key_values=past_key_values,
                           output_hidden_states=True,
                           use_cache=True)
        generated_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
        return generated_token, out

    def check_end_of_generation(self, generated_token = None):
        if generated_token is not None:
            for i, token in enumerate(generated_token):
                if self.generation_finished[i]:
                    continue
                elif token == self.eos_token_id or token == self.pad_token_id:
                    self.generation_finished[i] = True

        if sum(self.generation_finished) == len(self.generation_finished):
            return True
        return False

    def reconstruct_actions(self, decoded_actions, batch):
        batch_size = batch[OBS_STATE].shape[0]
        device = batch[OBS_STATE].device
        # print(f"decoded actions: {decoded_actions}")
        discretized_actions = torch.stack(decoded_actions, dim=0).reshape(batch_size, -1, self.action_dim)

        # Assuming same bin setup
        bins = torch.linspace(-1.0 - EPS, 1.0 + EPS, self.config.n_state_bins + 1, device=device)

        # Compute bin centers (midpoints between edges)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])  # shape: [n_state_bins]

        # Map discretized indices back to continuous states
        reconstructed_actions = bin_centers[discretized_actions.clamp(0, self.config.n_state_bins - 1)]
        if self.config.relative_actions:
            reconstructed_actions += batch[OBS_STATE].unsqueeze(1)
        
        return reconstructed_actions

    def prefill(self, batch):
        images = self.prepare_images(batch)
        batch_size = batch[OBS_STATE].shape[0]

        padded_outs, _ = self.create_input_tokens(
            states=batch[OBS_STATE],
            images=images,
            lang_text=batch.get("task", ""),
            actions=None,
        )

        self.prefix_len = padded_outs["input_ids"].shape[1] - 1
        self.input_ids = torch.full(
            (batch_size, self.prefix_len + self.config.max_decoding_steps),
            self.pad_token_id,
            device=padded_outs["input_ids"].device,
            dtype=padded_outs["input_ids"].dtype,
        )
        self.input_ids[:, :self.prefix_len] = padded_outs["input_ids"][:,1:]

        with torch.inference_mode():
            out = self.vlm(**padded_outs,
                           use_cache=True,
                           output_hidden_states=True)

        generated_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
        return generated_token, out
        
    def initialise_new_generation(self, batch):
        batch_size = batch[OBS_STATE].shape[0]

        self.new_obs = False
        self.generation_batch = batch
        self.generation_finished = [False]*batch_size
        self.action_index = 0
        self.input_idx_base = 0
        self.input_idx_eagle = 0
        self.input_ids_len = 0

    def generate_one_action(self, batch):
        device = batch[OBS_STATE].device
        batch_size = batch[OBS_STATE].shape[0]

        next_action_is_generated = [False]*batch_size
        decoded_actions = [None]*batch_size

        if self.new_obs:
            self.initialise_new_generation(batch)

            generated_token, output = self.prefill(batch=batch)

            self.input_ids_len = self.prefix_len
            self.input_idx_base = self.prefix_len
            self.input_idx_eagle = 0

            self.input_ids[:, self.input_ids_len:self.input_ids_len + 1] = generated_token
            self.input_ids_len += 1

            self.past_key_values = output.past_key_values
            
            if self.inference_mtp:
                self.eagle_past_key_values = DynamicCache(config=self.eagle_model.cfg)
                base_hidden_states = [output.hidden_states[id] for id in self.config.eagle_layers_ids]
                self.hidden_state = self.eagle_model.fuse_base_model_hidden_states(base_hidden_states)

        # generate one action
        eagle_heads = self.config.num_inference_eagle_heads if self.inference_mtp else 0
        max_remained_steps = int((self.config.max_decoding_steps - (self.input_ids_len - self.prefix_len)) / (eagle_heads + 1))
        for _ in range(max_remained_steps):
            generated_token, out = self.generate_next_token(
                input_ids=self.input_ids[:, self.input_idx_base:self.input_ids_len],
                past_key_values=self.past_key_values,
            )
            self.input_idx_base = self.input_ids_len
            self.input_ids[:, self.input_ids_len:self.input_ids_len + 1] = generated_token
            self.input_ids_len += 1

            self.check_end_of_generation(generated_token)

            if self.inference_mtp:
                base_hidden_states = [out.hidden_states[id] for id in self.config.eagle_layers_ids]
                fused_hidden_state = self.eagle_model.fuse_base_model_hidden_states(base_hidden_states)
                self.hidden_state = torch.cat([self.hidden_state, fused_hidden_state], dim = 1)           


            for head_id in range(self.config.num_inference_eagle_heads):
                if head_id == 0:
                    generated_token, out = self.eagle_model.generate_next_token(
                        input_ids=self.input_ids[:, self.input_idx_eagle:self.input_ids_len],
                        hidden_states=self.hidden_state,
                        past_key_values=self.eagle_past_key_values,
                    )
                    local_eagle_past_key_values = Cache(layers=[copy.copy(layer) for layer in self.eagle_past_key_values.layers])
                    self.input_idx_eagle = self.input_ids_len
                else:
                    generated_token, out = self.eagle_model.generate_next_token(
                        input_ids=self.input_ids[:, self.input_ids_len - 1:self.input_ids_len],
                        hidden_states=out.last_hidden_state[:,-1:,:],
                        past_key_values=local_eagle_past_key_values,
                    )
                    self.hidden_state = torch.empty((batch_size,0,self.hidden_state.shape[2]),
                                                    dtype=self.hidden_state.dtype,
                                                    device=self.hidden_state.device)
                self.input_ids[:, self.input_ids_len:self.input_ids_len + 1] = generated_token
                self.input_ids_len += 1
                self.check_end_of_generation(generated_token)
            
            # decode every new sequence and count amount of spaces 
            decoded_texts = self.processor.batch_decode(
                self.input_ids[:, self.prefix_len:self.input_ids_len],
                skip_special_tokens=True,
            ) # return list of lists
            for i in range(batch_size):
                if next_action_is_generated[i]:
                    continue

                output = decoded_texts[i].strip().split()

                # if output is invalid just add zeros
                if not all(a.isdigit() and 0 <= int(a) < self.config.n_state_bins for a in output):
                    next_action_is_generated[i] = True
                    decoded_actions[i] = torch.zeros(self.action_dim, device=device, dtype=torch.long)

                # check if we finished next action generation
                if self.generation_finished[i] or len(output) > self.action_dim*(self.action_index + 1):
                    next_action_is_generated[i] = True
                    action_text = output[self.action_dim*(self.action_index):self.action_dim*(self.action_index + 1)]
                    decoded_actions[i] = torch.tensor([int(a) for a in action_text], device=device)
            
            if sum(next_action_is_generated) == len(next_action_is_generated):
                self.action_index += 1
                if self.action_index == self.config.n_action_steps:
                    self.new_obs = True
                return self.reconstruct_actions(decoded_actions, self.generation_batch)

        return torch.zeros((batch_size, 1, self.action_dim), device=device, dtype=torch.long)

    def generate_actions(self, batch: dict[str, torch.Tensor]):
        actions = []
        self.new_obs = True

        for _ in range(self.config.n_action_steps):
            action = self.generate_one_action(batch=batch)
            actions.append(action)

        action_chunk = torch.cat(actions, dim=1)
        return action_chunk
