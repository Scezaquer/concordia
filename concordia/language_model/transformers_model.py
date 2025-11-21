"""Language Model that uses HuggingFace Transformers for local inference.

This wrapper allows Concordia to use locally hosted models through Transformers.
It supports loading a base model once and attaching multiple LoRA adapters to it.
"""

from collections.abc import Collection, Sequence
from typing import Any, Mapping

from concordia.language_model import language_model
from concordia.utils.deprecated import measurements as measurements_lib
from typing_extensions import override
import torch

try:
  from transformers import AutoModelForCausalLM, AutoTokenizer
  TRANSFORMERS_AVAILABLE = True
except ImportError:
  TRANSFORMERS_AVAILABLE = False

try:
  from peft import PeftModel
  PEFT_AVAILABLE = True
except ImportError:
  PEFT_AVAILABLE = False

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TransformersLanguageModel(language_model.LanguageModel):
  """Language model wrapper for Transformers local inference."""

  def __init__(
      self,
      model_name: str,
      *,
      device: str = _DEFAULT_DEVICE,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      **kwargs: Any,
  ):
    """Initialize the Transformers language model.

    Args:
      model_name: The name or path of the model to load.
      device: Device to load the model on (e.g., 'cuda', 'cpu').
      measurements: Measurements object for logging statistics.
      channel: Channel name for measurements.
      **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained.
    """
    if not TRANSFORMERS_AVAILABLE:
      raise ImportError(
          "Transformers is required but not installed. "
          "Install it with: pip install transformers"
      )

    self._model_name = model_name
    self._device = device
    self._measurements = measurements
    self._channel = channel
    
    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure pad_token is set
    if self._tokenizer.pad_token is None:
        self._tokenizer.pad_token = self._tokenizer.eos_token

    self._model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map=device, 
        **kwargs
    )
    
    self._is_peft_model = False
    self._adapter_count = 0

  def load_adapter(self, lora_path: str) -> str:
      """Loads a LoRA adapter and returns its name."""
      if not PEFT_AVAILABLE:
          raise ImportError("PEFT is required for LoRA support.")
      
      adapter_name = f"adapter_{self._adapter_count}"
      self._adapter_count += 1

      if not self._is_peft_model:
          self._model = PeftModel.from_pretrained(
              self._model, 
              lora_path, 
              adapter_name=adapter_name
          )
          self._is_peft_model = True
      else:
          self._model.load_adapter(lora_path, adapter_name=adapter_name)
      
      return adapter_name

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
      adapter_name: str | None = None,
  ) -> str:
    """Sample text from the model."""
    
    if seed is not None:
        torch.manual_seed(seed)

    inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
    
    # Prepare generation args
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": temperature > 0,
        "pad_token_id": self._tokenizer.pad_token_id,
        "eos_token_id": self._tokenizer.eos_token_id,
    }

    if self._is_peft_model:
        if adapter_name:
            self._model.set_adapter(adapter_name)
            outputs = self._model.generate(**inputs, **gen_kwargs)
        else:
            with self._model.disable_adapter():
                outputs = self._model.generate(**inputs, **gen_kwargs)
    else:
        outputs = self._model.generate(**inputs, **gen_kwargs)

    # Decode only the new tokens
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Handle terminators
    if terminators:
        for term in terminators:
            if term in generated_text:
                generated_text = generated_text.split(term)[0]
                break

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(generated_text)},
      )

    return generated_text

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
      adapter_name: str | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    
    if not responses:
        raise ValueError("No responses provided.")

    if seed is not None:
        torch.manual_seed(seed)

    def score_func():
        logprobs = []
        for response in responses:
            score = self._score_response(prompt, response)
            logprobs.append(score)
        return logprobs

    if self._is_peft_model:
        if adapter_name:
            self._model.set_adapter(adapter_name)
            logprobs = score_func()
        else:
            with self._model.disable_adapter():
                logprobs = score_func()
    else:
        logprobs = score_func()

    best_idx = int(max(range(len(logprobs)), key=lambda i: logprobs[i]))

    debug_info = {
        'logprobs': {response: logprobs[i] for i, response in enumerate(responses)},
        'method': 'logprobs'
    }

    if self._measurements is not None:
        self._measurements.publish_datum(
            self._channel,
            {'choice_method': 'logprobs', 'num_choices': len(responses)},
        )

    return best_idx, responses[best_idx], debug_info

  def _score_response(self, prompt: str, response: str) -> float:
      full_text = prompt + response
      full_input = self._tokenizer(full_text, return_tensors="pt").to(self._device)
      
      prompt_input = self._tokenizer(prompt, return_tensors="pt")
      prompt_len = prompt_input.input_ids.shape[1]
      
      if full_input.input_ids.shape[1] <= prompt_len:
          return -float('inf')

      with torch.no_grad():
          outputs = self._model(**full_input)
          logits = outputs.logits
      
      target_ids = full_input.input_ids[0, prompt_len:]
      # logits[i] predicts input_ids[i+1]
      # We want logits for indices [prompt_len-1, ..., end-2]
      # to predict input_ids [prompt_len, ..., end-1]
      
      start_idx = prompt_len - 1
      end_idx = start_idx + len(target_ids)
      
      target_logits = logits[0, start_idx:end_idx]
      
      loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
      loss = loss_fct(target_logits, target_ids)
      
      return -loss.item()


class TransformersLora(language_model.LanguageModel):
  """Language model wrapper for Transformers local inference with LoRA."""

  def __init__(
      self,
      model_name: str,
      *,
      lora_path: str | None = None,
      transformers_language_model: TransformersLanguageModel | None = None,
      device: str = _DEFAULT_DEVICE,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      **kwargs: Any,
  ):
    """Initialize the Transformers language model with LoRA.

    Args:
      model_name: The name or path of the model to load.
      lora_path: Path to LoRA adapter weights (must be provided).
      transformers_language_model: An existing TransformersLanguageModel instance.
        If None, a new one is created.
      device: Device to load the model on.
      measurements: Measurements object for logging statistics.
      channel: Channel name for measurements.
      **kwargs: Additional arguments passed to TransformersLanguageModel.
    """
    if transformers_language_model is not None:
        self._model = transformers_language_model
    else:
        self._model = TransformersLanguageModel(
            model_name=model_name,
            device=device,
            measurements=measurements,
            channel=channel,
            **kwargs
        )
    
    if lora_path is None:
        raise ValueError("lora_path must be provided.")
        
    self._adapter_name = self._model.load_adapter(lora_path)

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    return self._model.sample_text(
        prompt,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=timeout,
        seed=seed,
        adapter_name=self._adapter_name,
    )

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    return self._model.sample_choice(
        prompt,
        responses,
        seed=seed,
        adapter_name=self._adapter_name,
    )
