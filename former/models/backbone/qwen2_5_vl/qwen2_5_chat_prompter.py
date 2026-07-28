"""
llama2_prompter.py

Defines a PromptBuilder for building LLaMa-2 Chat Prompts --> not sure if this is "optimal", but this is the pattern
that's used by HF and other online tutorials.

Reference: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
"""

from typing import Any, Optional

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


def format_system_prompt(system_prompt: str) -> str:
    return f"""
        <|im_start|>system
        {system_prompt}<|im_end|>
        """


class Qwen2_5ChatPromptBuilder:
    def __init__(
        self, model_family: str, system_prompt: Optional[str] = None, processor: AutoProcessor | None = None
    ) -> None:
        super().__init__()
        self.system_prompt = format_system_prompt('You are a helpful assistant.')

        # LLaMa-2 Specific
        self.bos, self.eos = '<|im_start|>', '<|im_end|>'

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f'{self.bos} {msg} {self.eos}'
        self.wrap_gpt = lambda msg: f'{msg if msg != "" else " "}{self.eos}'

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = '', 0

        self.messages: list[dict[str, Any]] = []
        self.processor = processor

    def add_turn(self, role: str, message: str) -> str:
        assert (role == 'human') if (self.turn_count % 2 == 0) else (role == 'gpt')
        message = message.replace('<image>', '').strip()

        self.messages.append({'role': role, 'content': message})

        return self._get_messages()

    def clear_prompt(self):
        self.messages = []
        self.turn_count = 0

    def get_potential_prompt(self, message: str) -> None:
        raise NotImplementedError('Qwen2_5ChatPromptBuilder does not support get_potential_prompt')

    def get_prompt(self) -> str:
        return self._get_messages()

    def _get_messages(self):
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(self.messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        )
        return inputs
