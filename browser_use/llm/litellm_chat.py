"""
Unified LLM adapter using LiteLLM.

LiteLLM provides a unified interface for calling 100+ LLM APIs using the OpenAI format.
This adapter replaces the individual provider-specific adapters (anthropic, openai, google, etc.)
with a single unified implementation.

Usage:
    from browser_use.llm import ChatLiteLLM

    # OpenAI
    llm = ChatLiteLLM(model="openai/gpt-4o")

    # Anthropic
    llm = ChatLiteLLM(model="anthropic/claude-3-sonnet-20240229")

    # Google
    llm = ChatLiteLLM(model="gemini/gemini-2.0-flash")

    # Azure OpenAI
    llm = ChatLiteLLM(model="azure/gpt-4o", api_base="https://your-resource.openai.azure.com")

    # Ollama
    llm = ChatLiteLLM(model="ollama/llama3")
"""

import json
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import (
	AssistantMessage,
	BaseMessage,
	ContentPartImageParam,
	ContentPartTextParam,
	SystemMessage,
	UserMessage,
)
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatLiteLLM(BaseChatModel):
	"""
	A unified LLM adapter using LiteLLM.

	LiteLLM provides a consistent interface across 100+ LLM providers,
	automatically handling message format conversion, API differences,
	and structured output.

	Args:
		model: Model identifier in LiteLLM format (e.g., "openai/gpt-4o", "anthropic/claude-3-sonnet-20240229")
		temperature: Temperature for response generation (default: 0.2)
		max_tokens: Maximum tokens in the response (default: 8192)
		top_p: Top-p sampling parameter (default: None)
		seed: Random seed for reproducibility (default: None)
		api_key: API key for the provider (default: None, uses environment variable)
		api_base: Base URL for the API (default: None)
		timeout: Request timeout in seconds (default: 90)
		max_retries: Number of retries for failed requests (default: 5)
		drop_params: Drop unsupported params instead of raising error (default: True)

	Example:
		llm = ChatLiteLLM(model="openai/gpt-4o", temperature=0.5)
		response = await llm.ainvoke(messages)
	"""

	# Model configuration
	model: str

	# Generation parameters
	temperature: float | None = 0.2
	max_tokens: int = 8192
	top_p: float | None = None
	seed: int | None = None

	# Client configuration
	api_key: str | None = None
	api_base: str | None = None
	timeout: float = 90.0
	max_retries: int = 5

	# LiteLLM-specific options
	drop_params: bool = True  # Drop unsupported params instead of raising error

	# Additional headers (e.g., for Anthropic cache control)
	extra_headers: dict[str, str] | None = None

	# Provider-specific options passed to litellm
	extra_kwargs: dict[str, Any] = field(default_factory=dict)

	@property
	def provider(self) -> str:
		"""Extract provider from model string (e.g., 'openai' from 'openai/gpt-4o')."""
		if '/' in self.model:
			return self.model.split('/')[0]
		# Default to openai for models without prefix
		return 'openai'

	@property
	def name(self) -> str:
		"""Return the full model name."""
		return self.model

	def _convert_messages(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
		"""
		Convert internal message format to OpenAI-compatible format for LiteLLM.

		LiteLLM uses OpenAI's message format as the standard input format
		and automatically converts it to the appropriate format for each provider.
		"""
		converted: list[dict[str, Any]] = []

		for message in messages:
			if isinstance(message, SystemMessage):
				content = self._convert_content(message.content)
				converted.append({'role': 'system', 'content': content})

			elif isinstance(message, UserMessage):
				content = self._convert_user_content(message.content)
				msg: dict[str, Any] = {'role': 'user', 'content': content}
				if message.name:
					msg['name'] = message.name
				converted.append(msg)

			elif isinstance(message, AssistantMessage):
				msg = {'role': 'assistant'}
				if message.content is not None:
					msg['content'] = self._convert_content(message.content)
				if message.tool_calls:
					msg['tool_calls'] = [
						{
							'id': tc.id,
							'type': 'function',
							'function': {
								'name': tc.function.name,
								'arguments': tc.function.arguments,
							},
						}
						for tc in message.tool_calls
					]
				converted.append(msg)

		return converted

	def _convert_content(
		self, content: str | list[ContentPartTextParam] | list[ContentPartTextParam | ContentPartImageParam] | None
	) -> str | list[dict[str, Any]]:
		"""Convert content to OpenAI format."""
		if content is None:
			return ''
		if isinstance(content, str):
			return content

		parts: list[dict[str, Any]] = []
		for part in content:
			if hasattr(part, 'type'):
				if part.type == 'text':
					parts.append({'type': 'text', 'text': part.text})
				elif part.type == 'image_url' and hasattr(part, 'image_url'):
					parts.append({
						'type': 'image_url',
						'image_url': {
							'url': part.image_url.url,
							'detail': part.image_url.detail,
						},
					})
		return parts if parts else ''

	def _convert_user_content(
		self, content: str | list[ContentPartTextParam | ContentPartImageParam]
	) -> str | list[dict[str, Any]]:
		"""Convert user message content to OpenAI format."""
		return self._convert_content(content)

	def _get_completion_params(self) -> dict[str, Any]:
		"""Build parameters for litellm.acompletion call."""
		params: dict[str, Any] = {
			'model': self.model,
			'max_tokens': self.max_tokens,
			'timeout': self.timeout,
			'num_retries': self.max_retries,
			'drop_params': self.drop_params,
		}

		if self.temperature is not None:
			params['temperature'] = self.temperature

		if self.top_p is not None:
			params['top_p'] = self.top_p

		if self.seed is not None:
			params['seed'] = self.seed

		if self.api_key is not None:
			params['api_key'] = self.api_key

		if self.api_base is not None:
			params['api_base'] = self.api_base

		if self.extra_headers is not None:
			params['extra_headers'] = self.extra_headers

		# Merge any additional kwargs
		params.update(self.extra_kwargs)

		return params

	def _parse_usage(self, response: Any) -> ChatInvokeUsage | None:
		"""Parse usage information from LiteLLM response."""
		if not hasattr(response, 'usage') or response.usage is None:
			return None

		usage = response.usage

		# LiteLLM returns usage in OpenAI format
		prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
		completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
		total_tokens = getattr(usage, 'total_tokens', 0) or (prompt_tokens + completion_tokens)

		# Try to extract cached tokens if available (provider-specific)
		prompt_cached_tokens = None
		if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
			prompt_cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', None)

		# Anthropic-specific: cache creation tokens
		prompt_cache_creation_tokens = None
		if hasattr(usage, 'cache_creation_input_tokens'):
			prompt_cache_creation_tokens = usage.cache_creation_input_tokens
		elif hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
			prompt_cache_creation_tokens = getattr(usage.prompt_tokens_details, 'cache_creation_input_tokens', None)

		return ChatInvokeUsage(
			prompt_tokens=prompt_tokens,
			completion_tokens=completion_tokens,
			total_tokens=total_tokens,
			prompt_cached_tokens=prompt_cached_tokens,
			prompt_cache_creation_tokens=prompt_cache_creation_tokens,
			prompt_image_tokens=None,  # Not commonly available
		)

	def _get_stop_reason(self, response: Any) -> str | None:
		"""Extract stop reason from LiteLLM response."""
		if hasattr(response, 'choices') and response.choices:
			choice = response.choices[0]
			if hasattr(choice, 'finish_reason'):
				return choice.finish_reason
		return None

	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
	) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any
	) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the LLM with the given messages.

		Args:
			messages: List of chat messages
			output_format: Optional Pydantic model class for structured output
			**kwargs: Additional parameters passed to litellm.acompletion

		Returns:
			ChatInvokeCompletion with either a string or structured output
		"""
		# Import litellm lazily to avoid import overhead
		import litellm

		# Convert messages to OpenAI format
		openai_messages = self._convert_messages(messages)

		# Build completion parameters
		params = self._get_completion_params()
		params['messages'] = openai_messages

		# Merge any additional kwargs from the call
		params.update(kwargs)

		try:
			if output_format is None:
				# Simple text completion
				response = await litellm.acompletion(**params)

				content = ''
				if hasattr(response, 'choices') and response.choices:
					message = response.choices[0].message
					if hasattr(message, 'content') and message.content:
						content = message.content

				return ChatInvokeCompletion(
					completion=content,
					usage=self._parse_usage(response),
					stop_reason=self._get_stop_reason(response),
				)

			else:
				# Structured output using response_format
				schema = SchemaOptimizer.create_optimized_json_schema(output_format)

				# Use JSON schema response format
				params['response_format'] = {
					'type': 'json_schema',
					'json_schema': {
						'name': output_format.__name__,
						'strict': True,
						'schema': schema,
					},
				}

				response = await litellm.acompletion(**params)

				content = ''
				if hasattr(response, 'choices') and response.choices:
					message = response.choices[0].message
					if hasattr(message, 'content') and message.content:
						content = message.content

				if not content:
					raise ModelProviderError(
						message='No content in structured output response',
						status_code=500,
						model=self.model,
					)

				# Parse the JSON response
				try:
					# Handle potential markdown code blocks
					text = content.strip()
					if text.startswith('```json') and text.endswith('```'):
						text = text[7:-3].strip()
					elif text.startswith('```') and text.endswith('```'):
						text = text[3:-3].strip()

					parsed_data = json.loads(text)
					parsed = output_format.model_validate(parsed_data)

					return ChatInvokeCompletion(
						completion=parsed,
						usage=self._parse_usage(response),
						stop_reason=self._get_stop_reason(response),
					)

				except (json.JSONDecodeError, ValueError) as e:
					raise ModelProviderError(
						message=f'Failed to parse structured output: {e}. Response: {content[:200]}',
						status_code=500,
						model=self.model,
					) from e

		except Exception as e:
			# Handle LiteLLM-specific exceptions
			error_message = str(e)
			status_code = 502

			# Check for rate limit errors
			if 'rate' in error_message.lower() and 'limit' in error_message.lower():
				raise ModelRateLimitError(
					message=error_message,
					status_code=429,
					model=self.model,
				) from e

			# Check for authentication errors
			if any(indicator in error_message.lower() for indicator in ['auth', '401', '403', 'api key', 'invalid key']):
				status_code = 401

			# Check for timeout errors
			if 'timeout' in error_message.lower():
				status_code = 408

			# Check for server errors
			if any(indicator in error_message.lower() for indicator in ['500', '502', '503', '504']):
				status_code = 503

			raise ModelProviderError(
				message=error_message,
				status_code=status_code,
				model=self.model,
			) from e


# Convenience aliases for common providers
def create_openai_chat(model: str = 'gpt-4o', **kwargs: Any) -> ChatLiteLLM:
	"""Create a ChatLiteLLM instance for OpenAI models."""
	if not model.startswith('openai/'):
		model = f'openai/{model}'
	return ChatLiteLLM(model=model, **kwargs)


def create_anthropic_chat(model: str = 'claude-3-sonnet-20240229', **kwargs: Any) -> ChatLiteLLM:
	"""Create a ChatLiteLLM instance for Anthropic models."""
	if not model.startswith('anthropic/'):
		model = f'anthropic/{model}'
	return ChatLiteLLM(model=model, **kwargs)


def create_google_chat(model: str = 'gemini-2.0-flash', **kwargs: Any) -> ChatLiteLLM:
	"""Create a ChatLiteLLM instance for Google Gemini models."""
	if not model.startswith('gemini/'):
		model = f'gemini/{model}'
	return ChatLiteLLM(model=model, **kwargs)

