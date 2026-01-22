"""
Tests for the ChatLiteLLM unified adapter.

These tests verify that ChatLiteLLM correctly implements the BaseChatModel protocol
and can be used as a drop-in replacement for provider-specific adapters.
"""

import pytest
from pydantic import BaseModel

from browser_use.llm.litellm_chat import ChatLiteLLM, create_anthropic_chat, create_google_chat, create_openai_chat
from browser_use.llm.messages import AssistantMessage, SystemMessage, UserMessage
from browser_use.llm.views import ChatInvokeCompletion


class TestChatLiteLLMBasics:
	"""Test basic ChatLiteLLM functionality."""

	def test_provider_extraction_with_prefix(self):
		"""Test that provider is correctly extracted from model name with prefix."""
		llm = ChatLiteLLM(model='openai/gpt-4o')
		assert llm.provider == 'openai'

		llm = ChatLiteLLM(model='anthropic/claude-3-sonnet-20240229')
		assert llm.provider == 'anthropic'

		llm = ChatLiteLLM(model='gemini/gemini-2.0-flash')
		assert llm.provider == 'gemini'

	def test_provider_extraction_without_prefix(self):
		"""Test that provider defaults to openai when no prefix is present."""
		llm = ChatLiteLLM(model='gpt-4o')
		assert llm.provider == 'openai'

	def test_name_property(self):
		"""Test that name returns the full model name."""
		llm = ChatLiteLLM(model='openai/gpt-4o')
		assert llm.name == 'openai/gpt-4o'

	def test_default_parameters(self):
		"""Test default parameter values."""
		llm = ChatLiteLLM(model='openai/gpt-4o')
		assert llm.temperature == 0.2
		assert llm.max_tokens == 8192
		assert llm.timeout == 90.0
		assert llm.max_retries == 5
		assert llm.drop_params is True

	def test_custom_parameters(self):
		"""Test custom parameter values."""
		llm = ChatLiteLLM(
			model='openai/gpt-4o',
			temperature=0.7,
			max_tokens=4096,
			api_key='test-key',
			api_base='https://custom.api.com',
			timeout=120.0,
			max_retries=3,
		)
		assert llm.temperature == 0.7
		assert llm.max_tokens == 4096
		assert llm.api_key == 'test-key'
		assert llm.api_base == 'https://custom.api.com'
		assert llm.timeout == 120.0
		assert llm.max_retries == 3


class TestMessageConversion:
	"""Test message conversion to OpenAI format."""

	def test_convert_system_message(self):
		"""Test converting a system message."""
		llm = ChatLiteLLM(model='openai/gpt-4o')
		messages = [SystemMessage(content='You are a helpful assistant.')]

		converted = llm._convert_messages(messages)

		assert len(converted) == 1
		assert converted[0]['role'] == 'system'
		assert converted[0]['content'] == 'You are a helpful assistant.'

	def test_convert_user_message(self):
		"""Test converting a user message."""
		llm = ChatLiteLLM(model='openai/gpt-4o')
		messages = [UserMessage(content='Hello!')]

		converted = llm._convert_messages(messages)

		assert len(converted) == 1
		assert converted[0]['role'] == 'user'
		assert converted[0]['content'] == 'Hello!'

	def test_convert_assistant_message(self):
		"""Test converting an assistant message."""
		llm = ChatLiteLLM(model='openai/gpt-4o')
		messages = [AssistantMessage(content='Hi there!')]

		converted = llm._convert_messages(messages)

		assert len(converted) == 1
		assert converted[0]['role'] == 'assistant'
		assert converted[0]['content'] == 'Hi there!'

	def test_convert_conversation(self):
		"""Test converting a full conversation."""
		llm = ChatLiteLLM(model='openai/gpt-4o')
		messages = [
			SystemMessage(content='You are a helpful assistant.'),
			UserMessage(content='Hello!'),
			AssistantMessage(content='Hi there!'),
			UserMessage(content='How are you?'),
		]

		converted = llm._convert_messages(messages)

		assert len(converted) == 4
		assert converted[0]['role'] == 'system'
		assert converted[1]['role'] == 'user'
		assert converted[2]['role'] == 'assistant'
		assert converted[3]['role'] == 'user'


class TestCompletionParams:
	"""Test completion parameter building."""

	def test_get_completion_params_default(self):
		"""Test default completion parameters."""
		llm = ChatLiteLLM(model='openai/gpt-4o')
		params = llm._get_completion_params()

		assert params['model'] == 'openai/gpt-4o'
		assert params['max_tokens'] == 8192
		assert params['timeout'] == 90.0
		assert params['num_retries'] == 5
		assert params['drop_params'] is True
		assert params['temperature'] == 0.2

	def test_get_completion_params_with_api_key(self):
		"""Test completion parameters with API key."""
		llm = ChatLiteLLM(model='openai/gpt-4o', api_key='test-key')
		params = llm._get_completion_params()

		assert params['api_key'] == 'test-key'

	def test_get_completion_params_with_extra_headers(self):
		"""Test completion parameters with extra headers."""
		llm = ChatLiteLLM(
			model='anthropic/claude-3-sonnet-20240229',
			extra_headers={'anthropic-beta': 'prompt-caching-2024-07-31'},
		)
		params = llm._get_completion_params()

		assert params['extra_headers'] == {'anthropic-beta': 'prompt-caching-2024-07-31'}


class TestConvenienceFunctions:
	"""Test convenience functions for creating ChatLiteLLM instances."""

	def test_create_openai_chat(self):
		"""Test creating an OpenAI chat instance."""
		llm = create_openai_chat('gpt-4o', temperature=0.5)
		assert llm.model == 'openai/gpt-4o'
		assert llm.provider == 'openai'
		assert llm.temperature == 0.5

	def test_create_openai_chat_with_prefix(self):
		"""Test creating an OpenAI chat instance with prefix already present."""
		llm = create_openai_chat('openai/gpt-4o')
		assert llm.model == 'openai/gpt-4o'

	def test_create_anthropic_chat(self):
		"""Test creating an Anthropic chat instance."""
		llm = create_anthropic_chat('claude-3-sonnet-20240229')
		assert llm.model == 'anthropic/claude-3-sonnet-20240229'
		assert llm.provider == 'anthropic'

	def test_create_google_chat(self):
		"""Test creating a Google Gemini chat instance."""
		llm = create_google_chat('gemini-2.0-flash')
		assert llm.model == 'gemini/gemini-2.0-flash'
		assert llm.provider == 'gemini'


class TestStructuredOutput:
	"""Test structured output schema preparation."""

	def test_structured_output_schema(self):
		"""Test that structured output schema is correctly prepared."""

		class TestOutput(BaseModel):
			name: str
			value: int

		llm = ChatLiteLLM(model='openai/gpt-4o')

		# We can't test the full ainvoke without mocking litellm,
		# but we can test that the model implements the protocol correctly
		assert hasattr(llm, 'ainvoke')
		assert hasattr(llm, 'provider')
		assert hasattr(llm, 'name')


class TestUsageParsing:
	"""Test usage information parsing."""

	def test_parse_usage_none(self):
		"""Test parsing when usage is None."""
		llm = ChatLiteLLM(model='openai/gpt-4o')

		class MockResponse:
			usage = None

		result = llm._parse_usage(MockResponse())
		assert result is None

	def test_parse_usage_basic(self):
		"""Test parsing basic usage information."""
		llm = ChatLiteLLM(model='openai/gpt-4o')

		class MockUsage:
			prompt_tokens = 100
			completion_tokens = 50
			total_tokens = 150
			prompt_tokens_details = None

		class MockResponse:
			usage = MockUsage()

		result = llm._parse_usage(MockResponse())

		assert result is not None
		assert result.prompt_tokens == 100
		assert result.completion_tokens == 50
		assert result.total_tokens == 150


# Integration tests (require API keys)
@pytest.mark.integration
class TestIntegration:
	"""Integration tests that require actual API keys."""

	@pytest.mark.asyncio
	async def test_openai_simple_completion(self):
		"""Test simple completion with OpenAI."""
		llm = ChatLiteLLM(model='openai/gpt-4o-mini', max_tokens=100)
		messages = [
			SystemMessage(content='You are a helpful assistant. Respond briefly.'),
			UserMessage(content='Say hello in one word.'),
		]

		result = await llm.ainvoke(messages)

		assert isinstance(result, ChatInvokeCompletion)
		assert isinstance(result.completion, str)
		assert len(result.completion) > 0

	@pytest.mark.asyncio
	async def test_anthropic_simple_completion(self):
		"""Test simple completion with Anthropic."""
		llm = ChatLiteLLM(model='anthropic/claude-3-haiku-20240307', max_tokens=100)
		messages = [
			SystemMessage(content='You are a helpful assistant. Respond briefly.'),
			UserMessage(content='Say hello in one word.'),
		]

		result = await llm.ainvoke(messages)

		assert isinstance(result, ChatInvokeCompletion)
		assert isinstance(result.completion, str)
		assert len(result.completion) > 0

	@pytest.mark.asyncio
	async def test_structured_output(self):
		"""Test structured output with OpenAI."""

		class Greeting(BaseModel):
			message: str
			language: str

		llm = ChatLiteLLM(model='openai/gpt-4o-mini', max_tokens=100)
		messages = [
			SystemMessage(content='You are a helpful assistant.'),
			UserMessage(content='Generate a greeting in Spanish.'),
		]

		result = await llm.ainvoke(messages, output_format=Greeting)

		assert isinstance(result, ChatInvokeCompletion)
		assert isinstance(result.completion, Greeting)
		assert result.completion.message is not None
		assert result.completion.language is not None

