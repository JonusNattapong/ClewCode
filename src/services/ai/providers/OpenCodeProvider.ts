import { APIError } from '@anthropic-ai/sdk';
import { OpenAICompatibleAdapter, registerAdapter } from '../adapter/AnthropicAdapter.js';
import { ChatGPTResponsesAdapter, type ResponsesClient } from './ChatGPTProvider.js';
import { OpenAICompatibleProvider } from './OpenAICompatibleProvider.js';
import type { ProviderClient, ProviderInitOptions, ProviderInterface } from './ProviderInterface.js';

const OPENCODE_PROVIDER_ID = 'opencode' as const;

function getBaseUrl(options: ProviderInitOptions): string {
  return (options.baseUrl ?? process.env.OPENCODE_BASE_URL ?? 'https://opencode.ai/zen/v1').replace(/\/$/, '');
}

async function parseErrorResponse(response: Response): Promise<Error> {
  const text = await response.text();
  let body: Record<string, unknown> | undefined;
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === 'object') body = parsed as Record<string, unknown>;
  } catch {
    // Preserve the raw response below when it is not JSON.
  }

  const nested = body?.error;
  const message =
    (nested && typeof nested === 'object' && typeof (nested as Record<string, unknown>).message === 'string'
      ? ((nested as Record<string, unknown>).message as string)
      : undefined) ??
    (typeof body?.message === 'string' ? body.message : undefined) ??
    (text || `${response.status} ${response.statusText}`);

  return APIError.generate(response.status, body ?? { error: { message } }, message, response.headers);
}

async function* parseServerSentEvents(response: Response): AsyncGenerator<unknown, void, undefined> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body for OpenCode stream');

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() ?? '';
      for (const eventBlock of events) {
        const dataLines = eventBlock
          .split(/\r?\n/)
          .filter(line => line.startsWith('data:'))
          .map(line => line.slice(5).trimStart());
        if (dataLines.length === 0) continue;
        const data = dataLines.join('\n');
        if (data === '[DONE]') return;
        try {
          yield JSON.parse(data);
        } catch {
          // Ignore keepalive or malformed SSE frames.
        }
      }
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      // Best effort cleanup after an interrupted stream.
    }
    reader.releaseLock();
  }
}

function usesResponsesApi(model: string): boolean {
  const normalized = model.split('/').pop()?.toLowerCase() ?? model.toLowerCase();
  return normalized.startsWith('gpt-') || normalized.startsWith('grok-') || normalized.startsWith('muse-spark-');
}

function usesMessagesApi(model: string): boolean {
  const normalized = model.split('/').pop()?.toLowerCase() ?? model.toLowerCase();
  return normalized.startsWith('claude-') || normalized.startsWith('qwen');
}

function usesGoogleApi(model: string): boolean {
  const normalized = model.split('/').pop()?.toLowerCase() ?? model.toLowerCase();
  return normalized.startsWith('gemini-');
}

class OpenCodeAdapter {
  private readonly responses: ChatGPTResponsesAdapter;
  private readonly chat: OpenAICompatibleAdapter;
  private readonly messages: {
    createMessage(params: any, options?: { signal?: AbortSignal }): Promise<unknown>;
    streamMessage(params: any, options?: { signal?: AbortSignal }): any;
    normalizeError(error: unknown): Error;
  };

  constructor(client: ResponsesClient & { chat: unknown; beta: any }) {
    this.responses = new ChatGPTResponsesAdapter(client, 'OpenCode');
    this.chat = new OpenAICompatibleAdapter(client, OPENCODE_PROVIDER_ID, 'OpenCode');
    this.messages = {
      createMessage: (params, options) => client.beta.messages.create({ ...params, stream: false }, options),
      streamMessage: (params, options) => client.beta.messages.create({ ...params, stream: true }, options),
      normalizeError: error =>
        error instanceof Error ? new Error(`[OpenCode] ${error.message}`) : new Error(`[OpenCode] ${String(error)}`),
    };
  }

  createMessage(params: any, options?: { signal?: AbortSignal }): Promise<unknown> {
    if (usesResponsesApi(params.model)) return this.responses.createMessage(params, options);
    if (usesMessagesApi(params.model)) return this.messages.createMessage(params, options);
    if (usesGoogleApi(params.model))
      throw new Error(`[OpenCode] ${params.model} uses the Google model endpoint, which is not supported yet.`);
    return this.chat.createMessage(params, options);
  }

  streamMessage(params: any, options?: { signal?: AbortSignal }): any {
    if (usesResponsesApi(params.model)) return this.responses.streamMessage(params, options);
    if (usesMessagesApi(params.model)) return this.messages.streamMessage(params, options);
    if (usesGoogleApi(params.model))
      throw new Error(`[OpenCode] ${params.model} uses the Google model endpoint, which is not supported yet.`);
    return this.chat.streamMessage(params, options);
  }

  normalizeError(error: unknown): Error {
    return this.chat.normalizeError(error);
  }

  readonly label = 'OpenCode';
  readonly streamTimeoutMs = 90_000;
}

export class OpenCodeProvider implements ProviderInterface {
  readonly providerId = OPENCODE_PROVIDER_ID;
  readonly label = 'OpenCode';

  getProviderId() {
    return this.providerId;
  }

  getProviderLabel() {
    return this.label;
  }

  getProviderApiKeyEnvVar() {
    return 'OPENCODE_API_KEY';
  }

  async createClient(options: ProviderInitOptions): Promise<ProviderClient> {
    const apiKey = (options.apiKey ?? process.env.OPENCODE_API_KEY)?.trim();
    if (!apiKey) throw new Error('Missing API key for provider opencode. Set OPENCODE_API_KEY.');

    const baseUrl = getBaseUrl(options);
    const responsesClient: ResponsesClient = {
      responses: {
        create: async (params, requestOptions) => {
          const stream = params.stream === true;
          const response = await fetch(`${baseUrl}/responses`, {
            method: 'POST',
            headers: {
              accept: stream ? 'text/event-stream' : 'application/json',
              'content-type': 'application/json',
              authorization: `Bearer ${apiKey}`,
            },
            body: JSON.stringify(params),
            signal: requestOptions?.signal,
          });

          if (!response.ok) throw await parseErrorResponse(response);
          return stream ? parseServerSentEvents(response) : response.json();
        },
      },
    };

    const chatClient = await new OpenAICompatibleProvider(
      OPENCODE_PROVIDER_ID,
      this.label,
      this.getProviderApiKeyEnvVar(),
      baseUrl,
    ).createClient(options);

    const { default: Anthropic } = await import('@anthropic-ai/sdk');
    const messagesClient = new Anthropic({
      apiKey,
      baseURL: baseUrl,
      dangerouslyAllowBrowser: true,
    });

    return { ...responsesClient, chat: (chatClient as any).chat, beta: messagesClient.beta };
  }

  async listModels(options: ProviderInitOptions): Promise<Array<{ id: string; label: string }>> {
    const apiKey = (options.apiKey ?? process.env.OPENCODE_API_KEY)?.trim();
    if (!apiKey) return [];

    try {
      const response = await fetch(`${getBaseUrl(options)}/models`, {
        headers: { accept: 'application/json', authorization: `Bearer ${apiKey}` },
      });
      if (!response.ok) return [];
      const data = (await response.json()) as { data?: Array<{ id?: string }> };
      return (data.data ?? [])
        .filter((model): model is { id: string } => typeof model.id === 'string')
        .map(model => ({ id: model.id, label: model.id }));
    } catch {
      return [];
    }
  }
}

registerAdapter(
  OPENCODE_PROVIDER_ID,
  ((client: ResponsesClient & { chat: unknown; beta: any }) => new OpenCodeAdapter(client)) as any,
);
