import { expect, test } from 'bun:test';
import { getAdapter } from '../adapter/AnthropicAdapter.js';
import { OpenCodeProvider } from './OpenCodeProvider.js';

test('OpenCode uses the Responses API endpoint for Muse Spark models', async () => {
  const originalFetch = globalThis.fetch;
  const calls: Array<{ url: string; body: Record<string, unknown> }> = [];

  globalThis.fetch = (async (input, init) => {
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    calls.push({ url: String(input), body });
    return new Response(
      JSON.stringify({
        id: 'resp_test',
        model: body.model,
        status: 'completed',
        output_text: 'Done',
        output: [],
        usage: { input_tokens: 1, output_tokens: 1 },
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    );
  }) as typeof fetch;

  try {
    const provider = new OpenCodeProvider();
    const client = (await provider.createClient({ apiKey: 'test-key', baseUrl: 'https://example.test/v1' })) as any;
    const adapterFactory = getAdapter('opencode');
    expect(adapterFactory).toBeDefined();

    const adapter = adapterFactory!(client, 'opencode');
    const message = await adapter.createMessage({
      model: 'muse-spark-1.3-contributor-free',
      max_tokens: 16,
      messages: [{ role: 'user', content: 'hello' }],
    });

    expect(message.content as any).toEqual([{ type: 'text', text: 'Done' }]);
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe('https://example.test/v1/responses');
    expect(calls[0]?.body).toMatchObject({ model: 'muse-spark-1.3-contributor-free', stream: false });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('OpenCode keeps chat-completions models on the chat endpoint', async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = '';

  globalThis.fetch = (async (input, init) => {
    requestedUrl = String(input);
    expect(init?.body).toContain('deepseek-v4-flash');
    return new Response(
      JSON.stringify({
        id: 'chatcmpl_test',
        model: 'deepseek-v4-flash',
        choices: [{ index: 0, message: { role: 'assistant', content: 'Done' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    );
  }) as typeof fetch;

  try {
    const provider = new OpenCodeProvider();
    const client = (await provider.createClient({ apiKey: 'test-key', baseUrl: 'https://example.test/v1' })) as any;
    const adapterFactory = getAdapter('opencode');
    const adapter = adapterFactory!(client, 'opencode');
    const message = await adapter.createMessage({
      model: 'deepseek-v4-flash',
      max_tokens: 16,
      messages: [{ role: 'user', content: 'hello' }],
    });

    expect(message.content as any).toEqual([{ type: 'text', text: 'Done' }]);
    expect(requestedUrl).toBe('https://example.test/v1/chat/completions');
  } finally {
    globalThis.fetch = originalFetch;
  }
});
