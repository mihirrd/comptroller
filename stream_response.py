import time
import tiktoken
from openai import OpenAI

client = OpenAI()

MODEL = "gpt-4o-mini"
enc = tiktoken.encoding_for_model(MODEL)

def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def stream_with_live_token_usage(prompt: str):
    total_output_tokens = 0
    start_time = time.time()
    full_response = ""

    print("Streaming response:\n")

    # ✅ MUST use context manager
    with client.responses.stream(
        model=MODEL,
        input=prompt,
    ) as stream:

        # ✅ Now it's iterable
        for event in stream:
            if event.type == "response.output_text.delta":
                chunk = event.delta
                full_response += chunk
                tokens = count_tokens(chunk)
                total_output_tokens += tokens

                elapsed = time.time() - start_time

                # print(chunk, end="", flush=True)
                # print(
                #     f"\n[Tokens so far: {total_output_tokens} | {elapsed:.2f}s]\n",
                #     end="",
                #     flush=True,
                # )

            elif event.type == "response.completed":
                usage = event.response.usage
                print("\n=== FINAL USAGE (SERVER) ===")
                print(f"Input tokens:  {usage.input_tokens}")
                print(f"Output tokens: {usage.output_tokens}")
                print(f"Total tokens:  {usage.total_tokens}")

    return full_response


if __name__ == "__main__":
    response = stream_with_live_token_usage(
        "Explain how Raft consensus works step by step."
    )
    print(response)
