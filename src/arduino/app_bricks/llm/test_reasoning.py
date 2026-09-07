from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_utils import App
from arduino.app_bricks.llm import ReasoningChunk, ContentChunk

llm = LargeLanguageModel(model="genie:qwen3_8b")

def ask_prompt():
    print_reasoning = True
    print_answer = True
    for chunk in llm.chat_stream_reasoning("Why is the sky blue?"):
        if isinstance(chunk, ReasoningChunk):
            if print_reasoning:
                print(f"[reasoning] ", end="", flush=True)
                print_reasoning = False
            print(f"{chunk.content}", end="", flush=True)
        elif isinstance(chunk, ContentChunk):
            if print_answer:
                print(f"[answer] ", end="", flush=True)
                print_answer = False
            print(f"{chunk.content}", end="", flush=True)
    print()
    raise StopIteration  # This stops the user loop


App.run(user_loop=ask_prompt)
