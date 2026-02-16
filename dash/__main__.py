"""CLI entry point: python -m dash"""

from dash.dash_agent import DashAgent, TextDelta, ToolCallStarted, ToolCallCompleted, StreamDone


def main():
    """Simple CLI chat loop."""
    print("Dash - Data Agent")
    print("Type 'exit' or 'quit' to exit\n")

    agent = DashAgent()

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            print("\nDash: ", end="", flush=True)
            for event in agent.run(user_input):
                if isinstance(event, TextDelta):
                    print(event.content, end="", flush=True)
                elif isinstance(event, ToolCallStarted):
                    print(f"\n[Running {event.tool_name}...]", end="", flush=True)
                elif isinstance(event, ToolCallCompleted):
                    print(" done", end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
