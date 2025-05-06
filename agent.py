# agent.py
import requests
import sys
from fastmcp import FastMCP

mcp = FastMCP("panda")  # MCP namespace matching server configuration


class WhoIsAgent:
    """Agent that queries a server for information about a given name using specified LLM."""

    def __init__(self, name: str, model: str = "openai"):
        """Initialize the WhoIsAgent with a name and model choice.

        Args:
            name (str): The name of the person to query.
            model (str, optional): The LLM model to use (openai, anthropic, llama, gemini).
                                   Defaults to "openai".
        """
        self.name = name
        self.model = model
        self.server_url = "http://localhost:8000/query"

    def get_who_is(self) -> str:
        """Send a request to the server to retrieve information about the person.

        Returns:
            str: Response from the selected LLM containing information about the person.
                 If an error occurs, returns an error message.
        """
        prompt = f"Who is {self.name}?"
        response = requests.post(
            self.server_url,
            json={"prompt": prompt, "model": self.model}
        )
        if response.status_code == 200:
            return response.json().get("response")
        else:
            return f"Error: {response.text}"


def main():
    """Entry point of the script to run the WhoIsAgent from command-line arguments."""
    if len(sys.argv) != 3:
        print("Usage: python agent.py <name> <model>")
        print("Model options: openai, anthropic, llama, gemini")
        sys.exit(1)

    name, model = sys.argv[1], sys.argv[2].lower()

    agent = WhoIsAgent(name, model)
    result = agent.get_who_is()
    print(f"Answer from {model.capitalize()}:\n{result}")


if __name__ == "__main__":
    main()
