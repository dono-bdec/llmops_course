#Tools serve as interfaces for agents, chains, or LLMs to interact with the external world

from langchain_community.tools import YouTubeSearchTool
tool = YouTubeSearchTool()
tool.run(“Andrej Karpathy")
