"""movie_rag Package Metadata

The movie_rag package implements three scripts:
  - stager.py => initial preparation of incoming movie data for further processing.
  - prepper.py => prepare the incoming movie data for chatbot interaction.
  - chat_server.py => provide an online chatbot interface to the prepared movie data.

The chatbot script (chat_server.py) uses the MovieChatter class in chatter.py to manage the
interaction between the user and LLMs. That class uses agent classes from agents.py
to access the prepared movie data.

1.0.0:
  - Original version.
1.0.1:
  - Working prototype with SQLAgent, VectorAgent, GraphAgent in a LangChain graph using Ollama.
1.0.2:
  - Refactoring, editing, improvement of queries and graph control.
1.2.0:
  - Refactored version of chatter called MovieChatter.
1.3.0:
  - Initial implementation of chat history.

"""

__all__ = [
    '__title__', '__summary__', '__author__', '__copyright__', '__version__',
    '__date__', '__status__',
]

__title__ = 'movie_rag'
__summary__ = 'A showcase chatbot with data prep pipeline, to interact with movies/actors.'
__author__ = 'Rana Coskun, John Thompson'
__copyright__ = 'Copyright 2025, Information Professionals GmbH'
__version__ = '1.3.0'
__date__ = '24 Jun 2025'
__status__ = 'Stable'
