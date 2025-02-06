


* Introduction
On this page[![Open on GitHub](https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github&logoColor=white)](https://github.com/langchain-ai/langchain/blob/master/docs/docs/introduction.mdx)

Introduction
============

**LangChain** is a framework for developing applications powered by large language models (LLMs).

LangChain simplifies every stage of the LLM application lifecycle:

* **Development**: Build your applications using LangChain's open-source [components](/docs/concepts/) and [third-party integrations](/docs/integrations/providers/).
  Use [LangGraph](/docs/concepts/architecture/#langgraph) to build stateful agents with first-class streaming and human-in-the-loop support.
* **Productionization**: Use [LangSmith](https://docs.smith.langchain.com/) to inspect, monitor and evaluate your applications, so that you can continuously optimize and deploy with confidence.
* **Deployment**: Turn your LangGraph applications into production-ready APIs and Assistants with [LangGraph Platform](https://langchain-ai.github.io/langgraph/cloud/).


![Diagram outlining the hierarchical organization of the LangChain framework, displaying the interconnected parts across multiple layers.](/svg/langchain_stack_112024.svg "LangChain Framework Overview")![Diagram outlining the hierarchical organization of the LangChain framework, displaying the interconnected parts across multiple layers.](/svg/langchain_stack_112024_dark.svg "LangChain Framework Overview")

LangChain implements a standard interface for large language models and related
technologies, such as embedding models and vector stores, and integrates with
hundreds of providers. See the [integrations](/docs/integrations/providers/) page for
more.


Select [chat model](/docs/integrations/chat/):Groq▾- [Groq](#)
- [OpenAI](#)
- [Anthropic](#)
- [Azure](#)
- [Google](#)
- [AWS](#)
- [Cohere](#)
- [NVIDIA](#)
- [Fireworks AI](#)
- [Mistral AI](#)
- [Together AI](#)
- [Databricks](#)
```
pip install -qU langchain-groq  

```
```
import getpass  
import os  
  
if not os.environ.get("GROQ_API_KEY"):  
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")  
  
from langchain_groq import ChatGroq  
  
model = ChatGroq(model="llama3-8b-8192")  

```

```
model.invoke("Hello, world!")  

```

note

These docs focus on the Python LangChain library. [Head here](https://js.langchain.com) for docs on the JavaScript LangChain library.

Architecture[​](#architecture "Direct link to Architecture")
------------------------------------------------------------

The LangChain framework consists of multiple open-source libraries. Read more in the
[Architecture](/docs/concepts/architecture/) page.

* **`langchain-core`**: Base abstractions for chat models and other components.
* **Integration packages** (e.g. `langchain-openai`, `langchain-anthropic`, etc.): Important integrations have been split into lightweight packages that are co-maintained by the LangChain team and the integration developers.
* **`langchain`**: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.
* **`langchain-community`**: Third-party integrations that are community maintained.
* **`langgraph`**: Orchestration framework for combining LangChain components into production-ready applications with persistence, streaming, and other key features. See [LangGraph documentation](https://langchain-ai.github.io/langgraph/).

Guides[​](#guides "Direct link to Guides")
------------------------------------------

### [Tutorials](/docs/tutorials/)[​](#tutorials "Direct link to tutorials")

If you're looking to build something specific or are more of a hands-on learner, check out our [tutorials section](/docs/tutorials/).
This is the best place to get started.

These are the best ones to get started with:

* [Build a Simple LLM Application](/docs/tutorials/llm_chain/)
* [Build a Chatbot](/docs/tutorials/chatbot/)
* [Build an Agent](/docs/tutorials/agents/)
* [Introduction to LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

Explore the full list of LangChain tutorials [here](/docs/tutorials/), and check out other [LangGraph tutorials here](https://langchain-ai.github.io/langgraph/tutorials/). To learn more about LangGraph, check out our first LangChain Academy course, *Introduction to LangGraph*, available [here](https://academy.langchain.com/courses/intro-to-langgraph).

### [How-to guides](/docs/how_to/)[​](#how-to-guides "Direct link to how-to-guides")

[Here](/docs/how_to/) you’ll find short answers to “How do I….?” types of questions.
These how-to guides don’t cover topics in depth – you’ll find that material in the [Tutorials](/docs/tutorials/) and the [API Reference](https://python.langchain.com/api_reference/).
However, these guides will help you quickly accomplish common tasks using [chat models](/docs/how_to/#chat-models),
[vector stores](/docs/how_to/#vector-stores), and other common LangChain components.

Check out [LangGraph-specific how-tos here](https://langchain-ai.github.io/langgraph/how-tos/).

### [Conceptual guide](/docs/concepts/)[​](#conceptual-guide "Direct link to conceptual-guide")

Introductions to all the key parts of LangChain you’ll need to know! [Here](/docs/concepts/) you'll find high level explanations of all LangChain concepts.

For a deeper dive into LangGraph concepts, check out [this page](https://langchain-ai.github.io/langgraph/concepts/).

### [Integrations](/docs/integrations/providers/)[​](#integrations "Direct link to integrations")

LangChain is part of a rich ecosystem of tools that integrate with our framework and build on top of it.
If you're looking to get up and running quickly with [chat models](/docs/integrations/chat/), [vector stores](/docs/integrations/vectorstores/),
or other LangChain components from a specific provider, check out our growing list of [integrations](/docs/integrations/providers/).

### [API reference](https://python.langchain.com/api_reference/)[​](#api-reference "Direct link to api-reference")

Head to the reference section for full documentation of all classes and methods in the LangChain Python packages.

Ecosystem[​](#ecosystem "Direct link to Ecosystem")
---------------------------------------------------

### [🦜🛠️ LangSmith](https://docs.smith.langchain.com)[​](#️-langsmith "Direct link to ️-langsmith")

Trace and evaluate your language model applications and intelligent agents to help you move from prototype to production.

### [🦜🕸️ LangGraph](https://langchain-ai.github.io/langgraph)[​](#️-langgraph "Direct link to ️-langgraph")

Build stateful, multi-actor applications with LLMs. Integrates smoothly with LangChain, but can be used without it. LangGraph powers production-grade agents, trusted by Linkedin, Uber, Klarna, GitLab, and many more.

Additional resources[​](#additional-resources "Direct link to Additional resources")
------------------------------------------------------------------------------------

### [Versions](/docs/versions/v0_3/)[​](#versions "Direct link to versions")

See what changed in v0.3, learn how to migrate legacy code, read up on our versioning policies, and more.

### [Security](/docs/security/)[​](#security "Direct link to security")

Read up on [security](/docs/security/) best practices to make sure you're developing safely with LangChain.

### [Contributing](/docs/contributing/)[​](#contributing "Direct link to contributing")

Check out the developer's guide for guidelines on contributing and help getting your dev environment set up.

[Edit this page](https://github.com/langchain-ai/langchain/edit/master/docs/docs/introduction.mdx)

---

#### Was this page helpful?

[NextTutorials](/docs/tutorials/)