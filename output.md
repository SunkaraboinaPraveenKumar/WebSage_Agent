[ Skip to content ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#conceptual-guide>)
**Join us at[ Interrupt: The Agent AI Conference by LangChain](https://langchain-ai.github.io/langgraph/concepts/<https:/interrupt.langchain.com/>) on May 13 & 14 in San Francisco!**
[ ![logo](https://langchain-ai.github.io/langgraph/static/wordmark_dark.svg) ![logo](https://langchain-ai.github.io/langgraph/static/wordmark_light.svg) ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/>)
Concepts 
[ ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/?q=> "Share")
Initializing search 
[ GitHub  ](https://langchain-ai.github.io/langgraph/concepts/<https:/github.com/langchain-ai/langgraph> "Go to repository")
  * [ Home ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/>)
  * [ API reference ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/reference/graphs/>)


[ ![logo](https://langchain-ai.github.io/langgraph/static/wordmark_dark.svg) ![logo](https://langchain-ai.github.io/langgraph/static/wordmark_light.svg) ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/>)
[ GitHub  ](https://langchain-ai.github.io/langgraph/concepts/<https:/github.com/langchain-ai/langgraph> "Go to repository")
  * [ Home  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/>)
Home 
    * Get started  Get started 
      * [ Learn the basics  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/tutorials/introduction/>)
      * [ Deployment  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/tutorials/deployment/>)
    * Guides  Guides 
      * [ How-to Guides  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/how-tos/>)
      * [ Concepts  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/>)
Concepts 
        * [ LangGraph  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts#langgraph>)
        * [ LangGraph Platform  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts#langgraph-platform>)
      * [ Tutorials  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/tutorials/>)
    * Resources  Resources 
      * [ Prebuilt Agents  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/prebuilt/>)
      * [ Adopters  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/adopters/>)
      * [ FAQ  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/faq/>)
      * [ Troubleshooting  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/troubleshooting/errors/>)
      * [ LangGraph Academy Course  ](https://langchain-ai.github.io/langgraph/concepts/<https:/academy.langchain.com/courses/intro-to-langgraph>)
  * [ API reference  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/reference/graphs/>)


Table of contents 
  * [ LangGraph  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#langgraph>)
    * [ High Level  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#high-level>)
    * [ Concepts  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#concepts>)
  * [ LangGraph Platform  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#langgraph-platform>)
    * [ High Level  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#high-level_1>)
    * [ Components  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#components>)
    * [ LangGraph Server  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#langgraph-server>)
    * [ Deployment Options  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#deployment-options>)


  1. [ Home  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/>)
  2. [ Guides  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/how-tos/>)
  3. [ Concepts  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/>)

[ ](https://langchain-ai.github.io/langgraph/concepts/<https:/github.com/langchain-ai/langgraph/edit/main/docs/docs/concepts/index.md> "Edit this page")
# Conceptual Guide[¶](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#conceptual-guide> "Permanent link")
This guide provides explanations of the key concepts behind the LangGraph framework and AI applications more broadly.
We recommend that you go through at least the [Quickstart](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/tutorials/introduction/>) before diving into the conceptual guide. This will provide practical context that will make it easier to understand the concepts discussed here.
The conceptual guide does not cover step-by-step instructions or specific implementation examples — those are found in the [Tutorials](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/tutorials/>) and [How-to guides](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/how-tos/>). For detailed reference material, please see the [API reference](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/reference/>).
## LangGraph[¶](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#langgraph> "Permanent link")
### High Level[¶](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#high-level> "Permanent link")
  * [Why LangGraph?](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/high_level/>): A high-level overview of LangGraph and its goals.


### Concepts[¶](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#concepts> "Permanent link")
  * [LangGraph Glossary](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/low_level/>): LangGraph workflows are designed as graphs, with nodes representing different components and edges representing the flow of information between them. This guide provides an overview of the key concepts associated with LangGraph graph primitives.
  * [Common Agentic Patterns](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/agentic_concepts/>): An agent uses an LLM to pick its own control flow to solve more complex problems! Agents are a key building block in many LLM applications. This guide explains the different types of agent architectures and how they can be used to control the flow of an application.
  * [Multi-Agent Systems](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/multi_agent/>): Complex LLM applications can often be broken down into multiple agents, each responsible for a different part of the application. This guide explains common patterns for building multi-agent systems.
  * [Breakpoints](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/breakpoints/>): Breakpoints allow pausing the execution of a graph at specific points. Breakpoints allow stepping through graph execution for debugging purposes.
  * [Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/human_in_the_loop/>): Explains different ways of integrating human feedback into a LangGraph application.
  * [Time Travel](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/time-travel/>): Time travel allows you to replay past actions in your LangGraph application to explore alternative paths and debug issues.
  * [Persistence](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/persistence/>): LangGraph has a built-in persistence layer, implemented through checkpointers. This persistence layer helps to support powerful capabilities like human-in-the-loop, memory, time travel, and fault-tolerance.
  * [Memory](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/memory/>): Memory in AI applications refers to the ability to process, store, and effectively recall information from past interactions. With memory, your agents can learn from feedback and adapt to users' preferences.
  * [Streaming](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/streaming/>): Streaming is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs.
  * [Functional API](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/functional_api/>): `@entrypoint` and `@task` decorators that allow you to add LangGraph functionality to an existing codebase.
  * [Durable Execution](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/durable_execution/>): LangGraph's built-in [persistence](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/persistence/>) layer provides durable execution for workflows, ensuring that the state of each execution step is saved to a durable store. 
  * [Pregel](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/pregel/>): Pregel is LangGraph's runtime, which is responsible for managing the execution of LangGraph applications.
  * [FAQ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/faq/>): Frequently asked questions about LangGraph.


## LangGraph Platform[¶](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#langgraph-platform> "Permanent link")
LangGraph Platform is a commercial solution for deploying agentic applications in production, built on the open-source LangGraph framework.
The LangGraph Platform offers a few different deployment options described in the [deployment options guide](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/deployment_options/>).
Tip
  * LangGraph is an MIT-licensed open-source library, which we are committed to maintaining and growing for the community.
  * You can always deploy LangGraph applications on your own infrastructure using the open-source LangGraph project without using LangGraph Platform.


### High Level[¶](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#high-level_1> "Permanent link")
  * [Why LangGraph Platform?](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/langgraph_platform/>): The LangGraph platform is an opinionated way to deploy and manage LangGraph applications. This guide provides an overview of the key features and concepts behind LangGraph Platform.
  * [Platform Architecture](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/platform_architecture/>): A high-level overview of the architecture of the LangGraph Platform.
  * [Scalability and Resilience](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/scalability_and_resilience/>): LangGraph Platform is designed to be scalable and resilient. This document explains how the platform achieves this.
  * [Deployment Options](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/deployment_options/>): LangGraph Platform offers four deployment options: [Self-Hosted Lite](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/self_hosted/#self-hosted-lite>), [Self-Hosted Enterprise](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/self_hosted/#self-hosted-enterprise>), [bring your own cloud (BYOC)](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/bring_your_own_cloud/>), and [Cloud SaaS](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/langgraph_cloud/>). This guide explains the differences between these options, and which Plans they are available on.
  * [Plans](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/plans/>): LangGraph Platforms offer three different plans: Developer, Plus, Enterprise. This guide explains the differences between these options, what deployment options are available for each, and how to sign up for each one.
  * [Template Applications](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/template_applications/>): Reference applications designed to help you get started quickly when building with LangGraph.


### Components[¶](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#components> "Permanent link")
The LangGraph Platform comprises several components that work together to support the deployment and management of LangGraph applications:
  * [LangGraph Server](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/langgraph_server/>): The LangGraph Server is designed to support a wide range of agentic application use cases, from background processing to real-time interactions.
  * [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/langgraph_studio/>): LangGraph Studio is a specialized IDE that can connect to a LangGraph Server to enable visualization, interaction, and debugging of the application locally.
  * [LangGraph CLI](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/langgraph_cli/>): LangGraph CLI is a command-line interface that helps to interact with a local LangGraph
  * [Python/JS SDK](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/sdk/>): The Python/JS SDK provides a programmatic way to interact with deployed LangGraph Applications.
  * [Remote Graph](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/how-tos/use-remote-graph/>): A RemoteGraph allows you to interact with any deployed LangGraph application as though it were running locally.


### LangGraph Server[¶](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#langgraph-server> "Permanent link")
  * [Application Structure](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/application_structure/>): A LangGraph application consists of one or more graphs, a LangGraph API Configuration file (`langgraph.json`), a file that specifies dependencies, and environment variables.
  * [Assistants](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/assistants/>): Assistants are a way to save and manage different configurations of your LangGraph applications.
  * [Web-hooks](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/langgraph_server/#webhooks>): Webhooks allow your running LangGraph application to send data to external services on specific events.
  * [Cron Jobs](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/langgraph_server/#cron-jobs>): Cron jobs are a way to schedule tasks to run at specific times in your LangGraph application.
  * [Double Texting](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/double_texting/>): Double texting is a common issue in LLM applications where users may send multiple messages before the graph has finished running. This guide explains how to handle double texting with LangGraph Deploy.
  * [Authentication & Access Control](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/auth/>): Learn about options for authentication and access control when deploying the LangGraph Platform.


### Deployment Options[¶](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#deployment-options> "Permanent link")
  * [Self-Hosted Lite](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/self_hosted/>): A free (up to 1 million nodes executed per year), limited version of LangGraph Platform that you can run locally or in a self-hosted manner
  * [Cloud SaaS](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/langgraph_cloud/>): Hosted as part of LangSmith.
  * [Bring Your Own Cloud](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/bring_your_own_cloud/>): We manage the infrastructure, so you don't have to, but the infrastructure all runs within your cloud.
  * [Self-Hosted Enterprise](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/self_hosted/>): Completely managed by you.

Was this page helpful? 
Thanks for your feedback! 
Thanks for your feedback! Please help us improve this page by adding to the discussion below. 
## Comments
Back to top  [ Previous  Adding nodes as dataset examples in Studio  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/cloud/how-tos/datasets_studio/>) [ Next  Why LangGraph?  ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/high_level/>)
Copyright © 2025 LangChain, Inc | [Consent Preferences](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraph/concepts/#__consent>)
Made with [ Material for MkDocs Insiders ](https://langchain-ai.github.io/langgraph/concepts/<https:/squidfunk.github.io/mkdocs-material/>)
[ ](https://langchain-ai.github.io/langgraph/concepts/<https:/langchain-ai.github.io/langgraphjs/> "langchain-ai.github.io") [ ](https://langchain-ai.github.io/langgraph/concepts/<https:/github.com/langchain-ai/langgraph> "github.com") [ ](https://langchain-ai.github.io/langgraph/concepts/<https:/twitter.com/LangChainAI> "twitter.com")
#### Cookie consent
We use cookies to recognize your repeated visits and preferences, as well as to measure the effectiveness of our documentation and whether users find what they're searching for. **Clicking "Accept" makes our documentation better. Thank you!** ❤️
  * Google Analytics 
  * GitHub 


Accept Reject
