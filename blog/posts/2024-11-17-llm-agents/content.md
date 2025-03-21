# LLM Agents

Over the last few weeks, I have been imparting a full-fledged course on LLMs at my job at oga.ai. It is a fast-paced, deep-dive program consisting of eight lessons of two hours each, and covers much of the current landscape of LLMs in the industry.

The fifth module of the course is about LLM agents, and it has been the most exciting one to prepare and teach, with the seventh about large multimodal models (LMMs) being a close second. Agents truly are a striking application of technology and the results that they can achieve are truly impressive. Even though they are in the beginning phase of development, they commit many errors and are difficult to control, so much so that there are few agents in production environments, they really shine with potential.

The course is in Spanish, and with it being internal to the company I cannot freely share the videos and materials on my own. Still, I want to write about LLM agents in a series of blog posts.

In this first post, I will cover the basics of agents, how they work and how to implement them, along with a few examples. It will be fairly technical at the beginning, but once the basics are covered we can go on to the practical part. In the second, I showcase the development tools used for LLM agents as well as many examples from the industry and the academia. 

In the following couple of posts, I plan to program and explain in depth one or two agents. I have already implemented a ReAct agent in pure Python (we will see in this post what these are), and I will probably implement a more complex one using a framework like LangChain or AutoGen.

Finally, I want to use an agent to expand my [Torch-Tracer Project](https://mvazquez.ai/blog/output/2024-11-12-torch-tracing-01/content.html), with the objective that it needs as little human input as possible.

# What are agents?

Artificial Intelligence: A Modern Approach defines an **"agent"** as[^1]

> _"Anything that can be viewed as perceiving its environment through sensors and acting upon that environment through actuators"_

This definition gives a general idea of what agents do, but it is way too broad, since a simple thermostat has sensors and actuators. A smarter definition is the one of a **"rational agent"** [^1]

> _"For each possible percept sequence, a rational agent should select an action that is expected to maximize its performance measure, given the evidence provided by the percept sequence and whatever built-in knowledge the agent has."_

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/simple_agents.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Agent Definitions.</p>

This means that a rational agent will try to accomplish an objective as defined by the performance measure. This is what we want, to tell our agent to do something and that it tries, to the best of its abilities, to follow the orders. Here is where the LLM part comes into play.

A **"LLM agent"** is one that uses a LLM as its _brain_ to reason. The central idea is that they use a language model to choose what **actions** they take to **accomplish an objective** given a current **state** or environment.

If you are familiar with prompt chains, as the ones used in LangChain and in Retrieval Augmented Generation (RAG), they have many things in common with agents, but agents are more flexible and can show more complex behavior as it chooses what action to take at each moment by itself, while in chaining the possible workflow of the actions is fixed, rigidly specified in code.

### Agentic workflows

An analogy that I find illustrative to understand LLM agents can be made with the writing of an essay. This analogy comes from Andrew Ng[^2].

In a regular LLM workflow you ask the LLM to write an essay about some topic X. Since they are autoregressive models, they will write it directly from the beginning to the end without ever going back to fix errors or improve any section, or stopping to reflect and research more about topic X. This task would be very difficult for most humans, and yet LLMs are surprisingly good at it.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/regular_llm_workflow.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Regular LLM workflow to write an essay. </p>

In an agentic workflow, you remove all those restrictions. The agent will be able to reason to take actions to better write the essay. It could decide to start by specifying the essay's structure and researching the topic in an external data storage (databases, documents, the internet...). Then it may decide to write a first draft and iteratively improve it until it is finished. At any point in the process it can reflect on what the best action to take among the set of possible actions, which gives it all these capabilities. Obviously, this workflow will often give better results than the first direct approach.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/agentic_llm_workflow.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Agentic LLM workflows allow to take to choose what actions to take to get the best results. </p>

LLM agents are based on the chain-of-thought. By dividing a complex problem into simpler subproblems, it can solve them sequentially to reach the final answer.

1. **Plan** what action to take to get closer to its objective.
2. Perform an **action** and **observe** its consequences.
3. Iterate until reaching the objective _"LLM in a loop"_

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/agent_cot_loop.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Agentic LLM workflows as loop: Plan, Act, Observe. </p>

As a simple example, consider the common case of a software developer that uses ChatGPT to write a code program. They start by stating the problem and asking the model to "think step by step". The agent **plans** the actions it should take. Then it **writes** the code. The developer copies that code, pastes it into the script and runs the program. If it fails, they paste the **error trace** to ChatGPT to fix it, and if it **works** then the task is finished.

If you automate this in a loop, it becomes a simple agentic workflow, where the words in bold font from the previous paragraph correspond to planning, acting and observing.

All the text that is generated by the model during a task is called the **reasoning trace**.

### ReAct agents

You might have noticed that the planning step is not strictly necessary. An agent could just observe the environment and act, and in fact the first agents based on LLMs did just that, but they did not work too well. In 2022 in the ReAct paper[^3] Yao, Shunyu, et al. introduced the Reason + Act framework and showed that it works better than just acting.

The setup is an agent with access to three different actions that leverage a "simple Wikipedia web API: (1)**search**\[entity] returns the first 5 sentences from the corresponding _entity_ wiki page if it exists, or else suggests top-5 similar entities from the Wikipedia search engine, (2)**lookup**\[string], which returns the next sentence in the page containing _string_ simulating a ctrl+F command, and (3)**finish**\[answer] which would finish the current task with _answer_."[^3]

With this environment they compare four different approaches: standard zero-shot, chain of thought prompting, act-only agent and Reason + Act agent. The following example from the paper shows how they try to solve a question about the Apple Remote device. Let's review the first three approaches first.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/react_01.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Example of standard zero-shot, chain of thought prompting, act-only from the ReAct paper. </p>

In (1a) zero-shot the LLM just answers directly and gets it wrong. With (1b) chain-of-thought the LLM is prompted to "think step by step before answering", a technique that improves accuracy of language models[^4], but still gets it wrong. In (1c) we have a simple agentic workflow that acts and observes, and allows to use the Wikipedia tools. This time it actually gets close to the answer, but ends up returning "yes" as its final answer. The problem with this approach is that the model cannot reflect on what tool to use, how to use it or plan how to get the final answer. The only possibility is to act, stating the action and its argument. ReAct was created to address this problem.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/react_02.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Example of a ReAct agent from the ReAct paper. In this case it manages to get the right answer.</p>

In this last case the agent follows a loop of reason-act-observe that overcomes the previously stated limitations, and it actually gets the correct answer: "keyboard function keys". This example showcases how the model is able to plan and reason about the result of its actions. This is a simple yet extremely powerful workflow, and most state-of-the-art agents follow it, with improvements in the reasoning step and an increase in freedom to act. It leverages the powerful large language models by using them as the "brain" of the agent.

### Actions as tools

To implement agents we need to define a **set of possible actions for the agent to take**, among which the agent will have to decide in each iteration. For example it could have access to the following:

-   Ask the user for information.
-   Search the web.
-   Using an external database.
-   Using a calculator or symbolic programming.
-   Using a Python code interpreter.

These possible actions are commonly referred to as **tools**, and a set of actions is a **toolbox**.

As an example, ChatGPT has access to three different tools.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/chatgpt_tools.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">The GPT-4 model from the ChatGPT web UI has access to web browsing, DALL·E image generator, and code interpreter. </p>

At the time of writing GPT-4 has the knowledge cutoff date of October 2024. That means that the pretraining has data until that date, and it knows nothing that happened thereafter. If I ask it about events after that date, it will use a web search tool to retrieve the necessary information.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/chatgpt_web_search.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">GPT does not know the democratic candidate of 2024, so it uses web search tool to answer . </p>

In this [conversation](https://chatgpt.com/share/e/6734e362-6720-800a-ad98-0fe320703b3a) I make ChatGPT use the code interpreter tool to generate a plot to showcase it. As of the moment I am writing this post, it is not possible to share conversations in which DALL·E is used to generate images, but you can guess how it works: you ask ChatGPT to generate an image of a puppy and it decides to call DALL·E, writing the image prompt by itself.

Another example is the [LangChain tools](https://Python.LangChain.com/docs/integrations/tools/). These are implemented in the LangChain library to be used by language models, and there is a great number and variety of them: several web search providers and code interpreters, a few productivity tools like GitHub, Jira, or Gmail; tools to access databases and even more.

## Agent Showcase

Let's proceed with a full agent workflow as an example. In this case we have an agent, let's call him JARVIS, that assists the user with data queries.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/workflow_example.webp" width="90%" />
</p>

<p style="text-align:center; font-style: italic;">Jarvis helps the user to get the total number of customers in the database. </p>

When the user asks JARVIS to find out how many clients are in the database, the agent has to figure out the best course of action to get the information. Let’s walk through the sequence step by step to see how JARVIS accomplishes this seemingly simple task:

### Breaking Down the Workflow

The user starts by asking, **"Hey JARVIS, how many clients are there in our database?"** This is the initial input that sets the whole process in motion. Even though this question seems straightforward, there are several steps involved before reaching the final answer.

1. **Understanding the Query**:

    - JARVIS recognizes that it needs to interact with a database to fulfill the user’s request. The initial plan involves listing out the tools available to it, which include accessing tables and querying information.

2. **Exploring the Environment**:

    - To proceed, JARVIS needs to understand what data it has access to. It starts by using a tool to **list all the tables** in the database. The agent observes that there are tables named `AGENTS`, `CUSTOMERS`, `ORDERS`, among others.

3. **Focusing on Relevant Information**:

    - Since the user is asking specifically about clients, JARVIS infers that the relevant information should be in the `CUSTOMERS` table. However, before it can query this table, it needs to understand its structure.

4. **Querying the Table Schema**:

    - JARVIS retrieves the schema of the `CUSTOMERS` table to see what fields are available. It finds that the table includes columns like `CLIENT_ID` and `CLIENT_NAME`.

5. **Formulating a Plan to Extract Information**:

    - Now that JARVIS knows the structure of the table, it formulates a plan to count the entries. The goal is to determine how many rows (i.e., clients) are present in the table.

6. **Executing the SQL Query**:

    - JARVIS constructs a simple SQL query:

        ```sql
        SELECT COUNT(*) FROM CUSTOMERS
        ```

        This query will return the total number of rows in the `CUSTOMERS` table, which corresponds to the number of customers.

7. **Interpreting the Results**:

    - The query is executed, and JARVIS receives the result: `[(25,)]`, indicating there are 25 customers in the database.

8. **Delivering the Final Answer**:

    - With the result in hand, JARVIS returns to the user with the final answer:
      **"There are 25 clients in the database."**

### Key Takeaways from This Example

This workflow showcases a **classic agentic pattern** where JARVIS uses a loop of **planning, acting, and observing**:

-   **Planning**: At multiple steps, JARVIS formulates a plan to achieve the desired outcome. It doesn’t jump straight to querying the database without first understanding the environment.
-   **Acting**: It uses tools effectively to explore the environment, fetch the schema, and run the SQL query.
-   **Observing**: After each action, it observes the output to decide on the next step.

The diagram above reflects how even seemingly simple tasks require agents to break down problems into smaller actions, reflect on the information available, and decide on the best next step. The flexibility of this approach is what makes LLM agents so powerful.

### Reasoning Trace

Through all this process the LLM generates text that is recursively added to the prompt. This generated text is the reasoning trace.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/reasoning_trace.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Reasoning trace generated in the example agentic workflow.</p>

There are only two ways for a language model to access information: weight updates and prompts (in context learning). Since we are only using inference during an agentic task, this means all information about the conversation with the user and the current state that is needed to accomplish the objective must be passed through the prompt for every call. This makes prompt management a crucial aspect of agents.

The simplest approach to accomplish this is to paste all the user interaction and the reasoning trace for every call to the model. This works well for simple tasks that do not generate much text, that does not need access to large quantities of external data and that do not depend on previous interactions with the same or other users. For other tasks a more complex and customized prompt management strategy must be implemented. Through this post many agent design patterns that can be useful will be explained.

### Code Implementation

I will now show a simple implementation of the example using LangChain. I will use the OpenAI api for the language model.

First, we build a sample database.

```Python
# 01_create_and_fill_database.py
import sqlite3
import os

# File path
database_file_path = "./sql_lite_database.db"

# Check if database file exists and delete if it does
if os.path.exists(database_file_path):
    os.remove(database_file_path)
    message = "File 'sql_lite_database.db' found and deleted."
else:
    message = "File 'sql_lite_database.db' does not exist."

# Step 1: Connect to the database or create it if it doesn't exist
conn = sqlite3.connect(database_file_path)

# Step 2: Create a cursor
cursor = conn.cursor()

# Step 3: Create tables
create_table_query1 = """
                        CREATE TABLE IF NOT EXISTS   "AGENTS"
                        (
                            "AGENT_CODE" CHAR(6) NOT NULL PRIMARY KEY,
                            "AGENT_NAME" CHAR(40),
                            "WORKING_AREA" CHAR(35),
                            "COMMISSION" NUMBER(10,2),
                            "PHONE_NO" CHAR(15),
                            "COUNTRY" VARCHAR2(25)
                            );
                        """
create_table_query2 = """
                        CREATE TABLE IF NOT EXISTS   "CUSTOMER"
                        ( "CUST_CODE" VARCHAR2(6) NOT NULL PRIMARY KEY,
                            "CUST_NAME" VARCHAR2(40) NOT NULL,
                            "CUST_CITY" CHAR(35),
                            "WORKING_AREA" VARCHAR2(35) NOT NULL,
                            "CUST_COUNTRY" VARCHAR2(20) NOT NULL,
                            "GRADE" NUMBER,
                            "OPENING_AMT" NUMBER(12,2) NOT NULL,
                            "RECEIVE_AMT" NUMBER(12,2) NOT NULL,
                            "PAYMENT_AMT" NUMBER(12,2) NOT NULL,
                            "OUTSTANDING_AMT" NUMBER(12,2) NOT NULL,
                            "PHONE_NO" VARCHAR2(17) NOT NULL,
                            "AGENT_CODE" CHAR(6) NOT NULL REFERENCES AGENTS
                        );
                        """

create_table_query3 = """
                        CREATE TABLE IF NOT EXISTS "ORDERS"
                        (
                            "ORD_NUM" NUMBER(6,0) NOT NULL PRIMARY KEY,
                            "ORD_AMOUNT" NUMBER(12,2) NOT NULL,
                            "ADVANCE_AMOUNT" NUMBER(12,2) NOT NULL,
                            "ORD_DATE" DATE NOT NULL,
                            "CUST_CODE" VARCHAR2(6) NOT NULL REFERENCES CUSTOMER,
                            "AGENT_CODE" CHAR(6) NOT NULL REFERENCES AGENTS,
                            "ORD_DESCRIPTION" VARCHAR2(60) NOT NULL
                        );
                        """

queries = [create_table_query1, create_table_query2, create_table_query3]
# queries = [create_table_query1, create_table_query2]

for query in queries:
    # execute queries
    cursor.execute(query)

# Step 4: Insert data into tables Agents, Orders and Customers
# For space reasons I will omit most of the rows.
insert_query = """
INSERT INTO AGENTS VALUES ('A007', 'Ramasundar', 'Bangalore', '0.15', '077-25814763', '');
INSERT INTO AGENTS VALUES ('A003', 'Alex ', 'London', '0.13', '075-12458969', '');
...

INSERT INTO CUSTOMER VALUES (
  'C00013', 'Holmes', 'London', 'London', 'UK', '2', '6000.00', '5000.00', '7000.00', '4000.00', 'BBBBBBB', 'A003'
);
INSERT INTO CUSTOMER VALUES (
  'C00001', 'Micheal', 'New York', 'New York', 'USA', '2', '3000.00', '5000.00', '2000.00', '6000.00', 'CCCCCCC', 'A008'
);
...

INSERT INTO ORDERS VALUES('200100', '1000.00', '600.00', '2024-08-01', 'C00013', 'A003', 'SOD');
INSERT INTO ORDERS VALUES('200110', '3000.00', '500.00', '2024-04-15', 'C00019', 'A010', 'SOD');
...
"""

for row in insert_query.splitlines():
    try:
        cursor.execute(row)
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        print(row)

# Step 5: Fetch data from tables
list_of_queries = []
list_of_queries.append("SELECT * FROM AGENTS")
list_of_queries.append("SELECT * FROM CUSTOMER")
list_of_queries.append("SELECT * FROM ORDERS")

# execute queries
for query in list_of_queries:
    cursor.execute(query)
    data = cursor.fetchall()

    print(f"--- Data from tables ({query}) ---")
    for row in data:
        print(row)

# Step 7: Close the cursor and connection
cursor.close()
conn.commit()
conn.close()
```

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/tablas.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Sample database schema.</p>

Now let's implement a simple agent using LangChain. First we need to import the necessary libraries and set up our database connection and language model:

```Python
from LangChain.utilities import SQLDatabase
from LangChain.agents.agent_types import AgentType
from LangChain.agents.agent_toolkits import SQLDatabaseToolkit
from LangChain.agents import create_sql_agent
from LangChain_community.llms.openai import OpenAI

# define the database we want to use for our test
db = SQLDatabase.from_uri("sqlite:///sql_lite_database.db")

# choose llm model, in this case the default OpenAI model
llm = OpenAI(
    temperature=0,
    verbose=True,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
```

With our database and language model ready, we can create the agent. We'll use LangChain's SQL toolkit and the ReAct agent type:

```Python
# setup agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# define the user's question
question = "How many customers do we have in our database?"
agent_executor.invoke(question)
```

The output shows the agent's reasoning trace as it works through the problem. Let's analyze what's happening:

1. First, it lists the available tables using sql_db_list_tables
2. Then it examines the schema of the CUSTOMER table using sql_db_schema
3. Finally, it executes a simple COUNT query using sql_db_query

The agent concludes that there are 25 customers in the database.

```Python
> Entering new SQL Agent Executor chain...
Action: sql_db_list_tables
Action Input: AGENTS, CUSTOMER, ORDERS I should query the schema of the CUSTOMER table to see how many customers are in the database.
Action: sql_db_schema
Action Input: CUSTOMER
CREATE TABLE "CUSTOMER" (
	"CUST_CODE" TEXT(6) NOT NULL,
	"CUST_NAME" TEXT(40) NOT NULL,
	"CUST_CITY" CHAR(35),
	"WORKING_AREA" TEXT(35) NOT NULL,
	"CUST_COUNTRY" TEXT(20) NOT NULL,
	"GRADE" NUMERIC,
	"OPENING_AMT" NUMERIC(12, 2) NOT NULL,
	"RECEIVE_AMT" NUMERIC(12, 2) NOT NULL,
	"PAYMENT_AMT" NUMERIC(12, 2) NOT NULL,
	"OUTSTANDING_AMT" NUMERIC(12, 2) NOT NULL,
	"PHONE_NO" TEXT(17) NOT NULL,
	"AGENT_CODE" CHAR(6) NOT NULL,
	PRIMARY KEY ("CUST_CODE"),
	FOREIGN KEY("AGENT_CODE") REFERENCES "AGENTS" ("AGENT_CODE")
)

/*
3 rows from CUSTOMER table:
CUST_CODE	CUST_NAME	CUST_CITY	WORKING_AREA	CUST_COUNTRY	GRADE	OPENING_AMT	RECEIVE_AMT	PAYMENT_AMT	OUTSTANDING_AMT	PHONE_NO	AGENT_CODE
C00013	Holmes	London	London	UK	2.0000000000	6000.00	5000.00	7000.00	4000.00	BBBBBBB	A003
C00001	Micheal	New York	New York	USA	2.0000000000	3000.00	5000.00	2000.00	6000.00	CCCCCCC	A008
C00020	Albert	New York	New York	USA	3.0000000000	5000.00	7000.00	6000.00	6000.00	BBBBSBB	A008
*/ I should query the CUSTOMER table and count the number of rows to get the total number of customers.
Action: sql_db_query
Action Input: SELECT COUNT(*) FROM CUSTOMER[(25,)] I now know the final answer.
Final Answer: 25
```

This example demonstrates how the agent breaks down the problem into logical steps and uses the available tools to reach the correct answer, following the ReAct pattern we discussed earlier.

Here LangChain does most of the work for us with the `create_sql_agent` function, which allows us to have our ReAct agent in a few lines of code. In the next blog post, I will implement a similar agent in Python, since this post is already getting long.

### AgentGPT

As a second showcase of an agent I want to mention [AgentGPT](https://agentgpt.reworkd.ai/). I recommend you create a free account and give it some task. For example, ask it to parse the data of the current season of the Spanish football league's first division and export it in a csv file. It will search the web for the appropriate data, initialize a Python environment with some libraries, write a web scraping script and run it, and finally return the .csv file to the user. From a free account it will run out of iterations before achieving it, but it still is a good showcase of what a simple agent is able to do.

## Why use agents?

By this point I hope to have delivered an initial idea of what agents are. If you are still not convinced of their power, by the end of this post you will be. For now I want to clearly explain some of their best attributes in this section.

The first thing to understand is that agents can _augment_ anything a LLM already does. In any task, you can improve the zero-shot performance by implementing an agentic workflow. For example these are the best scores achieved by GPT-3.5 and GPT-4 in HumanEval[^5], a coding benchmark. Their 48.1 and 67 pass@1 original scores increase hugely by using agents, with the best implementation of GPT-4 reaching close to a 100% pass.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/humaneval_agents.webp" width="100%" />
</p>

<p style="text-align:center; font-style: italic;">Humaneval score comparison of zero-shot LLMs vs agentic implementations.</p>

Other advantages of agents are:

-   They are highly autonomous.
-   They are able to recover from errors.
-   They can perform complex workflows without having to explicitly program them like you would in prompt chaining.

For example, let's consider a SQL RAG system that:

1. Takes a natural language query as input.
2. Transforms the query into SQL.
3. Retrieves the results of the query from the database.
4. Communicates results back to the user.

This is a RAG workflow that works for simple tasks, but what happens if the SQL query returns an error? Or the retrieved data is different from what was expected? Or if the user's query is complex and needs to query several tables to return the correct answer?

These limitations of RAG systems are effectively addressed by agents.

## Memory in Agents

The last basic component of agents that we need to talk about is **memory**. Earlier in the post I teased the question when talking about the reasoning trace. Large Language Models do not have _memory_ of past interactions: all information for a call must be passed through the prompt.

So how do we implement the memory in our agent? The truth is that there is not an established solution yet, and it depends heavily on the system that is being built.

What we do have are some design patterns that have recently emerged as more advanced agentic applications are being built. In a recent paper about cognitive architectures for language agents[^6], Sumers, Theodore R., et al propose several memory patterns that are becoming standard in the field. They are analogies to different classes of human memory, as studied by psychologists.

### Types of Memory

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/memory_types.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Memory types found in humans and agents. From the CoALA[^6] paper.</p>

**Procedural Memory** represents the agent's core capabilities encoded in its model weights and implementation code. Just like humans don't consciously think about the mechanics of riding a bike, agents leverage their pre-trained knowledge and coded functions automatically. This includes the language model's understanding of syntax, reasoning patterns, and the defined tools and functions the agent can use. The procedural memory is typically fixed during inference, only changing through model updates or code modifications.

**Semantic Memory** acts as the agent's knowledge base, implemented through external data sources like vector stores, graph databases, or traditional SQL databases. This allows agents to access and reference factual information beyond their training data, similar to how humans draw upon learned knowledge from education and experience. By connecting to these data stores, agents can query relevant information, verify facts, and ground their reasoning in accurate, up-to-date data rather than relying solely on their pre-trained knowledge.

**Episodic Memory** maintains a record of the agent's past experiences and interactions, which can include conversation history, previous task attempts, or user preferences. This memory type helps agents maintain context across multiple interactions and learn from past successes or failures. For example, an agent might remember a user's preferred format for data visualization from earlier conversations, or recall specific approaches that worked well for similar tasks in the past. This can be implemented through conversation logs, task histories, or specialized databases tracking agent-user interactions.

**Working Memory** is the agent's active computational space, primarily manifested in the reasoning trace and the immediate context window of the language model. Like a human's short-term memory holding information for immediate use, working memory contains the current task state, recent observations, and immediate plans. This is typically implemented through prompt engineering and context management, carefully balancing the amount of information kept in the immediate context to avoid overwhelming the model while maintaining task coherence.

### Memory Updates

Memory updates can be performed at different frequencies, each with its own tradeoffs between knowledge freshness and system performance.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/memory_updates.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Two methods to update the memory state.</p>

**In the hot path** updates occur in real-time during each agent loop iteration. This ensures the agent always works with the latest information but introduces latency overhead that can impact response times, particularly in conversational applications.

**In the background** performs updates asynchronously at scheduled intervals. This approach maintains system responsiveness by avoiding update-related delays, though the agent may occasionally work with slightly outdated information.

## Schema of an Agent

By now, we've explored how agents can leverage planning, acting, and observing to solve complex tasks iteratively. However, an agent's full potential lies in integrating these components seamlessly within a cohesive architecture. Let’s revisit the diagram we’ve been building towards.

<p align="center"> <img src="../../media/2024-11-17-llm-agents/agent_schema.webp" width="100%" /> </p> 
<p style="text-align:center; font-style: italic;">General schema of an LLM Agent workflow.</p>
As shown above, the core architecture revolves around four primary components: Memory, Planning, Tools, and Action. Each part plays a crucial role in enabling the agent to operate autonomously:


**Planning** involves breaking down high-level objectives into actionable steps. Here, the agent employs reasoning techniques like reflection, self-critique, and objective decomposition to optimize its approach. By continuously evaluating its progress through chain-of-thought processes, the agent can refine its actions and adapt to changing circumstances or new information. This enables it to move beyond rigid workflows, making it more resilient in real-world scenarios where uncertainty is the norm.

**Tools** extend the agent's capabilities beyond text generation, granting it access to specialized functions like database queries, web searches, or even code execution. This is where LLM agents distinguish themselves from traditional LLM applications—they can dynamically interact with their environment to gather new data, calculate results, or even automate tasks. The toolbox concept allows for modularity, where new tools can be added or swapped out as the agent's needs evolve.

**Memory** serves as the knowledge backbone of the agent. By leveraging both long-term and short-term memory, agents can recall past interactions, user preferences, and contextual information to maintain coherence across sessions. This is akin to the way humans draw upon their experiences and knowledge when solving new problems. For instance, episodic memory might store a detailed conversation history, while semantic memory allows the agent to access databases or other factual sources dynamically.

Finally, **Action** is where plans come to fruition. Here, the agent executes the chosen actions, whether it's retrieving data, generating responses, or invoking external tools. By observing the results of its actions, it learns iteratively, adjusting its strategy in the next cycle if needed. This feedback loop—Plan, Act, Observe—is crucial for agents to handle complex, open-ended tasks effectively.

# Agent Design Patterns

LLM agents are an emerging technology still in its early stages. While many companies and talented developers are actively exploring applications powered by agents, these projects are largely still in their infancy. This is evident from the limited number of agents currently deployed in production environments. However, the remarkable potential and versatility of agents have sparked rapid development, leading to the emergence of several innovative design patterns. In this section, I’ll introduce some of the most notable ones shaping the future of agent-based systems.

## Reflection

The concept of reflection in LLM agents is centered around enabling the agent to iteratively evaluate and improve its own output. Think of it as an agent working towards a goal while continuously critiquing itself until the desired outcome is achieved.

Imagine you prompt a coding agent to write a function to accomplish a specific task. Initially, the agent drafts a solution and then immediately shifts to a self-evaluation mode. In this mode, it reviews the code it just generated, checking for correctness, efficiency, and coding style. If the agent identifies any issues—be it logical errors, inefficiencies, or stylistic inconsistencies—it provides feedback and attempts to improve the function.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/reflection.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Example of reflection in an agent.</p>

This iterative loop of generation, reflection, and revision continues until the agent is confident that the final result meets all specified requirements. By incorporating this self-critical process, agents can significantly improve their outputs, reducing errors and producing more robust solutions.

## Tool Use

We’ve previously discussed the concept of tools in LLM agents, enabling models to interact with external resources like databases, APIs, or search engines to extend their capabilities. Now, let's explore what happens when this concept is pushed to its limits through the Gorilla[^7] model.

**Gorilla** is a fine-tuned LLaMA-based model specifically designed to excel at generating precise API calls. This system surpasses even GPT-4 in the accuracy of writing API invocations, demonstrating that specialized finetuning can lead to significant performance improvements over general-purpose LLMs.

The core idea behind Gorilla is integrating **document retrieval**. By combining the model with a retrieval mechanism, Gorilla can dynamically access the latest API documentation at test time, ensuring that the generated API calls are not only accurate but also aligned with the most current information.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/gorilla_tool_use.webp" width="100%" />
</p>

<p style="text-align:center; font-style: italic;">Image from Gorilla: Gorilla model significantly outperforms other LLMs in accuracy while reducing hallucinations.</p>

#### How Gorilla Works

The process begins with curating a dataset comprising **1,645 API calls** from popular sources like HuggingFace, Torch Hub, and TensorFlow Hub. Using this extensive dataset, the team generated **16,450 (instruction, API) pairs** through self-instruct techniques. This set of examples was then used to fine-tune a LLaMA-based model, resulting in Gorilla-7B.

When a user submits a natural language query (e.g., “Generate an image of dancing cats”), Gorilla first retrieves relevant documentation from its **API Database** using an information retriever. The retrieved context helps the model to understand which API is best suited for the task and how to properly call it. This retrieval step significantly reduces hallucinations by grounding the model's responses in real, authoritative documentation.

The result is a system that can adapt to API changes and updates. As demonstrated in the example above, Gorilla can accurately generate API calls like `StableDiffusionPipeline.from_pretrained()` to achieve its objective.

You can explore a live demo of Gorilla [here](https://gorilla.cs.berkeley.edu/leaderboard.html#api-explorer) and experiment with it yourself using their [Colab notebook](https://colab.research.google.com/drive/1DEBPsccVLF_aUnmD0FwPeHFrtdC0QIUP?usp=sharing).

## Planning

To solve complex tasks with agents it is fundamental that they are able to plan a series of simpler steps. Planning capabilities in LLM agents can range from simple task decomposition to sophisticated hierarchical planning systems. More advanced approaches like tree-of-thoughts extend chain-of-thought prompting by exploring multiple reasoning paths simultaneously, evaluating them, and selecting the most promising one to pursue. Some agents even employ meta-planning, where they not only plan the steps to solve a task but also strategize about how to plan effectively, considering factors like resource constraints and potential failure points.

An illustrative example is the HuggingGPT paper[^8], which uses planning in order to leverage other machine learning models found in web communities (e.g. Hugging Face) to solve AI tasks. Specifically, they use ChatGPT to conduct task planning when receiving a user request, select models according to their function descriptions available in Hugging Face, execute each subtask with the selected AI model, and summarize the response according to the execution results.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/agent_planning_hugging_gpt.webp" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Overview of HuggingGPT. With an LLM (e.g., ChatGPT) as the core controller and the expert models as the executors, the workflow of HuggingGPT consists of four stages: 1) Task planning: LLM parses the user request into a task list and determines the execution order and resource dependencies among tasks; 2) Model selection: LLM assigns appropriate models to tasks based on the description of expert models on Hugging Face; 3) Task execution: Expert models on hybrid endpoints execute the assigned tasks; 4) Response generation: LLM integrates the inference results of experts and generates a summary of workflow logs to respond to the user. Image from [^8]</p>

## Multi-Agent Systems

Some complex use cases might be too much for a single agent to solve. In multi-agent systems a set of agents interact with each other and with the environment to achieve the objectives, through a collaborative, competitive, or hybrid effort.

These systems can be as simple as two agents performing the same tasks as a single agent divided between them, which simplifies the prompt and state management as well as allowing for a more intuitive system design with specified roles and tasks, or as complex as a multitudinous swarm of agents with complex and differentiated behaviors which can show emergence of new properties.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/multi_agents_critic.webp" width="80%" />
</p>
<p style="text-align:center; font-style: italic;">Simple multi-agentic system that replicates the reflection example, with two agents with differentiated roles.</p>

In _ChatDev: Communicative Agents for Software Development_[^9] Qian, Chen, et al. develop "a chat-powered software development framework in which specialized agents driven by large language models (LLMs) are guided in what to communicate (via chat chain) and how to communicate (via communicative dehallucination). These agents actively contribute to the design, coding, and testing phases through unified language-based communication, with solutions derived from their multi-turn dialogues. We found their utilization of natural language is advantageous for system design, and communicating in programming language proves helpful in debugging. This paradigm demonstrates how linguistic communication facilitates multi-agent collaboration, establishing language as a unifying bridge for autonomous task-solving among LLM agents."

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/chatdev.webp" width="80%" />
</p>
<p style="text-align:center; font-style: italic;">ChatDev, a chat-powered software development framework, integrates LLM agents with various social roles, working autonomously to develop comprehensive solutions via multi-agent collaboration. </p>

A cool demonstration video of ChatDev can be found in their [GitHub repository](https://github.com/OpenBMB/ChatDev?tab=readme-ov-file#-what-can-chatdev-do).

In "Improving factuality and reasoning in language models through multiagent debate."[^10] Du, Yilun, et al. evaluate the performance of multi-agent systems in a diverse set of tasks and compare them with single-model solutions, and they show that the former consistently outperform the latter.

<p align="center">
  <img src="../../media/2024-11-17-llm-agents/multi_agents_evaluation.webp" width="80%" />
</p>
<p style="text-align:center; font-style: italic;">Accuracy of traditional inference and our multi-agent debate over six benchmarks. </p>

Overall, multi-agent systems demonstrate superior performance compared to single agents across various tasks. However, this improved capability comes with some trade-offs: increased latency due to multiple agent interactions, higher computational and API costs from running multiple models, and greater system complexity. These factors must be weighed when deciding between single and multi-agent architectures for a specific application.

Multi-agent systems can be classified in many ways:

-   **Cooperative**: they collaborate with each other. E.g. chatdev's team working together on a software project.

-   **Competitive**: they have conflicting objectives. E.g. stock brokers that compete to have the best investment portfolio, each trying to outperform the others.

-   **Hybrid**: sometimes they collaborate, other times they compete. E.g. a group of researchers sharing data but competing for funding.

-   **Centralized/orchestrated**: one agent controls and organizes the others. E.g. a project manager coordinating tasks among team members.

-   **Decentralized**: they make decisions locally without a central controller. E.g. a group of open-source contributors working independently on different features.

-   **Hierarchical**: some agents have greater authority or coordination capabilities over others. E.g. a corporate structure where managers oversee teams of employees.

-   **Homogeneous**: all agents are the same. E.g. a swarm of drones programmed to perform identical tasks.

-   **Heterogeneous**: there are differences in roles, tools, etc. E.g. a team of engineers, designers, and marketers each using their specialized skills and tools.

## Execution Termination

In autonomous workflows, it's crucial to clearly define termination conditions, as if they are too lax the loops could keep running and wasting resources, and if they are too strict the loops could be terminated before enough effort has been put into the task. There are several approaches to determine when an agent should stop executing:

### Parameter-Based Limits

The simplest approach is setting hard limits on execution parameters:

-   Maximum number of tokens generated
-   Maximum number of turns or iterations
-   Maximum number of actions performed
-   Time-based limits
-   Cost/budget constraints

These limits act as safety guards to prevent infinite loops or excessive resource consumption.

### Agent-Determined Completion

A more flexible approach lets the agent determine when to finish:

-   Self-evaluation: The agent assesses if it has achieved its objective.
-   External validator: A separate agent reviews and validates the completion.
-   Confidence threshold: The agent continues until reaching a certain confidence level.
-   Failure recognition: The agent can determine if the task is impossible and gracefully terminate.

### Problem-Specific Conditions

Some tasks have natural completion criteria based on their domain:

-   A development agent finishes when all automated tests pass.
-   A data scientist agent stops when reaching a specified accuracy threshold.
-   A search agent terminates upon finding the target information.
-   A game-playing agent concludes when winning or losing.

### Hybrid Approaches

In practice, most systems combine multiple termination conditions:

```Python
def agent_execution():
    max_iterations = 100
    while iterations < max_iterations:
        if agent.objective_achieved():
            return SUCCESS
        if agent.is_impossible():
            return FAILURE
        if cost > budget:
            return BUDGET_EXCEEDED

        agent.execute_next_action()
        iterations += 1

    return MAX_ITERATIONS_REACHED
```

This ensures both task completion and system safety while maintaining flexibility in the execution flow.

### Human-in-the-Loop

While agents can operate autonomously, there are scenarios where human intervention is necessary or beneficial. Human-in-the-loop (HITL) is a design pattern where human operators are integrated into the agent's workflow as a special type of tool.

Some key scenarios where HITL is valuable:

-   **Critical Decisions**: When actions have significant consequences (e.g., financial transactions, medical decisions)
-   **Verification**: Validating agent outputs before implementation
-   **Guidance**: Providing additional context or clarification when the agent is uncertain
-   **Learning**: Using human feedback to improve agent performance
-   **Safety**: Acting as a safeguard against potential harmful actions

Humans can be thought of as an AGI tool - they have general problem-solving capabilities and can provide input that helps agents overcome their limitations.

#### Implementation Approaches

There are several ways to implement HITL in agent systems:

```Python
# Example of a simple human-in-the-loop tool
class HumanInputTool:
    def run(self, query: str) -> str:
        """Get input from human operator"""
        print(f"\nHuman input needed: {query}")
        return input("Your response: ")

# Example usage in an agent workflow
if confidence_score < THRESHOLD:
    human_input = human_tool.run(
        "I'm uncertain about this step. Should I proceed with action X?"
    )
    if human_input.lower() != 'yes':
        return alternative_action()
```

LangChain provides an already implemented human tool.

```Python
from LangChain.agents import AgentType, initialize_agent, load_tools
from LangChain_openai import ChatOpenAI, OpenAI

llm = ChatOpenAI(temperature=0.0)
math_llm = OpenAI(temperature=0.0)
tools = load_tools(
    ["human", "llm-math"],
    llm=math_llm,
)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent_chain.run("What's my friend Eric's surname?")
```

#### Balancing Autonomy and Oversight

The key is finding the right balance between agent autonomy and human oversight:

-   **Selective Intervention**: Only involve humans for critical decisions
-   **Asynchronous Review**: Queue non-urgent decisions for batch human review
-   **Confidence Thresholds**: Use human input when agent confidence is low
-   **Progressive Autonomy**: Gradually reduce human involvement as agent reliability improves

By incorporating HITL effectively, agents can leverage human expertise while maintaining efficient operation. This creates a symbiotic system where both automated and human intelligence work together to achieve optimal results.

# Closing Remarks

We've talked about what agents are, how to implement them and the most important design patterns. By now, you should have an idea about how to work with them and the incredible potential of these systems. They can augment existing AI capabilities and provide more robust solutions to complex problems. This makes them particularly well-suited for tasks that require dynamic decision-making and adaptability, such as automated customer support, data analysis, and even creative endeavors like content generation.

While the promise of LLM agents is immense, there are still challenges to overcome. Their potential for such broad applications also means that we need to confront many complex failure modes, with varying degrees of negative effects. The main reason for there not being many agents in production environments is that, although it is easy to develop one that works right most of the time, it becomes increasingly harder to fight the remaining proportion of failure cases.

However, these challenges also present opportunities for innovation. As the field of LLM agents continues to evolve, there is a growing need for new tools, frameworks, and methodologies that can support the development and deployment of these systems. 

I wanted to write about the development ecosystem in this post, but it has already become too long, so I will reserve this along with examples of open and commercial applications and benchmarks of agentic skills for the next post. Next, I will upload another post showing the implementation of a ReAct agent from basic Python. You can find it in its [GitHub repository](https://github.com/miguelvc6/react-agent). The code is already finished, I just need some free time to keep writing in this blog. I also want to implement a comments section and an email subscription that notifies when a new post is published. In the meantime, feel free to contact me via email.

# References

[^1]: Russell, S., & Norvig, P. (2020). _Artificial Intelligence: A Modern Approach_ (4th ed.). Pearson.
[^2]: [What's next for AI agentic workflows ft. Andrew Ng of AI Fund](https://www.youtube.com/watch?v=sal78ACtGTc)
[^3]: Yao, Shunyu, et al. "React: Synergizing reasoning and acting in language models." arXiv preprint arXiv:2210.03629 (2022). [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
[^4]: Wei, Jason, et al. "Chain-of-thought prompting elicits reasoning in large language models." Advances in neural information processing systems 35 (2022): 24824-24837. [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
[^5]: [HumanEval Benchmark in paperswithcode](https://paperswithcode.com/sota/code-generation-on-humaneval)
[^6]: CoALA: Cognitive Architectures for Language Agents. Sumers, Theodore R., et al Transactions on Machine Learning Research (oct 2024). [https://arxiv.org/abs/2309.02427](https://arxiv.org/abs/2309.02427)
[^7]: Patil, Shishir G., et al. "Gorilla: Large language model connected with massive apis." arXiv preprint arXiv:2305.15334 (2023). [https://arxiv.org/abs/2305.15334](https://arxiv.org/abs/2305.15334)
[^8]: Shen, Yongliang, et al. "Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face." Advances in Neural Information Processing Systems 36 (2024). [https://arxiv.org/abs/2303.17580](https://arxiv.org/abs/2303.17580)
[^9]: Qian, Chen, et al. "Communicative agents for software development." arXiv preprint arXiv:2307.07924 6 (2023). [https://arxiv.org/abs/2307.07924](https://arxiv.org/abs/2307.07924)
[^10]: Du, Yilun, et al. "Improving factuality and reasoning in language models through multiagent debate." arXiv preprint arXiv:2305.14325 (2023). [https://arxiv.org/abs/2305.14325](https://arxiv.org/abs/2305.14325)