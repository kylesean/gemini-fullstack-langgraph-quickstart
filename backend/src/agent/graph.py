import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
# from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    # get_citations,
    get_research_topic,
    # insert_citation_markers,
    format_sources,
    resolve_urls,
)

load_dotenv()


# if os.getenv("GEMINI_API_KEY") is None:
#     raise ValueError("GEMINI_API_KEY is not set")
#
# # Used for Google Search API
# genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    # llm = ChatGoogleGenerativeAI(
    #     model=configurable.query_generator_model,
    #     temperature=1.0,
    #     max_retries=2,
    #     api_key=os.getenv("GEMINI_API_KEY"),
    # )

    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList, method="function_calling")

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


# def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
#     """LangGraph node that performs web research using the native Google Search API tool.
#
#     Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.
#
#     Args:
#         state: Current graph state containing the search query and research loop count
#         config: Configuration for the runnable, including search API settings
#
#     Returns:
#         Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
#     """
#     # Configure
#     configurable = Configuration.from_runnable_config(config)
#     formatted_prompt = web_searcher_instructions.format(
#         current_date=get_current_date(),
#         research_topic=state["search_query"],
#     )
#
#     # Uses the google genai client as the langchain client doesn't return grounding metadata
#     response = genai_client.models.generate_content(
#         model=configurable.query_generator_model,
#         contents=formatted_prompt,
#         config={
#             "tools": [{"google_search": {}}],
#             "temperature": 0,
#         },
#     )
#     # resolve the urls to short urls for saving tokens and time
#     resolved_urls = resolve_urls(
#         response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
#     )
#     # Gets the citations and adds them to the generated text
#     citations = get_citations(response, resolved_urls)
#     modified_text = insert_citation_markers(response.text, citations)
#     sources_gathered = [item for citation in citations for item in citation["segments"]]
#
#     return {
#         "sources_gathered": sources_gathered,
#         "search_query": [state["search_query"]],
#         "web_research_result": [modified_text],
#     }

def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """
    LangGraph node that performs web research using a generic search tool.
    This version is corrected to handle data types properly and avoid JSON errors.
    """
    # 1. 初始化工具和 LLM
    search_tool = TavilySearchResults(max_results=3)
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    llm_with_tools = llm.bind_tools([search_tool])

    # 2. 准备 prompt 并让 LLM 决定是否调用工具
    prompt = f"Current date: {get_current_date()}. Please search for information on: {state['search_query']}"
    ai_msg = llm_with_tools.invoke(prompt)

    if not ai_msg.tool_calls:
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": ["No information found from web search."],
        }

    # 3. 执行工具调用，并同时保存原始结果和为 LLM 准备的消息
    tool_outputs_for_llm = []
    raw_search_results = []
    for tool_call in ai_msg.tool_calls:
        # 调用工具，得到的是 Python 列表/字典
        tool_output_data = search_tool.invoke(tool_call["args"])

        # 直接保存原始的 Python 对象，用于后续提取来源
        raw_search_results.extend(tool_output_data)

        # 为 LLM 准备 ToolMessage，这里 content 必须是字符串
        tool_outputs_for_llm.append(
            ToolMessage(content=str(tool_output_data), tool_call_id=tool_call["id"])
        )

    # 4. 将工具结果发回给 LLM 进行总结
    final_prompt = [HumanMessage(content=prompt), ai_msg, *tool_outputs_for_llm]
    summary_llm = ChatOpenAI(
        model=configurable.answer_model,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    summary = summary_llm.invoke(final_prompt).content

    # 5. 从原始结果中提取来源，这里不再需要 json.loads()
    sources = []
    for res in raw_search_results:
        # res 本身就是一个字典，可以直接访问
        if res.get("url") and res.get("title"):
            sources.append({"value": res["url"], "label": res["title"], "short_url": res["url"]})

    return {
        "sources_gathered": sources,
        "search_query": [state["search_query"]],
        "web_research_result": [summary],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    # llm = ChatGoogleGenerativeAI(
    #     model=reasoning_model,
    #     temperature=1.0,
    #     max_retries=2,
    #     api_key=os.getenv("GEMINI_API_KEY"),
    # )

    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    result = llm.with_structured_output(Reflection, method="function_calling").invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
        state: ReflectionState,
        config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


# def finalize_answer(state: OverallState, config: RunnableConfig):
#     """LangGraph node that finalizes the research summary.
#
#     Prepares the final output by deduplicating and formatting sources, then
#     combining them with the running summary to create a well-structured
#     research report with proper citations.
#
#     Args:
#         state: Current graph state containing the running summary and sources gathered
#
#     Returns:
#         Dictionary with state update, including running_summary key containing the formatted final summary with sources
#     """
#     configurable = Configuration.from_runnable_config(config)
#     reasoning_model = state.get("reasoning_model") or configurable.answer_model
#
#     # Format the prompt
#     current_date = get_current_date()
#     formatted_prompt = answer_instructions.format(
#         current_date=current_date,
#         research_topic=get_research_topic(state["messages"]),
#         summaries="\n---\n\n".join(state["web_research_result"]),
#     )
#
#     # init Reasoning Model, default to Gemini 2.5 Flash
#     # llm = ChatGoogleGenerativeAI(
#     #     model=reasoning_model,
#     #     temperature=0,
#     #     max_retries=2,
#     #     api_key=os.getenv("GEMINI_API_KEY"),
#     # )
#
#     llm = ChatOpenAI(
#         model=configurable.query_generator_model,
#         temperature=1.0,
#         max_retries=2,
#         api_key=os.getenv("OPENAI_API_KEY"),
#         base_url=os.getenv("OPENAI_API_BASE"),
#     )
#     result = llm.invoke(formatted_prompt)
#
#     # Replace the short urls with the original urls and add all used urls to the sources_gathered
#     unique_sources = []
#     for source in state["sources_gathered"]:
#         if source["short_url"] in result.content:
#             result.content = result.content.replace(
#                 source["short_url"], source["value"]
#             )
#             unique_sources.append(source)
#
#     return {
#         "messages": [AIMessage(content=result.content)],
#         "sources_gathered": unique_sources,
#     }

def finalize_answer(state: OverallState, config: RunnableConfig):
    """
    LangGraph node that finalizes the research summary.
    It generates the final answer and appends a formatted list of sources.
    """
    configurable = Configuration.from_runnable_config(config)
    # 在改造时，这里应该使用 ChatOpenAI
    llm = ChatOpenAI(
        model=state.get("reasoning_model") or configurable.answer_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    # 准备 Prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # 1. 调用 LLM 生成核心答案
    answer_text = llm.invoke(formatted_prompt).content

    # 2. 获取所有收集到的来源
    sources_gathered = state.get("sources_gathered", [])

    # 3. 使用我们的新工具函数格式化来源列表
    sources_md = format_sources(sources_gathered)

    # 4. 将来源列表附加到答案末尾
    final_content = answer_text + sources_md

    return {
        "messages": [AIMessage(content=final_content)],
        "sources_gathered": sources_gathered,  # 仍然可以传回 sources 用于调试或前端其他用途
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
