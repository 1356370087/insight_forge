import asyncio
import uuid

from langsmith import Client

from open_deep_research.agents.query_engine import QueryEngine

client = Client()

dataset_name = "ODR: First Supervisor Parallelism"
def right_parallelism_evaluator(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    return {
        "key": "right_parallelism", 
        "score": len(outputs["output"]["supervisor_messages"][-1].tool_calls) == reference_outputs["parallel"]
    }

async def target(inputs: dict):
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    # NOTE: Configure the right dataset and evaluators
    config["configurable"]["max_structured_output_retries"] = 3
    config["configurable"]["allow_clarification"] = False
    config["configurable"]["max_concurrent_research_units"] = 10
    config["configurable"]["search_api"] = "tavily"     # NOTE: We use Tavily to stay consistent
    config["configurable"]["max_researcher_iterations"] = 3
    config["configurable"]["max_react_tool_calls"] = 10
    config["configurable"]["summarization_model"] = "openai:gpt-4.1-nano"
    config["configurable"]["summarization_model_max_tokens"] = 8192
    config["configurable"]["research_model"] = "openai:gpt-4.1"
    config["configurable"]["research_model_max_tokens"] = 10000
    config["configurable"]["compression_model"] = "openai:gpt-4.1-mini"
    config["configurable"]["compression_model_max_tokens"] = 10000
    config["configurable"]["final_report_model"] = "openai:gpt-4.1"
    config["configurable"]["final_report_model_max_tokens"] = 10000
    # NOTE: We do not use MCP tools to stay consistent
    engine = QueryEngine(config)
    final_state = await engine.submit_message(
        [{"role": "user", "content": inputs["messages"][0]["content"]}],
        config,
    )
    return final_state



async def main():
    return await client.aevaluate(
        target,
        data=dataset_name,
        evaluators=[right_parallelism_evaluator],
        experiment_prefix="v1 #",
        max_concurrency=1,
    )

if __name__ == "__main__":
    results = asyncio.run(main())
    print(results)  # noqa: T201


