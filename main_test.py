from forecasting_tools import MonetaryCostManager, clean_indents, Estimator
import asyncio
import inspect
import json
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Coroutine, Literal, Sequence, TypeVar, cast, overload

from exceptiongroup import ExceptionGroup
from pydantic import BaseModel

from forecasting_tools.ai_models.agent_wrappers import general_trace_or_span
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.data_models.data_organizer import DataOrganizer, PredictionTypes
from forecasting_tools.data_models.forecast_report import (
    ForecastReport,
    ReasonedPrediction,
    ResearchWithPredictions,
)
from forecasting_tools.data_models.markdown_tree import MarkdownTree
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.helpers.metaculus_api import MetaculusApi
from forecasting_tools import (
    TemplateBot,
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    structure_output,
)
import asyncio
from prompt_researcher import PromptResearcher
import time
import argparse
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Enable debug logging for forecasting_tools models
logging.getLogger("forecasting_tools.ai_models.general_llm").setLevel(logging.DEBUG)
logging.getLogger("LiteLLM").setLevel(logging.DEBUG)


def test_main():
    prompt_researcher = PromptResearcher(
            research_reports_per_question=5,
            predictions_per_research_report=1,
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=False,
            skip_previously_forecasted_questions=True,
            )
    prompt_researcher.set_models(
            forecaster = "openrouter/anthropic/claude-opus-4.1"
            )
    question = MultipleChoiceQuestion(
            question_text="Who will win the 2025 College Football Championship?",
            options=["Texas", "Ohio State", "Penn State", "Georgia", "Oregon", "other"]
            )
    report = asyncio.run(prompt_researcher.forecast_question(question))
    print(report.prediction)
    print(report.explanation)


def remote_main():
    with MonetaryCostManager(5.00) as cost_manager:
        start = time.time()
        prompt_researcher = PromptResearcher(
                research_reports_per_question=5,
                predictions_per_research_report=1,
                use_research_summary_to_forecast=False,
                publish_reports_to_metaculus=True,
                skip_previously_forecasted_questions=True,
                )
        prompt_researcher.set_models(
                forecaster = "openrouter/anthropic/claude-opus-4.1"
                )
        parser = argparse.ArgumentParser(
                description="Run the Q1TemplateBot forecasting system"
                )
        parser.add_argument(
                "--mode",
                type=str,
                choices=["tournament", "metaculus_cup", "test_questions"],
                default="tournament",
                help="Specify the run mode (default: tournament)",
                )
        args = parser.parse_args()
        run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
        assert run_mode in [
                "tournament",
                "metaculus_cup",
                "test_questions",
                ], "Invalid run mode"

        try:
            if run_mode == "tournament":
                logger.info("Starting tournament forecasting...")
                logger.info(f"Forecasting on AI Competition (ID: {MetaculusApi.CURRENT_AI_COMPETITION_ID})")
                seasonal_tournament_reports = asyncio.run(
                        prompt_researcher.forecast_on_tournament(
                            MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
                            )
                        )
                logger.info(f"AI Competition reports: {len(seasonal_tournament_reports)} forecasts completed")

                logger.info(f"Forecasting on Minibench (ID: {MetaculusApi.CURRENT_MINIBENCH_ID})")
                minibench_reports = asyncio.run(
                        prompt_researcher.forecast_on_tournament(
                            MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
                            )
                        )
                logger.info(f"Minibench reports: {len(minibench_reports)} forecasts completed")
                forecast_reports = seasonal_tournament_reports + minibench_reports

            elif run_mode == "metaculus_cup":
                logger.info("Starting Metaculus Cup forecasting...")
                # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
                # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
                prompt_researcher.skip_previously_forecasted_questions = False
                logger.info(f"Forecasting on Metaculus Cup (ID: {MetaculusApi.CURRENT_METACULUS_CUP_ID})")
                forecast_reports = asyncio.run(
                        prompt_researcher.forecast_on_tournament(
                            MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
                            )
                        )
                logger.info(f"Metaculus Cup reports: {len(forecast_reports)} forecasts completed")

            elif run_mode == "test_questions":
                logger.info("Starting test questions forecasting...")
                # Example questions are a good way to test the bot's performance on a single question
                EXAMPLE_QUESTIONS = [
                        #"https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
                        #"https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
                        "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
                        #"https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
                        ]
                prompt_researcher.skip_previously_forecasted_questions = False
                logger.info(f"Loading {len(EXAMPLE_QUESTIONS)} test questions...")
                questions = [
                        MetaculusApi.get_question_by_url(question_url)
                        for question_url in EXAMPLE_QUESTIONS
                        ]
                logger.info("Running forecasts on test questions...")
                forecast_reports = asyncio.run(
                        prompt_researcher.forecast_questions(questions, return_exceptions=True)
                        )
                logger.info(f"Test questions reports: {len(forecast_reports)} forecasts completed")

            # Log report summaries and errors
            successful_reports = []
            failed_reports = []

            for report in forecast_reports:
                if isinstance(report, Exception):
                    failed_reports.append(report)
                    logger.error(f"Forecast failed with exception: {type(report).__name__}: {str(report)}")
                else:
                    successful_reports.append(report)
                    if hasattr(report, 'question') and hasattr(report.question, 'question_text'):
                        logger.info(f"Successfully forecasted: {report.question.question_text[:100]}...")
                        if hasattr(report, 'prediction'):
                            logger.info(f"  Prediction: {report.prediction}")
                        if hasattr(report, 'explanation') and report.explanation:
                            logger.debug(f"  Reasoning: {report.explanation[:200]}...")

            logger.info(f"Forecasting completed: {len(successful_reports)} successful, {len(failed_reports)} failed")

            if failed_reports:
                logger.warning(f"Failed forecasts: {len(failed_reports)}")
                for i, error in enumerate(failed_reports):
                    logger.error(f"  Error {i+1}: {type(error).__name__}: {str(error)}")

            # Log cost information
            current_cost = cost_manager.current_usage
            logger.info(f"Total cost: ${current_cost:.2f}")

        except Exception as e:
            logger.error(f"Fatal error during forecasting: {type(e).__name__}: {str(e)}")
            raise

        end = time.time()
        logger.info(f"Total execution time: {(end-start)/60:.2f} minutes")
        print(f"time_taken: {(end-start)/60:.2f}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    remote_main()

#question = BinaryQuestion(
#        question_text="Georgia will beat Alabama next week"
#)
#question = BinaryQuestion(
#        question_text="Will USC win the College Football National Championship in 2025"
#)
#question = MultipleChoiceQuestion(
#        question_text="Who will win the 2025 College Football Championship?",
#        options=["Texas", "Ohio State", "Penn State", "Georgia", "Oregon", "other"]
#        )
#question = BinaryQuestion(
#        question_text="Will Trump declare a National Emergence in September 2025?"
#)
#question = NumericQuestion(
#        question_text="How many touchdowns will Oregon's QB  Dante Moore throw in the 2025-2026 season?",
#        upper_bound=80,
#        lower_bound=0,
#        open_upper_bound=False,
#        open_lower_bound=False
#)
