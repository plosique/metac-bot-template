import argparse
import asyncio
import json
import logging
import numpy as np
from datetime import datetime
from typing import Literal, Dict, Any, Tuple
from dataclasses import dataclass

from forecasting_tools import (
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
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)


@dataclass
class DebateResult:
    """Result of a debate turn"""
    debater_id: str
    reasoning: str
    prediction: Any  # Can be float for binary, PredictedOptionList for multiple choice, NumericDistribution for numeric
    evidence_analysis: str = ""
    questions_for_opponent: str = ""


@dataclass
class DebateConfig:
    """Configuration for debate system"""
    max_turns: int = 3
    kl_threshold: float = 0.1


def load_bot_config(bot_mode: str) -> dict:
    """Load bot configuration from bots.json based on mode"""
    try:
        with open("bots.json", "r") as f:
            config = json.load(f)
        
        bot_config = config.get(bot_mode, {})
        
        # Convert researcher config to GeneralLlm if it's a dict
        if "researcher" in bot_config and isinstance(bot_config["researcher"], dict):
            researcher_config = bot_config["researcher"]
            bot_config["researcher"] = GeneralLlm(**researcher_config)
        
        # Convert default (predictor) config to GeneralLlm if it's a dict
        if "default" in bot_config and isinstance(bot_config["default"], dict):
            default_config = bot_config["default"]
            bot_config["default"] = GeneralLlm(**default_config)
        
        # Convert debater configs to GeneralLlm if they're dicts
        if "debater1" in bot_config and isinstance(bot_config["debater1"], dict):
            debater1_config = bot_config["debater1"]
            bot_config["debater1"] = GeneralLlm(**debater1_config)
            
        if "debater2" in bot_config and isinstance(bot_config["debater2"], dict):
            debater2_config = bot_config["debater2"]
            bot_config["debater2"] = GeneralLlm(**debater2_config)
            
        return bot_config
        
    except FileNotFoundError:
        logger.warning("bots.json not found, using default configuration")
        return {}
    except Exception as e:
        logger.error(f"Error reading bots.json: {e}")
        return {}


class FallTemplateBot2025(ForecastBot):
    """
    This is a copy of the template bot for Fall 2025 Metaculus AI Tournament.
    This bot is what is used by Metaculus in our benchmark, but is also provided as a template for new bot makers.
    This template is given as-is, and though we have covered most test cases
    in forecasting-tools it may be worth double checking key components locally.

    Main changes since Q2:
    - An LLM now parses the final forecast output (rather than programmatic parsing)
    - Added resolution criteria and fine print explicitly to the research prompt
    - Previously in the prompt, nothing about upper/lower bound was shown when the bounds were open. Now a suggestion is made when this is the case.
    - Support for nominal bounds was added (i.e. when there are discrete questions and normal upper/lower bounds are not as intuitive)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ones.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLMto intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions for the
    MiniBench and Seasonal AIB tournaments. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/deep-research/low",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/deep-research/low":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "model_name").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    def _llm_config_defaults(self) -> dict[str, str]:
        """Override to add debate-specific LLM roles"""
        defaults = super()._llm_config_defaults()
        defaults.update({
            "debater1": "openrouter/openai/gpt-4o",
            "debater2": "openrouter/openai/gpt-4o",
            "debate_config": {}  # This will be handled specially
        })
        return defaults

    _max_concurrent_questions = (
        2  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                Background information:
                {question.background_info}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif researcher == "asknews/news-summaries":
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            elif researcher == "asknews/deep-research/medium-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews", "google"],
                    search_depth=2,
                    max_depth=4,
                )
            elif researcher == "asknews/deep-research/high-depth":
                research = await AskNewsSearcher().get_formatted_deep_research(
                    question.question_text,
                    sources=["asknews", "google"],
                    search_depth=4,
                    max_depth=6,
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    async def _run_debate_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """Run a debate between two LLMs to reach consensus on binary question"""
        debate_config = self._get_debate_config()
        
        # Log debater models
        debater1_llm = self.get_llm("debater1")
        debater2_llm = self.get_llm("debater2")
        debater1_model = getattr(debater1_llm, 'model', str(debater1_llm))
        debater2_model = getattr(debater2_llm, 'model', str(debater2_llm))
        logger.info(f"Starting debate: Debater1={debater1_model} vs Debater2={debater2_model}")
        
        # Initial debate round
        debater1_result = await self._get_initial_debate_position(question, research, "debater1", 1)
        debater2_result = await self._get_initial_debate_position(question, research, "debater2", 2)
        
        logger.info(f"Initial positions - Debater1: {debater1_result.prediction:.4f}, Debater2: {debater2_result.prediction:.4f}")
        
        # Check for immediate consensus using KL divergence
        if self._check_consensus_kl_divergence(debater1_result.prediction, debater2_result.prediction, debate_config.kl_threshold):
            final_prediction = (debater1_result.prediction + debater2_result.prediction) / 2
            combined_reasoning = f"Debater 1 reasoning:\n{debater1_result.reasoning}\n\nDebater 2 reasoning:\n{debater2_result.reasoning}\n\nConsensus reached immediately."
            return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)
        
        # Run debate turns
        current_results = [debater1_result, debater2_result]
        
        for turn in range(1, debate_config.max_turns + 1):
            logger.info(f"Starting debate turn {turn}")
            
            # Cross-examination phase
            debater1_cross_exam = await self._cross_examine_opponent(question, research, current_results[1], "debater1", turn)
            debater2_cross_exam = await self._cross_examine_opponent(question, research, current_results[0], "debater2", turn)
            
            # Update predictions based on cross-examination
            debater1_result = await self._update_prediction_after_debate(
                question, research, current_results[0], debater2_cross_exam, current_results[1], "debater1", turn
            )
            debater2_result = await self._update_prediction_after_debate(
                question, research, current_results[1], debater1_cross_exam, current_results[0], "debater2", turn
            )
            
            current_results = [debater1_result, debater2_result]
            logger.info(f"Turn {turn} positions - Debater1: {debater1_result.prediction:.4f}, Debater2: {debater2_result.prediction:.4f}")
            
            # Check for consensus using KL divergence
            if self._check_consensus_kl_divergence(debater1_result.prediction, debater2_result.prediction, debate_config.kl_threshold):
                final_prediction = (debater1_result.prediction + debater2_result.prediction) / 2
                combined_reasoning = self._combine_debate_reasoning(current_results, f"Consensus reached after {turn} turns.")
                return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)
        
        # No consensus reached, use midpoint
        final_prediction = (current_results[0].prediction + current_results[1].prediction) / 2
        combined_reasoning = self._combine_debate_reasoning(current_results, f"No consensus after {debate_config.max_turns} turns. Using midpoint.")
        
        logger.info(f"Final prediction (midpoint): {final_prediction}")
        return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)

    async def _run_debate_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        """Run a debate between two LLMs to reach consensus on multiple choice question"""
        debate_config = self._get_debate_config()
        
        # Log debater models
        debater1_llm = self.get_llm("debater1")
        debater2_llm = self.get_llm("debater2")
        debater1_model = getattr(debater1_llm, 'model', str(debater1_llm))
        debater2_model = getattr(debater2_llm, 'model', str(debater2_llm))
        logger.info(f"Starting multiple choice debate: Debater1={debater1_model} vs Debater2={debater2_model}")
        
        # Initial debate round
        debater1_result = await self._get_initial_multiple_choice_position(question, research, "debater1", 1)
        debater2_result = await self._get_initial_multiple_choice_position(question, research, "debater2", 2)
        
        logger.info(f"Initial positions - Debater1: {[f'{opt.option_name}: {opt.probability:.3f}' for opt in self._get_option_list(debater1_result.prediction)]}")
        logger.info(f"Initial positions - Debater2: {[f'{opt.option_name}: {opt.probability:.3f}' for opt in self._get_option_list(debater2_result.prediction)]}")
        
        # Check for immediate consensus using KL divergence
        if self._check_consensus_kl_divergence(debater1_result.prediction, debater2_result.prediction, debate_config.kl_threshold):
            # Average the predictions
            final_prediction = self._average_multiple_choice_predictions([debater1_result.prediction, debater2_result.prediction])
            combined_reasoning = f"Debater 1 reasoning:\n{debater1_result.reasoning}\n\nDebater 2 reasoning:\n{debater2_result.reasoning}\n\nConsensus reached immediately."
            return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)
        
        # Run debate turns
        current_results = [debater1_result, debater2_result]
        
        for turn in range(1, debate_config.max_turns + 1):
            logger.info(f"Starting multiple choice debate turn {turn}")
            
            # Cross-examination phase
            debater1_cross_exam = await self._cross_examine_multiple_choice_opponent(question, research, current_results[1], "debater1", turn)
            debater2_cross_exam = await self._cross_examine_multiple_choice_opponent(question, research, current_results[0], "debater2", turn)
            
            # Update predictions based on cross-examination
            debater1_result = await self._update_multiple_choice_prediction_after_debate(
                question, research, current_results[0], debater2_cross_exam, current_results[1], "debater1", turn
            )
            debater2_result = await self._update_multiple_choice_prediction_after_debate(
                question, research, current_results[1], debater1_cross_exam, current_results[0], "debater2", turn
            )
            
            current_results = [debater1_result, debater2_result]
            logger.info(f"Turn {turn} positions - Debater1: {[f'{opt.option_name}: {opt.probability:.3f}' for opt in self._get_option_list(debater1_result.prediction)]}")
            logger.info(f"Turn {turn} positions - Debater2: {[f'{opt.option_name}: {opt.probability:.3f}' for opt in self._get_option_list(debater2_result.prediction)]}")
            
            # Check for consensus using KL divergence
            if self._check_consensus_kl_divergence(debater1_result.prediction, debater2_result.prediction, debate_config.kl_threshold):
                final_prediction = self._average_multiple_choice_predictions([debater1_result.prediction, debater2_result.prediction])
                combined_reasoning = self._combine_multiple_choice_debate_reasoning(current_results, f"Consensus reached after {turn} turns.")
                return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)
        
        # No consensus reached, use average
        final_prediction = self._average_multiple_choice_predictions([current_results[0].prediction, current_results[1].prediction])
        combined_reasoning = self._combine_multiple_choice_debate_reasoning(current_results, f"No consensus after {debate_config.max_turns} turns. Using average.")
        
        logger.info(f"Final multiple choice prediction: {[f'{opt.option_name}: {opt.probability:.3f}' for opt in self._get_option_list(final_prediction)]}")
        return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)

    async def _get_initial_multiple_choice_position(
        self, question: MultipleChoiceQuestion, research: str, debater_role: str, debater_num: int
    ) -> DebateResult:
        """Get initial position from a debater for multiple choice question"""
        prompt = clean_indents(
            f"""
            You are Debater {debater_num} in a forecasting debate. You will debate with another forecaster to reach the best possible prediction.

            Your task is to forecast on:
            {question.question_text}

            The options are: {question.options}

            Question background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print}

            Research available:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Provide your analysis focusing on:
            (a) The time left until the outcome is known
            (b) The status quo outcome if nothing changed
            (c) A scenario that results in an unexpected outcome
            (d) Key evidence and how it pushes towards or away from each option

            Remember that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and leave some moderate probability on most options to account for unexpected outcomes.

            IMPORTANT: All probabilities must be between 0.001 and 0.999 and sum to 1.0.

            End with your probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        
        reasoning = await self.get_llm(debater_role, "llm").invoke(prompt)
        logger.info(f"{debater_role} initial multiple choice reasoning: {reasoning}")
        
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        
        return DebateResult(
            debater_id=debater_role,
            reasoning=reasoning,
            prediction=predicted_option_list
        )

    async def _cross_examine_multiple_choice_opponent(
        self, question: MultipleChoiceQuestion, research: str, opponent_result: DebateResult, 
        debater_role: str, turn: int
    ) -> str:
        """Cross-examine opponent's reasoning for multiple choice question"""
        opponent_probs = [f"{opt.option_name}: {opt.probability:.3f}" for opt in self._get_option_list(opponent_result.prediction)]
        
        prompt = clean_indents(
            f"""
            You are examining your opponent's reasoning in a forecasting debate.

            Question: {question.question_text}
            Options: {question.options}
            Research: {research}

            Your opponent's reasoning:
            {opponent_result.reasoning}

            Your opponent's probabilities: {opponent_probs}

            Critically analyze your opponent's reasoning by answering these two key questions:

            1. Did your opponent miss any important evidence from the research or background information? What evidence did they overlook or underweight?

            2. Did your opponent rationally process the evidence they used? Are there any logical flaws, biases, or errors in their reasoning process?

            Be specific and constructive in your analysis. Focus on evidence and logical reasoning, not personal attacks.
            """
        )
        
        cross_exam = await self.get_llm(debater_role, "llm").invoke(prompt)
        logger.info(f"{debater_role} multiple choice cross-examination (turn {turn}): {cross_exam}")
        return cross_exam

    async def _update_multiple_choice_prediction_after_debate(
        self, question: MultipleChoiceQuestion, research: str, own_previous: DebateResult,
        opponent_cross_exam: str, opponent_result: DebateResult, debater_role: str, turn: int
    ) -> DebateResult:
        """Update multiple choice prediction after seeing opponent's cross-examination"""
        own_probs = [f"{opt.option_name}: {opt.probability:.3f}" for opt in self._get_option_list(own_previous.prediction)]
        opponent_probs = [f"{opt.option_name}: {opt.probability:.3f}" for opt in self._get_option_list(opponent_result.prediction)]
        
        prompt = clean_indents(
            f"""
            You are updating your forecast after reviewing your opponent's critique.

            Question: {question.question_text}
            Options: {question.options}
            Research: {research}

            Your previous reasoning:
            {own_previous.reasoning}

            Your previous probabilities: {own_probs}

            Your opponent's critique of your reasoning:
            {opponent_cross_exam}

            Your opponent's probabilities: {opponent_probs}

            Now consider:
            1. Are there valid points in your opponent's critique?
            2. Did you miss important evidence they identified?
            3. Should you adjust your reasoning based on their analysis?
            4. How much do you agree with your opponent's overall assessment?

            Provide your updated analysis and reasoning, taking into account:
            (a) Any valid criticisms from your opponent
            (b) Evidence you may have missed or underweighted
            (c) Whether your opponent's reasoning changed your confidence
            (d) Key evidence and how it pushes towards or away from each option

            IMPORTANT: All probabilities must be between 0.001 and 0.999 and sum to 1.0.

            End with your updated probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        
        updated_reasoning = await self.get_llm(debater_role, "llm").invoke(prompt)
        logger.info(f"{debater_role} updated multiple choice reasoning (turn {turn}): {updated_reasoning}")
        
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=updated_reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        
        return DebateResult(
            debater_id=debater_role,
            reasoning=updated_reasoning,
            prediction=predicted_option_list
        )

    def _average_multiple_choice_predictions(self, predictions: list[PredictedOptionList]) -> PredictedOptionList:
        """Average multiple choice predictions"""
        if not predictions:
            raise ValueError("No predictions to average")
        
        # Get all unique options
        all_options = set()
        for pred in predictions:
            for opt in self._get_option_list(pred):
                all_options.add(opt.option_name)
        
        # Calculate average probabilities for each option
        averaged_options = []
        for option in all_options:
            total_prob = 0.0
            count = 0
            for pred in predictions:
                for opt in self._get_option_list(pred):
                    if opt.option_name == option:
                        total_prob += opt.probability
                        count += 1
            
            if count > 0:
                avg_prob = total_prob / count
                # Create new PredictedOption object
                first_option_list = self._get_option_list(predictions[0])
                if first_option_list:
                    option_type = type(first_option_list[0])
                    averaged_options.append(option_type(option_name=option, probability=avg_prob))
                else:
                    # Fallback - create a simple object
                    from types import SimpleNamespace
                    averaged_options.append(SimpleNamespace(option_name=option, probability=avg_prob))
        
        # Normalize probabilities to sum to 1
        total_prob = sum(opt.probability for opt in averaged_options)
        if total_prob > 0:
            for opt in averaged_options:
                opt.probability /= total_prob
        
        # Try different constructor patterns for PredictedOptionList
        try:
            return PredictedOptionList(predicted_options=averaged_options)
        except:
            try:
                return PredictedOptionList(options=averaged_options)
            except:
                # If all else fails, create a simple wrapper
                from types import SimpleNamespace
                result = SimpleNamespace()
                result.predicted_options = averaged_options
                # Make it iterable like PredictedOptionList
                result.__iter__ = lambda: iter(averaged_options)
                return result

    def _combine_multiple_choice_debate_reasoning(self, results: list[DebateResult], conclusion: str) -> str:
        """Combine reasoning from all debaters for multiple choice into final reasoning"""
        combined = f"MULTIPLE CHOICE DEBATE-BASED FORECAST\n\n"
        
        for i, result in enumerate(results, 1):
            probs = [f"{opt.option_name}: {opt.probability:.3f}" for opt in result.prediction.options]
            combined += f"Debater {i} Final Position ({probs}):\n"
            combined += f"{result.reasoning}\n\n"
        
        combined += f"CONCLUSION: {conclusion}"
        return combined

    async def _run_debate_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """Run a debate between two LLMs to reach consensus on numeric question"""
        debate_config = self._get_debate_config()
        
        # Log debater models
        debater1_llm = self.get_llm("debater1")
        debater2_llm = self.get_llm("debater2")
        debater1_model = getattr(debater1_llm, 'model', str(debater1_llm))
        debater2_model = getattr(debater2_llm, 'model', str(debater2_llm))
        logger.info(f"Starting numeric debate: Debater1={debater1_model} vs Debater2={debater2_model}")
        
        # Initial debate round
        debater1_result = await self._get_initial_numeric_position(question, research, "debater1", 1)
        debater2_result = await self._get_initial_numeric_position(question, research, "debater2", 2)
        
        logger.info(f"Initial positions - Debater1: {[f'P{p.percentile}: {p.value}' for p in debater1_result.prediction.declared_percentiles]}")
        logger.info(f"Initial positions - Debater2: {[f'P{p.percentile}: {p.value}' for p in debater2_result.prediction.declared_percentiles]}")
        
        # Check for immediate consensus using KL divergence
        if self._check_consensus_kl_divergence(debater1_result.prediction, debater2_result.prediction, debate_config.kl_threshold):
            # Average the predictions
            final_prediction = self._average_numeric_predictions([debater1_result.prediction, debater2_result.prediction], question)
            combined_reasoning = f"Debater 1 reasoning:\n{debater1_result.reasoning}\n\nDebater 2 reasoning:\n{debater2_result.reasoning}\n\nConsensus reached immediately."
            return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)
        
        # Run debate turns
        current_results = [debater1_result, debater2_result]
        
        for turn in range(1, debate_config.max_turns + 1):
            logger.info(f"Starting numeric debate turn {turn}")
            
            # Cross-examination phase
            debater1_cross_exam = await self._cross_examine_numeric_opponent(question, research, current_results[1], "debater1", turn)
            debater2_cross_exam = await self._cross_examine_numeric_opponent(question, research, current_results[0], "debater2", turn)
            
            # Update predictions based on cross-examination
            debater1_result = await self._update_numeric_prediction_after_debate(
                question, research, current_results[0], debater2_cross_exam, current_results[1], "debater1", turn
            )
            debater2_result = await self._update_numeric_prediction_after_debate(
                question, research, current_results[1], debater1_cross_exam, current_results[0], "debater2", turn
            )
            
            current_results = [debater1_result, debater2_result]
            logger.info(f"Turn {turn} positions - Debater1: {[f'P{p.percentile}: {p.value}' for p in debater1_result.prediction.declared_percentiles]}")
            logger.info(f"Turn {turn} positions - Debater2: {[f'P{p.percentile}: {p.value}' for p in debater2_result.prediction.declared_percentiles]}")
            
            # Check for consensus using KL divergence
            if self._check_consensus_kl_divergence(debater1_result.prediction, debater2_result.prediction, debate_config.kl_threshold):
                final_prediction = self._average_numeric_predictions([debater1_result.prediction, debater2_result.prediction], question)
                combined_reasoning = self._combine_numeric_debate_reasoning(current_results, f"Consensus reached after {turn} turns.")
                return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)
        
        # No consensus reached, use average
        final_prediction = self._average_numeric_predictions([current_results[0].prediction, current_results[1].prediction], question)
        combined_reasoning = self._combine_numeric_debate_reasoning(current_results, f"No consensus after {debate_config.max_turns} turns. Using average.")
        
        logger.info(f"Final numeric prediction: {[f'P{p.percentile}: {p.value}' for p in final_prediction.declared_percentiles]}")
        return ReasonedPrediction(prediction_value=final_prediction, reasoning=combined_reasoning)

    async def _get_initial_numeric_position(
        self, question: NumericQuestion, research: str, debater_role: str, debater_num: int
    ) -> DebateResult:
        """Get initial position from a debater for numeric question"""
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        
        prompt = clean_indents(
            f"""
            You are Debater {debater_num} in a forecasting debate. You will debate with another forecaster to reach the best possible prediction.

            Your task is to forecast on:
            {question.question_text}

            Question background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Research available:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Provide your analysis focusing on:
            (a) The time left until the outcome is known
            (b) The outcome if nothing changed
            (c) The outcome if the current trend continued
            (d) The expectations of experts and markets
            (e) A scenario that results in a low outcome
            (f) A scenario that results in a high outcome
            (g) Key evidence and how it pushes towards higher or lower values

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        
        reasoning = await self.get_llm(debater_role, "llm").invoke(prompt)
        logger.info(f"{debater_role} initial numeric reasoning: {reasoning}")
        
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        
        return DebateResult(
            debater_id=debater_role,
            reasoning=reasoning,
            prediction=prediction
        )

    async def _cross_examine_numeric_opponent(
        self, question: NumericQuestion, research: str, opponent_result: DebateResult, 
        debater_role: str, turn: int
    ) -> str:
        """Cross-examine opponent's reasoning for numeric question"""
        opponent_percentiles = [f"P{p.percentile}: {p.value}" for p in opponent_result.prediction.declared_percentiles]
        
        prompt = clean_indents(
            f"""
            You are examining your opponent's reasoning in a forecasting debate.

            Question: {question.question_text}
            Research: {research}

            Your opponent's reasoning:
            {opponent_result.reasoning}

            Your opponent's distribution: {opponent_percentiles}

            Critically analyze your opponent's reasoning by answering these two key questions:

            1. Did your opponent miss any important evidence from the research or background information? What evidence did they overlook or underweight?

            2. Did your opponent rationally process the evidence they used? Are there any logical flaws, biases, or errors in their reasoning process?

            Be specific and constructive in your analysis. Focus on evidence and logical reasoning, not personal attacks.
            """
        )
        
        cross_exam = await self.get_llm(debater_role, "llm").invoke(prompt)
        logger.info(f"{debater_role} numeric cross-examination (turn {turn}): {cross_exam}")
        return cross_exam

    async def _update_numeric_prediction_after_debate(
        self, question: NumericQuestion, research: str, own_previous: DebateResult,
        opponent_cross_exam: str, opponent_result: DebateResult, debater_role: str, turn: int
    ) -> DebateResult:
        """Update numeric prediction after seeing opponent's cross-examination"""
        own_percentiles = [f"P{p.percentile}: {p.value}" for p in own_previous.prediction.declared_percentiles]
        opponent_percentiles = [f"P{p.percentile}: {p.value}" for p in opponent_result.prediction.declared_percentiles]
        
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        
        prompt = clean_indents(
            f"""
            You are updating your forecast after reviewing your opponent's critique.

            Question: {question.question_text}
            Research: {research}

            Your previous reasoning:
            {own_previous.reasoning}

            Your previous distribution: {own_percentiles}

            Your opponent's critique of your reasoning:
            {opponent_cross_exam}

            Your opponent's distribution: {opponent_percentiles}

            {lower_bound_message}
            {upper_bound_message}

            Now consider:
            1. Are there valid points in your opponent's critique?
            2. Did you miss important evidence they identified?
            3. Should you adjust your reasoning based on their analysis?
            4. How much do you agree with your opponent's overall assessment?

            Provide your updated analysis and reasoning, taking into account:
            (a) Any valid criticisms from your opponent
            (b) Evidence you may have missed or underweighted
            (c) Whether your opponent's reasoning changed your confidence
            (d) Key evidence and how it pushes towards higher or lower values

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your updated answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        
        updated_reasoning = await self.get_llm(debater_role, "llm").invoke(prompt)
        logger.info(f"{debater_role} updated numeric reasoning (turn {turn}): {updated_reasoning}")
        
        percentile_list: list[Percentile] = await structure_output(
            updated_reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        
        return DebateResult(
            debater_id=debater_role,
            reasoning=updated_reasoning,
            prediction=prediction
        )

    def _average_numeric_predictions(self, predictions: list[NumericDistribution], question: NumericQuestion) -> NumericDistribution:
        """Average numeric predictions"""
        if not predictions:
            raise ValueError("No predictions to average")
        
        # Average the percentile values for each percentile level
        percentile_levels = [p.percentile for p in predictions[0].declared_percentiles]
        averaged_percentiles = []
        
        for percentile_level in percentile_levels:
            total_value = 0.0
            count = 0
            for pred in predictions:
                for p in pred.declared_percentiles:
                    if p.percentile == percentile_level:
                        total_value += p.value
                        count += 1
            
            if count > 0:
                avg_value = total_value / count
                averaged_percentiles.append(Percentile(percentile=percentile_level, value=avg_value))
        
        return NumericDistribution.from_question(averaged_percentiles, question)

    def _combine_numeric_debate_reasoning(self, results: list[DebateResult], conclusion: str) -> str:
        """Combine reasoning from all debaters for numeric into final reasoning"""
        combined = f"NUMERIC DEBATE-BASED FORECAST\n\n"
        
        for i, result in enumerate(results, 1):
            percentiles = [f"P{p.percentile}: {p.value}" for p in result.prediction.declared_percentiles]
            combined += f"Debater {i} Final Position ({percentiles}):\n"
            combined += f"{result.reasoning}\n\n"
        
        combined += f"CONCLUSION: {conclusion}"
        return combined

    def _get_option_list(self, predicted_option_list) -> list:
        """Get options from PredictedOptionList regardless of attribute name"""
        # Handle different possible structures
        if hasattr(predicted_option_list, 'options'):
            return predicted_option_list.options
        elif hasattr(predicted_option_list, 'predicted_options'):
            return predicted_option_list.predicted_options  
        elif hasattr(predicted_option_list, '__iter__') and not isinstance(predicted_option_list, str):
            # The PredictedOptionList might be iterable itself
            return list(predicted_option_list)
        else:
            # Fallback
            return []

    async def _get_initial_debate_position(
        self, question: BinaryQuestion, research: str, debater_role: str, debater_num: int
    ) -> DebateResult:
        """Get initial position from a debater"""
        prompt = clean_indents(
            f"""
            You are Debater {debater_num} in a forecasting debate. You will debate with another forecaster to reach the best possible prediction.

            Your task is to forecast on:
            {question.question_text}

            Question background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print}

            Research available:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Provide your analysis focusing on:
            (a) The time left until the outcome is known
            (b) The status quo outcome if nothing changed
            (c) A scenario that results in a No outcome
            (d) A scenario that results in a Yes outcome
            (e) Key evidence and how it pushes towards or away from each resolution

            Remember that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            End with your probability estimate: "Probability: XX%"
            """
        )
        
        reasoning = await self.get_llm(debater_role, "llm").invoke(prompt)
        logger.info(f"{debater_role} initial reasoning: {reasoning}")
        
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        prediction = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        
        return DebateResult(
            debater_id=debater_role,
            reasoning=reasoning,
            prediction=prediction
        )

    async def _cross_examine_opponent(
        self, question: BinaryQuestion, research: str, opponent_result: DebateResult, 
        debater_role: str, turn: int
    ) -> str:
        """Cross-examine opponent's reasoning and evidence"""
        prompt = clean_indents(
            f"""
            You are examining your opponent's reasoning in a forecasting debate.

            Question: {question.question_text}
            Research: {research}

            Your opponent's reasoning:
            {opponent_result.reasoning}

            Your opponent predicted: {opponent_result.prediction * 100:.1f}%

            Critically analyze your opponent's reasoning by answering these two key questions:

            1. Did your opponent miss any important evidence from the research or background information? What evidence did they overlook or underweight?

            2. Did your opponent rationally process the evidence they used? Are there any logical flaws, biases, or errors in their reasoning process?

            Be specific and constructive in your analysis. Focus on evidence and logical reasoning, not personal attacks.
            """
        )
        
        cross_exam = await self.get_llm(debater_role, "llm").invoke(prompt)
        logger.info(f"{debater_role} cross-examination (turn {turn}): {cross_exam}")
        return cross_exam

    async def _update_prediction_after_debate(
        self, question: BinaryQuestion, research: str, own_previous: DebateResult,
        opponent_cross_exam: str, opponent_result: DebateResult, debater_role: str, turn: int
    ) -> DebateResult:
        """Update prediction after seeing opponent's cross-examination"""
        prompt = clean_indents(
            f"""
            You are updating your forecast after reviewing your opponent's critique.

            Question: {question.question_text}
            Research: {research}

            Your previous reasoning:
            {own_previous.reasoning}

            Your previous prediction: {own_previous.prediction * 100:.1f}%

            Your opponent's critique of your reasoning:
            {opponent_cross_exam}

            Your opponent's prediction: {opponent_result.prediction * 100:.1f}%

            Now consider:
            1. Are there valid points in your opponent's critique?
            2. Did you miss important evidence they identified?
            3. Should you adjust your reasoning based on their analysis?
            4. How much do you agree with your opponent's overall assessment?

            Provide your updated analysis and reasoning, taking into account:
            (a) Any valid criticisms from your opponent
            (b) Evidence you may have missed or underweighted
            (c) Whether your opponent's reasoning changed your confidence
            (d) Key evidence and how it pushes towards or away from each resolution

            End with your updated probability: "Probability: XX%"
            """
        )
        
        updated_reasoning = await self.get_llm(debater_role, "llm").invoke(prompt)
        logger.info(f"{debater_role} updated reasoning (turn {turn}): {updated_reasoning}")
        
        binary_prediction: BinaryPrediction = await structure_output(
            updated_reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        prediction = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        
        return DebateResult(
            debater_id=debater_role,
            reasoning=updated_reasoning,
            prediction=prediction
        )

    def _get_debate_config(self) -> DebateConfig:
        """Get debate configuration from llms config"""
        if hasattr(self, '_llms') and self._llms and 'debate_config' in self._llms:
            config_dict = self._llms['debate_config']
            return DebateConfig(
                max_turns=config_dict.get('max_turns', 3),
                kl_threshold=config_dict.get('kl_threshold', 0.1)
            )
        return DebateConfig()

    def _check_consensus(self, predictions: list[float], threshold: float) -> bool:
        """Check if predictions are within consensus threshold"""
        if len(predictions) < 2:
            return True
        return abs(predictions[0] - predictions[1]) <= threshold
    
    def _calculate_kl_divergence_binary(self, p1: float, p2: float) -> float:
        """Calculate KL divergence for binary predictions"""
        # Ensure probabilities are in valid range
        p1 = max(1e-10, min(1 - 1e-10, p1))
        p2 = max(1e-10, min(1 - 1e-10, p2))
        
        # KL divergence for binary: p1*log(p1/p2) + (1-p1)*log((1-p1)/(1-p2))
        kl_div = p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))
        return float(kl_div)
    
    def _calculate_kl_divergence_multiple_choice(self, probs1: list[float], probs2: list[float]) -> float:
        """Calculate KL divergence for multiple choice predictions"""
        if len(probs1) != len(probs2):
            raise ValueError("Probability distributions must have same length")
        
        # Ensure probabilities sum to 1 and are positive
        probs1 = np.array(probs1)
        probs2 = np.array(probs2)
        
        # Normalize to sum to 1
        probs1 = probs1 / np.sum(probs1)
        probs2 = probs2 / np.sum(probs2)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        probs1 = np.maximum(probs1, eps)
        probs2 = np.maximum(probs2, eps)
        
        # KL divergence: sum(p1 * log(p1/p2))
        kl_div = np.sum(probs1 * np.log(probs1 / probs2))
        return float(kl_div)
    
    def _calculate_kl_divergence_numeric(self, percentiles1: list[float], percentiles2: list[float]) -> float:
        """Calculate KL divergence for numeric distributions using percentiles"""
        if len(percentiles1) != len(percentiles2):
            raise ValueError("Percentile lists must have same length")
        
        # Convert percentiles to approximate probability densities
        # This is a simplified approach - for more accuracy, you'd want to fit distributions
        bins = 20  # Number of bins for discretization
        
        # Create bins from min to max of both distributions
        all_values = percentiles1 + percentiles2
        min_val, max_val = min(all_values), max(all_values)
        
        # Avoid identical min/max
        if min_val == max_val:
            return 0.0
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Convert percentiles to histogram (approximate PDF)
        hist1, _ = np.histogram(percentiles1, bins=bin_edges, density=True)
        hist2, _ = np.histogram(percentiles2, bins=bin_edges, density=True)
        
        # Normalize to probability distributions
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # Add epsilon to avoid log(0)
        eps = 1e-10
        hist1 = np.maximum(hist1, eps)
        hist2 = np.maximum(hist2, eps)
        
        # Calculate KL divergence
        kl_div = np.sum(hist1 * np.log(hist1 / hist2))
        return float(kl_div)
    
    def _check_consensus_kl_divergence(self, pred1, pred2, kl_threshold: float = 0.1) -> bool:
        """Check consensus using KL divergence for any prediction type"""
        if isinstance(pred1, float) and isinstance(pred2, float):
            # Binary case
            kl_div = self._calculate_kl_divergence_binary(pred1, pred2)
        elif hasattr(pred1, 'options') or hasattr(pred1, 'predicted_options'):
            # Multiple choice case - extract probabilities using helper
            options1 = self._get_option_list(pred1)
            options2 = self._get_option_list(pred2)
            if options1 and options2:
                probs1 = [opt.probability for opt in options1]
                probs2 = [opt.probability for opt in options2]
                kl_div = self._calculate_kl_divergence_multiple_choice(probs1, probs2)
            else:
                logger.warning(f"Could not extract options from predictions: {type(pred1)}, {type(pred2)}")
                return False
        elif hasattr(pred1, 'declared_percentiles') and hasattr(pred2, 'declared_percentiles'):
            # Numeric case - use percentile values
            vals1 = [p.value for p in pred1.declared_percentiles]
            vals2 = [p.value for p in pred2.declared_percentiles]
            kl_div = self._calculate_kl_divergence_numeric(vals1, vals2)
        else:
            # Fallback to simple difference for unknown types
            logger.warning(f"Unknown prediction types for KL divergence: {type(pred1)}, {type(pred2)}")
            return abs(float(pred1) - float(pred2)) <= kl_threshold
        
        logger.info(f"KL divergence: {kl_div:.6f}, threshold: {kl_threshold}")
        return kl_div <= kl_threshold

    def _combine_debate_reasoning(self, results: list[DebateResult], conclusion: str) -> str:
        """Combine reasoning from all debaters into final reasoning"""
        combined = f"DEBATE-BASED FORECAST\n\n"
        
        for i, result in enumerate(results, 1):
            combined += f"Debater {i} Final Position ({result.prediction * 100:.1f}%):\n"
            combined += f"{result.reasoning}\n\n"
        
        combined += f"CONCLUSION: {conclusion}"
        return combined

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """Run forecasting on binary question using debate system if debaters are configured"""
        # Check if debate system is configured by trying to get debater LLMs
        try:
            debater1_llm = self.get_llm("debater1")
            debater2_llm = self.get_llm("debater2")
            logger.info(f"Using debate system for URL {question.page_url}")
            return await self._run_debate_on_binary(question, research)
        except Exception as e:
            # Fall back to original single-forecaster method
            logger.info(f"Using single forecaster for URL {question.page_url} (debaters not available): {e}")
        
        # Single forecaster fallback
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        # Check if debate system is configured by trying to get debater LLMs
        try:
            debater1_llm = self.get_llm("debater1")
            debater2_llm = self.get_llm("debater2")
            logger.info(f"Using debate system for multiple choice URL {question.page_url}")
            return await self._run_debate_on_multiple_choice(question, research)
        except Exception as e:
            # Fall back to original single-forecaster method
            logger.info(f"Using single forecaster for multiple choice URL {question.page_url} (debaters not available): {e}")
        
        # Single forecaster fallback
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            IMPORTANT: All probabilities must be between 0.001 and 0.999 and sum to 1.0.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        # Check if debate system is configured by trying to get debater LLMs
        try:
            debater1_llm = self.get_llm("debater1")
            debater2_llm = self.get_llm("debater2")
            logger.info(f"Using debate system for numeric URL {question.page_url}")
            return await self._run_debate_on_numeric(question, research)
        except Exception as e:
            # Fall back to original single-forecaster method
            logger.info(f"Using single forecaster for numeric URL {question.page_url} (debaters not available): {e}")
        
        # Single forecaster fallback
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run debate forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions", "local_binary"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    parser.add_argument(
        "--bot-mode",
        type=str,
        choices=["template", "debug", "main", "debug_with_researcher", "main_with_researcher"],
        default="template",
        help="Specify the bot mode (default: template)",
    )
    parser.add_argument(
        "--research-reports",
        type=int,
        default=1,
        help="Number of research reports per question (default: 1)",
    )
    parser.add_argument(
        "--predictions-per-research",
        type=int,
        default=1,
        help="Number of predictions per research report (default: 1)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions", "local_binary"] = args.mode
    bot_mode: Literal["template", "debug", "main", "debug_with_researcher", "main_with_researcher"] = args.bot_mode
    research_reports: int = args.research_reports
    predictions_per_research: int = args.predictions_per_research
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
        "local_binary",
    ], "Invalid run mode"
    assert bot_mode in ["template", "debug", "main", "debug_with_researcher", "main_with_researcher"], "Invalid bot mode"

    # Load bot configuration based on mode
    llm_config = load_bot_config(bot_mode)
    logger.info(f"Loaded llm_config for bot_mode '{bot_mode}': {llm_config}")
    
    template_bot = FallTemplateBot2025(
        research_reports_per_question=research_reports,
        predictions_per_research_report=predictions_per_research,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to="./debate_reports",
        skip_previously_forecasted_questions=False,
        llms=llm_config if llm_config else None,
    )
    
    logger.info(f"Bot created. Has llms attribute: {hasattr(template_bot, 'llms')}")
    if hasattr(template_bot, 'llms'):
        logger.info(f"Bot llms: {template_bot.llms}")
    
    # Also check what the get_llm method returns for our debaters
    try:
        debater1_llm = template_bot.get_llm("debater1")
        logger.info(f"get_llm('debater1') returns: {debater1_llm}")
        debater2_llm = template_bot.get_llm("debater2") 
        logger.info(f"get_llm('debater2') returns: {debater2_llm}")
    except Exception as e:
        logger.info(f"Error getting debater LLMs: {e}")
    
    # Check _llm_configs (internal attribute the framework might use)
    if hasattr(template_bot, '_llm_configs'):
        logger.info(f"Bot _llm_configs: {template_bot._llm_configs}")
    if hasattr(template_bot, '_llms'):
        logger.info(f"Bot _llms: {template_bot._llms}")

    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            #"https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            #"https://www.metaculus.com/questions/38951/4-will-the-tiger-point-wastewater-treatment-facility-expansion-stay-in-budget/",
            #"https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            #"https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    elif run_mode == "local_binary":
        # Load local binary questions from JSON file
        import json
        from datetime import datetime
        
        try:
            with open("local_binary_questions.json", "r") as f:
                local_questions_data = json.load(f)
        except FileNotFoundError:
            logger.error("local_binary_questions.json not found. Please create this file with your binary questions.")
            exit(1)
        
        # Convert JSON data to BinaryQuestion objects
        questions = []
        for q_data in local_questions_data:
            # Create a mock BinaryQuestion-like object
            question = BinaryQuestion(
                id=q_data.get("id", "local"),
                question_text=q_data["question_text"],
                background_info=q_data.get("background_info", ""),
                resolution_criteria=q_data.get("resolution_criteria", ""),
                fine_print=q_data.get("fine_print", ""),
                page_url=f"local://binary/{q_data.get('id', 'unknown')}",
                # Set dates if provided, otherwise use distant future
                scheduled_resolve_time=datetime.fromisoformat(q_data.get("resolves_at", "2030-01-01T00:00:00Z").replace('Z', '+00:00')) if q_data.get("resolves_at") else datetime(2030, 1, 1),
                scheduled_close_time=datetime.fromisoformat(q_data.get("closes_at", "2029-01-01T00:00:00Z").replace('Z', '+00:00')) if q_data.get("closes_at") else datetime(2029, 1, 1),
            )
            questions.append(question)
        
        template_bot.skip_previously_forecasted_questions = False
        template_bot.publish_reports_to_metaculus = False  # Don't submit to Metaculus
        
        logger.info(f"Loaded {len(questions)} local binary questions")
        for q in questions:
            logger.info(f"- {q.question_text}")
        
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    
    template_bot.log_report_summary(forecast_reports)
