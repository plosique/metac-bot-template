
import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    TemplateBot,
    BinaryQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.forecast_report import (
    ForecastReport,
    ResearchWithPredictions,
)
import re
import os
import requests
import time
import asyncio


class PromptResearcher(TemplateBot): 

    def set_models( 
                 self, 
                 forecaster,
                 ): 
        self.forecaster = forecaster
        self.forecaster_guidelines = """ 
            Here's Great Judgements forecasting guidelines: 
            A superforecaster goes through three steps:
            (1) classifies evidence, 
            (2) judges the weights of evidence,
            (3) updates systematically.

            1. Classify the Data
            The first job is to split available information into categories that matter for reliability and relevance. Pay close attention to 
            the reliability and timeliness of the provided research. 
            2. Rank/Weight the Evidence
            Not all evidence should influence equally. Separate strong from weak signals:
            3. Update on the evidence
                Start from a base rate prior 
                Integrate higher-ranked evidence first, since these are more impactful. .
                Use Bayesian updating where possible, or at least a structured approximate adjustment
                Track how much weight your putting on each evidence cluster i.e if you the evidence you're considering
                is very weak compared to the evidence you've aready considered than your updates should be small. 

            While going through these steps know when to make exceptions. Consider that in many cases crowd-sourced wisdom, the status quo 
            , and well-founded simple statistical trends make the best predictions.
            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) What the crowd-sourced wisdom says 


            """
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

    async def optimize_guidelines(self, question: MetaculusQuestion):
        prompt = f""" 
            You are a superforecaster at a Great Judgement Inc. You have forecasting guidelines that
            are generally good but could be optimized to help with forecasters on specific questions. As a 
            superforecaster you know when the rules can be ignored and when they should be adhered to.Given the 
            following question and the guidelines provide new guidelines optimized for the question at hand. After
            considering everthing output the new guidelines between the tags <GUIDLINEs> </GUIDLINES>.

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            The question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            {self.forecaster_guidelines}

        """
        guidelines = await GeneralLlm(self.forecaster).invoke(prompt)

        prompt_match = re.search(r'<GUIDELINES>(.*?)</GUIDELINES>', guidelines, re.DOTALL)
        extracted_prompt = prompt_match.group(1).strip() if prompt_match else guidelines
        self.forecaster_guidelines = extracted_prompt
        print(f""" 
        <GUIDELINES> 
        {self.forecaster_guidelines}
        </GUIDELINES>
        """
        )
        # Format strings will be added at the end of individual forecast methods
    async def _run_individual_question(
        self, question: MetaculusQuestion
    ) -> ForecastReport:
        notepad = await self._initialize_notepad(question)
        await self.optimize_guidelines(question)
        async with self._note_pad_lock:
            self._note_pads.append(notepad)
        with MonetaryCostManager() as cost_manager:
            start_time = time.time()
            prediction_tasks = [
                self._research_and_make_predictions(question)
                for _ in range(self.research_reports_per_question)
            ]
            valid_prediction_set, research_errors, exception_group = (
                await self._gather_results_and_exceptions(prediction_tasks)  # type: ignore
            )
            if research_errors:
                logger.warning(
                    f"Encountered errors while researching: {research_errors}"
                )
            if len(valid_prediction_set) == 0:
                assert exception_group, "Exception group should not be None"
                self._reraise_exception_with_prepended_message(
                    exception_group,
                    f"All {self.research_reports_per_question} research reports/predictions failed",
                )
            prediction_errors = [
                error
                for prediction_set in valid_prediction_set
                for error in prediction_set.errors
            ]
            all_errors = research_errors + prediction_errors

            report_type = DataOrganizer.get_report_type_for_question_type(
                type(question)
            )
            all_predictions = [
                reasoned_prediction.prediction_value
                for research_prediction_collection in valid_prediction_set
                for reasoned_prediction in research_prediction_collection.predictions
            ]
            aggregated_prediction = await self._aggregate_predictions(
                all_predictions,
                question,
            )
            end_time = time.time()
            time_spent_in_minutes = (end_time - start_time) / 60
            final_cost = cost_manager.current_usage

        unified_explanation = self._create_unified_explanation(
            question,
            valid_prediction_set,
            aggregated_prediction,
            final_cost,
            time_spent_in_minutes,
        )
        report = report_type(
            question=question,
            prediction=aggregated_prediction,
            explanation=unified_explanation,
            price_estimate=final_cost,
            minutes_taken=time_spent_in_minutes,
            errors=all_errors,
        )
        if self.publish_reports_to_metaculus:
            await report.publish_report_to_metaculus()
        await self._remove_notepad(question)
        return report
            
    async def call_perplexity(self,system_prompt, user_prompt) -> str:
        url = "https://api.perplexity.ai/chat/completions"
        api_key = os.getenv("PERPLEXITY_API_KEY")
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",  # this is a system prompt designed to guide the perplexity assistant
                    "content": system_prompt
                },
                {
                    "role": "user",  # this is the actual prompt we ask the perplexity assistant to answer
                    "content": user_prompt
                },
            ],
        }
        response = requests.post(url=url, json=payload, headers=headers)
        if not response.ok:
            raise Exception(response.text)
        content = response.json()["choices"][0]["message"]["content"]
        return content

    async def run_research(self, question: MetaculusQuestion):

            forecaster_prompt = f"""
                You are a professional forecaster named Alice interviewing for a job at Great Judgement Inc.
                {self.forecaster_guidelines}

                    Your interview question is:
                    {question.question_text}

                    Question background:
                    {question.background_info}


                    This question's outcome will be determined by the specific criteria below.
                    {question.resolution_criteria}

                    {question.fine_print}

                    Today is {datetime.now().strftime("%Y-%m-%d")}.


                    You have been given an AI research assistant to help you forecast your question. This AI assistant can look for relevant news 
                    and provide an overview.  Ask for targeted queries that
                    gathers both important information and bird-eye views. Let it know the question outright. You won't be able to click on the links provided so ensure it gives a detailed 
                    overview of each link. More info is better than less. Focus first on crowd-sourced wisdom. 
                    Ensure that your researcher provides for each source its type (primary etc), credibility 
                    and timeliness.   
                    After thinking about how best to answer the question. 
                    provide a prompt to your AI asisstant. Put it between the <PROMPT> and </PROMPT> tags. 
            """
            instructions = await GeneralLlm(self.forecaster).invoke(forecaster_prompt)

            # Extract the prompt from between <PROMPT> and </PROMPT> tags
            prompt_match = re.search(r'<PROMPT>(.*?)</PROMPT>', instructions, re.DOTALL)
            extracted_prompt = prompt_match.group(1).strip() if prompt_match else instructions
            print(f""" 
                <RESEARCH>
                {extracted_prompt}
                </RESEARCH>
                  """
            )

            system_prompt = f"""
            You are assisting a research assistant to a superforecaster. Your job is to make queries and cite relevant
            information to the forecast. 
            Here are the specific instructions provided to you by the superforecaster:
            {extracted_prompt}
            """
            report = await self.call_perplexity(system_prompt, question.question_text)

            #searcher = SmartSearcher(
             #       model = GeneralLlm(self.researcher), 
             #       num_searches_to_run = 3,
             #       num_sites_per_search = 10)
            #report = await searcher.invoke(prompt)
            return report

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
            print(f"<Research> \n{research} \n </Research>")

            binary_format = """

                The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """

            initial_forecast_prompt = f"""
                You are a professional forecaster named Alice interviewing for a job.
                    {self.forecaster_guidelines}

                    Your interview question is:
                    {question.question_text}

                    Question background:
                    {question.background_info}


                    This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
                    {question.resolution_criteria}

                    {question.fine_print}

                    Today is {datetime.now().strftime("%Y-%m-%d")}.
                    
                    Here is the AI-provided research:
                    {research}

                    {binary_format}

            """

            forecaster = GeneralLlm(self.forecaster)
            reasoning = await forecaster.invoke(initial_forecast_prompt)
            binary_prediction: BinaryPrediction = await structure_output(
                reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
            )
            decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
            return ReasonedPrediction(
                prediction_value=decimal_pred, reasoning=reasoning
            )
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
                print(f"<Research> \n{research} \n </Research>")
                
                multiple_choices = ""
                for option in question.options:
                    multiple_choices+=f"{option} : Probability of {option}\n"

                multiple_choice_format = f"""

                IMPORTANT: All probabilities must be between 0.001 and 0.999 and sum to 1.0.

                End with your updated probabilities for the {len(question.options)} options in this order {question.options}:
                {multiple_choices}
                """

                initial_forecast_prompt = f"""
                    You are a professional forecaster named Alice interviewing for a job.
                        {self.forecaster_guidelines}

                        Your interview question is:
                        {question.question_text}

                        Question background:
                        {question.background_info}


                        This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
                        {question.resolution_criteria}

                        {question.fine_print}

                        Today is {datetime.now().strftime("%Y-%m-%d")}.

                        Here is the AI-provided research:
                        {research}

                        {multiple_choice_format}

                """


                forecaster = GeneralLlm(self.forecaster)
                reasoning = await forecaster.invoke(initial_forecast_prompt)
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
                return ReasonedPrediction(
                    prediction_value=predicted_option_list, reasoning=reasoning
                )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        print(f"<Research> \n{research} \n </Research>")

        lower_bound_message, upper_bound_message = self._create_upper_and_lower_bound_messages(question)
        numeric_choice_format = f"""
        {lower_bound_message}
        {upper_bound_message}

        Units for answer: {units}


        Formatting Instructions:
        - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
        - Never use scientific notation.
        - Always start with a smaller number (more negative if negative) and then increase from there

        You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unkowns.

        The last thing you write is your final answer as:
        "
        Percentile 10: XX
        Percentile 20: XX
        Percentile 40: XX
        Percentile 60: XX
        Percentile 80: XX
        Percentile 90: XX

        """

        initial_forecast_prompt = f"""
            You are a professional forecaster named Alice interviewing for a job.
                {self.forecaster_guidelines}

                Your interview question is:
                {question.question_text}

                Question background:
                {question.background_info}


                This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
                {question.resolution_criteria}

                {question.fine_print}

                Today is {datetime.now().strftime("%Y-%m-%d")}.

                Here is the AI-provided research:
                {research}

                {numeric_choice_format}

        """

        forecaster = GeneralLlm(self.forecaster)
        reasoning = await forecaster.invoke(initial_forecast_prompt)
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)

        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )
