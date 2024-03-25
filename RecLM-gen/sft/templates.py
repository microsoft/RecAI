# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from fastchat.model import get_conversation_template


class SFTTemplate:
    def __init__(self, sys: str, first_turn: list[str], inst: str, output: str, input_fields, output_fields, template_id):
        self.sys = sys
        self.first_turn = first_turn
        self.input_template = inst
        self.output_template = output
        self.template_id = template_id
        self.task = template_id.split('-')[0]

        for _ in input_fields:
            for __ in _.split('/'):
                assert __ in ['item', 'item_count', 'history', 'preference', 'synthetic_intention', 'target_category', 'category_proportion', 'category_count', 'candidate_titles']
        for _ in output_fields:
            for __ in _.split('/'):
                assert __ in ['item', 'item_list', 'target_category']

        self.input_fields = input_fields
        self.output_fields = output_fields

        self.conv = get_conversation_template("llama-2")
        self.conv.set_system_message(self.sys)
        self.conv.append_message(self.conv.roles[0], self.first_turn[0])
        self.conv.append_message(self.conv.roles[1], self.first_turn[1])
        self.conv.append_message(self.conv.roles[0], '')
        self.conv.append_message(self.conv.roles[1], None)

    def get_input_text(self, input_args: dict, llama2_chat_template=False):
        for _ in self.input_fields:
            assert _ in input_args
        instruction = self.input_template.format_map(input_args)
        if not llama2_chat_template:
            input_text = f'{self.sys}{self.first_turn[0]}{instruction}'
        else:
            self.conv.messages[-2][1] = instruction
            input_text = self.conv.get_prompt()
        return input_text

    def get_output_text(self, output_args: dict):
        for _ in self.output_fields:
            assert _ in output_args
        return self.output_template.format_map(output_args)


SeqRec_task_key = 'SeqRec'
SeqRanking_task_key = 'SeqRanking'
ControlRec_task_key = 'ControlRec'
PersonalControlRec_task_key = 'PersonalControlRec'
CategoryRate_task_key = 'CategoryRate'
PersonalCategoryRate_task_key = 'PersonalCategoryRate'

################################################################################################################
#                                                                                                              #
#                                sequential recommendation templates                                           #
#                                                                                                              #
################################################################################################################


SeqRec_group = {}

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list considering user's preference from historical interactions. ",
                   "Ok, I will consider user's preference. "],
    'inst': "The historical interactions are provided as follows: {history}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['history', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{SeqRec_task_key}-{len(SeqRec_group)}",
}
SeqRec_group[f"{SeqRec_task_key}-{len(SeqRec_group)}"] = SFTTemplate(**template)

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to select a recommendation list considering user's preference from historical interactions. ",
                   "Ok, I will do it considering user's preference."],
    'inst': "The historical interactions are provided as follows: {history}. " +
            "The candidate items are: {candidate_titles}. " +
            "Please select a recommendation list with {item_count} different items from candidate items. ",
    'output': "{item_list}",
    'input_fields': ['history', 'item_count', 'candidate_titles'],
    'output_fields': ['item_list'],
    'template_id': f"{SeqRec_task_key}-{len(SeqRec_group)}",
}
SeqRec_group[f"{SeqRec_task_key}-{len(SeqRec_group)}"] = SFTTemplate(**template)


################################################################################################################
#                                                                                                              #
#                                          control rec templates                                               #
#                                                                                                              #
################################################################################################################


ControlRec_group = {}

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list complying user's intention. ",
                   "Ok, I will do it considering user's intention. "],
    'inst': "Here is user's intention: {synthetic_intention}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['synthetic_intention', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{ControlRec_task_key}-{len(ControlRec_group)}",
}
ControlRec_group[f"{ControlRec_task_key}-{len(ControlRec_group)}"] = SFTTemplate(**template)

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to select a recommendation list complying user's intention from candidate items. ",
                   "Ok, I will do it considering user's intention and candidate items. "],
    'inst': "Here is user's intention: {synthetic_intention}. " +
            "The candidate items are: {candidate_titles}. " +
            "Please select a recommendation list with {item_count} different items from candidate items. ",
    'output': "{item_list}",
    'input_fields': ['synthetic_intention', 'item_count', 'candidate_titles'],
    'output_fields': ['item_list'],
    'template_id': f"{ControlRec_task_key}-{len(ControlRec_group)}",
}
ControlRec_group[f"{ControlRec_task_key}-{len(ControlRec_group)}"] = SFTTemplate(**template)


ControlRec_re_group = {}

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list complying user's intention. ",
                   "Ok, I will do it considering user's intention. "],
    'inst': "Here is user's intention: {synthetic_intention}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['synthetic_intention', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{ControlRec_task_key}-{len(ControlRec_re_group)}",
}
ControlRec_re_group[f"{ControlRec_task_key}-{len(ControlRec_re_group)}"] = SFTTemplate(**template)

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to select a recommendation list complying user's intention from candidate items. ",
                   "Ok, I will do it considering user's intention and candidate items. "],
    'inst': "Here is user's intention: {synthetic_intention}. " +
            "The candidate items are: {candidate_titles}. " +
            "Please select a recommendation list with {item_count} different items from candidate items. ",
    'output': "{item_list}",
    'input_fields': ['synthetic_intention', 'item_count', 'candidate_titles'],
    'output_fields': ['item_list'],
    'template_id': f"{ControlRec_task_key}-{len(ControlRec_re_group)}",
}
ControlRec_re_group[f"{ControlRec_task_key}-{len(ControlRec_re_group)}"] = SFTTemplate(**template)

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to answer user's question. ",
                   "Ok, I will do it. "],
    'inst': "What is the category of \"{item}\"? ",
    'output': "The category of \"{item}\" is \"{target_category}\".",
    'input_fields': ['item'],
    'output_fields': ['item', 'target_category'],
    'template_id': f"{ControlRec_task_key}-{len(ControlRec_re_group)}",
}
ControlRec_re_group[f"{ControlRec_task_key}-reverse"] = SFTTemplate(**template)

################################################################################################################
#                                                                                                              #
#                                     personal control rec templates                                           #
#                                                                                                              #
################################################################################################################


PersonalControlRec_group = {}

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list simultaneously considering user's preference inferred from historical interactions and user's intention. " +
                   "If user's preference is conflict with his intention, you should comply with his intention. ",
                   "Ok, I will do it considering user's preference and intention. "],
    'inst': "Here is user's historical interactions: {history}, and user's intention: {synthetic_intention}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['history', 'synthetic_intention', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{PersonalControlRec_task_key}-{len(PersonalControlRec_group)}",
}
PersonalControlRec_group[f"{PersonalControlRec_task_key}-{len(PersonalControlRec_group)}"] = SFTTemplate(**template)

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': [
        "You need to select a recommendation list from candidate items simultaneously considering user's preference inferred from historical interactions and user's intention. " +
        "If user's preference is conflict with his intention, you should comply with his intention. ",
        "Ok, I will do it considering user's preference, intention and candidate items. "],
    'inst': "Here is user's historical interactions: {history}, and user's intention: {synthetic_intention}. " +
            "The candidate items are: {candidate_titles}. " +
            "Please select a recommendation list with {item_count} different items from candidate items. ",
    'output': "{item_list}",
    'input_fields': ['history', 'synthetic_intention', 'item_count', 'candidate_titles'],
    'output_fields': ['item_list'],
    'template_id': f"{PersonalControlRec_task_key}-{len(PersonalControlRec_group)}",
}
PersonalControlRec_group[f"{PersonalControlRec_task_key}-{len(PersonalControlRec_group)}"] = SFTTemplate(**template)


################################################################################################################
#                                                                                                              #
#                                             intention plus templates                                         #
#                                                                                                              #
################################################################################################################


Intention_plus_group = {}
template = {
    'sys': '',
    'first_turn': ['', ''],
    'inst': "I like '{target_category}' products",
    'output': "",
    'input_fields': ['target_category'],
    'output_fields': [],
    'template_id': f"IntentionPlus-{len(Intention_plus_group)}",
}
Intention_plus_group[f"IntentionPlus-{len(Intention_plus_group)}"] = SFTTemplate(**template)

template = {
    'sys': '',
    'first_turn': ['', ''],
    'inst': "Please recommend some '{target_category}' items",
    'output': "",
    'input_fields': ['target_category'],
    'output_fields': [],
    'template_id': f"IntentionPlus-{len(Intention_plus_group)}",
}
Intention_plus_group[f"IntentionPlus-{len(Intention_plus_group)}"] = SFTTemplate(**template)

template = {
    'sys': '',
    'first_turn': ['', ''],
    'inst': "I'm interested in '{target_category}'",
    'output': "",
    'input_fields': ['target_category'],
    'output_fields': [],
    'template_id': f"IntentionPlus-{len(Intention_plus_group)}",
}
Intention_plus_group[f"IntentionPlus-{len(Intention_plus_group)}"] = SFTTemplate(**template)

template = {
    'sys': '',
    'first_turn': ['', ''],
    'inst': "I would like to buy some '{target_category}' products",
    'output': "",
    'input_fields': ['target_category'],
    'output_fields': [],
    'template_id': f"IntentionPlus-{len(Intention_plus_group)}",
}
Intention_plus_group[f"IntentionPlus-{len(Intention_plus_group)}"] = SFTTemplate(**template)

template = {
    'sys': '',
    'first_turn': ['', ''],
    'inst': "I would like to browse some '{target_category}' products",
    'output': "",
    'input_fields': ['target_category'],
    'output_fields': [],
    'template_id': f"IntentionPlus-{len(Intention_plus_group)}",
}
Intention_plus_group[f"IntentionPlus-{len(Intention_plus_group)}"] = SFTTemplate(**template)

template = {
    'sys': '',
    'first_turn': ['', ''],
    'inst': "I prefer in '{target_category}' items",
    'output': "",
    'input_fields': ['target_category'],
    'output_fields': [],
    'template_id': f"IntentionPlus-{len(Intention_plus_group)}",
}
Intention_plus_group[f"IntentionPlus-{len(Intention_plus_group)}"] = SFTTemplate(**template)


################################################################################################################
#                                                                                                              #
#                                            intention minus templates                                         #
#                                                                                                              #
################################################################################################################


Intention_minus_group = {}
template = {
    'sys': '',
    'first_turn': ['', ''],
    'inst': "I don't like '{target_category}' products",
    'output': "",
    'input_fields': ['target_category'],
    'output_fields': [],
    'template_id': f"IntentionMinus-{len(Intention_minus_group)}",
}
Intention_minus_group[f"IntentionMinus-{len(Intention_minus_group)}"] = SFTTemplate(**template)

template = {
    'sys': '',
    'first_turn': ['', ''],
    'inst': "Please exclude any '{target_category}' item",
    'output': "",
    'input_fields': ['target_category'],
    'output_fields': [],
    'template_id': f"IntentionMinus-{len(Intention_minus_group)}",
}
Intention_minus_group[f"IntentionMinus-{len(Intention_minus_group)}"] = SFTTemplate(**template)

template = {
    'sys': '',
    'first_turn': ['', ''],
    'inst': "I'm not interested in '{target_category}'",
    'output': "",
    'input_fields': ['target_category'],
    'output_fields': [],
    'template_id': f"IntentionMinus-{len(Intention_minus_group)}",
}
Intention_minus_group[f"IntentionMinus-{len(Intention_minus_group)}"] = SFTTemplate(**template)

template = {
    'sys': '',
    'first_turn': ['', ''],
    'inst': "Don't recommend me any '{target_category}' products",
    'output': "",
    'input_fields': ['target_category'],
    'output_fields': [],
    'template_id': f"IntentionMinus-{len(Intention_minus_group)}",
}
Intention_minus_group[f"IntentionMinus-{len(Intention_minus_group)}"] = SFTTemplate(**template)

template = {
    'sys': '',
    'first_turn': ['', ''],
    'inst': "I don't want to browse any '{target_category}' product",
    'output': "",
    'input_fields': ['target_category'],
    'output_fields': [],
    'template_id': f"IntentionMinus-{len(Intention_minus_group)}",
}
Intention_minus_group[f"IntentionMinus-{len(Intention_minus_group)}"] = SFTTemplate(**template)

template = {
    'sys': '',
    'first_turn': ['', ''],
    'inst': "I hate '{target_category}' items",
    'output': "",
    'input_fields': ['target_category'],
    'output_fields': [],
    'template_id': f"IntentionMinus-{len(Intention_minus_group)}",
}
Intention_minus_group[f"IntentionMinus-{len(Intention_minus_group)}"] = SFTTemplate(**template)


################################################################################################################
#                                                                                                              #
#                                            category rate templates                                           #
#                                                                                                              #
################################################################################################################


CategoryRate_group = {}

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list complying user's intention. ",
                   "Ok, I will do it considering user's intention. "],
    'inst': "Here is user's intention: I like '{target_category}' items, " +
            "but in the recommendation list, the proportion of '{target_category}' items should be less than {category_proportion}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['target_category', 'category_proportion', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{CategoryRate_task_key}-LP",
}
CategoryRate_group[f"{CategoryRate_task_key}-LP"] = SFTTemplate(**template)


template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list complying user's intention. ",
                   "Ok, I will do it considering user's intention. "],
    'inst': "Here is user's intention: I don't like '{target_category}' items, " +
            "but in the recommendation list, the proportion of '{target_category}' items should be more than {category_proportion}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['target_category', 'category_proportion', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{CategoryRate_task_key}-MP",
}
CategoryRate_group[f"{CategoryRate_task_key}-MP"] = SFTTemplate(**template)


template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list complying user's intention. ",
                   "Ok, I will do it considering user's intention. "],
    'inst': "Here is user's intention: I like '{target_category}' items, " +
            "but the recommendation list should containing less than {category_count} '{target_category}' items. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['item_count', 'category_count', 'target_category'],
    'output_fields': ['item_list'],
    'template_id': f"{CategoryRate_task_key}-LC",
}
CategoryRate_group[f"{CategoryRate_task_key}-LC"] = SFTTemplate(**template)


template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list complying user's intention. ",
                   "Ok, I will do it considering user's intention. "],
    'inst': "Here is user's intention: I don't like '{target_category}' items, " +
            "but the recommendation list should containing more than {category_count} '{target_category}' items. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['item_count', 'category_count', 'target_category'],
    'output_fields': ['item_list'],
    'template_id': f"{CategoryRate_task_key}-MC",
}
CategoryRate_group[f"{CategoryRate_task_key}-MC"] = SFTTemplate(**template)


################################################################################################################
#                                                                                                              #
#                                       personal category rate templates                                       #
#                                                                                                              #
################################################################################################################


PersonalCategoryRate_group = {}

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list simultaneously considering user's preference inferred from historical interactions and user's intention. ",
                   "Ok, I will do it considering user's preference and intention. "],
    'inst': "Here is user's historical interactions: {history}, and user's intention: I like '{target_category}' items, " +
            "but in the recommendation list, the proportion of '{target_category}' items should be less than {category_proportion}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['target_category', 'category_proportion', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{PersonalCategoryRate_task_key}-LP",
}
PersonalCategoryRate_group[f"{PersonalCategoryRate_task_key}-LP"] = SFTTemplate(**template)


template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list simultaneously considering user's preference inferred from historical interactions and user's intention. ",
                   "Ok, I will do it considering user's preference and intention. "],
    'inst': "Here is user's historical interactions: {history}, and user's intention: I don't like '{target_category}' items, " +
            "but in the recommendation list, the proportion of '{target_category}' items should be more than {category_proportion}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['target_category', 'category_proportion', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{PersonalCategoryRate_task_key}-MP",
}
PersonalCategoryRate_group[f"{PersonalCategoryRate_task_key}-MP"] = SFTTemplate(**template)


template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list simultaneously considering user's preference inferred from historical interactions and user's intention. ",
                   "Ok, I will do it considering user's preference and intention. "],
    'inst': "Here is user's historical interactions: {history}, and user's intention: I like '{target_category}' items, " +
            "but the recommendation list should containing less than {category_count} '{target_category}' items. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['item_count', 'category_count', 'target_category'],
    'output_fields': ['item_list'],
    'template_id': f"{PersonalCategoryRate_task_key}-LC",
}
PersonalCategoryRate_group[f"{PersonalCategoryRate_task_key}-LC"] = SFTTemplate(**template)


template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list simultaneously considering user's preference inferred from historical interactions and user's intention. ",
                   "Ok, I will do it considering user's preference and intention. "],
    'inst': "Here is user's historical interactions: {history}, and user's intention: I don't like '{target_category}' items, " +
            "but the recommendation list should containing more than {category_count} '{target_category}' items. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['item_count', 'category_count', 'target_category'],
    'output_fields': ['item_list'],
    'template_id': f"{PersonalCategoryRate_task_key}-MC",
}
PersonalCategoryRate_group[f"{PersonalCategoryRate_task_key}-MC"] = SFTTemplate(**template)


################################################################################################################
#                                                                                                              #
#                                         val and test templates                                               #
#                                                                                                              #
################################################################################################################


TestPersonalCategoryRateLP_group = {}

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': [
        "You need to generate a recommendation list simultaneously considering user's preference from historical interactions, and user's intention. ",
        "Ok, I will consider user's preference and intention. "],
    'inst': "Here is user's historical interactions: {history}, and user's intention: I like '{target_category}' items, " +
            "but in the recommendation list, the proportion of '{target_category}' items should be less than {category_proportion}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['history', 'target_category', 'category_proportion', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{PersonalCategoryRate_task_key}",
}
TestPersonalCategoryRateLP_group[f"{PersonalCategoryRate_task_key}"] = SFTTemplate(**template)

TestPersonalCategoryRateLP1_group = {}

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': [
        "You need to generate a recommendation list simultaneously considering user's preference from historical interactions, and user's intention. ",
        "Ok, I will consider user's preference and intention. "],
    'inst': "Here is user's historical interactions: {history}, and user's intention: In the recommendation list, "
            "the proportion of '{target_category}' items should be less than {category_proportion}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['history', 'target_category', 'category_proportion', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{PersonalCategoryRate_task_key}",
}
TestPersonalCategoryRateLP1_group[f"{PersonalCategoryRate_task_key}"] = SFTTemplate(**template)


TestPersonalCategoryRateMP_group = {}

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': [
        "You need to generate a recommendation list simultaneously considering user's preference from historical interactions, and user's intention. ",
        "Ok, I will consider user's preference and intention. "],
    'inst': "Here is user's historical interactions: {history}, and user's intention: " +
            "In the recommendation list, the proportion of '{target_category}' items should be more than {category_proportion}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['history', 'target_category', 'category_proportion', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{PersonalCategoryRate_task_key}",
}
TestPersonalCategoryRateMP_group[f"{PersonalCategoryRate_task_key}"] = SFTTemplate(**template)


TestPersonalCategoryRateEP_group = {}

template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': [
        "You need to generate a recommendation list simultaneously considering user's preference from historical interactions, and user's intention. ",
        "Ok, I will consider user's preference and intention. "],
    'inst': "Here is user's historical interactions: {history}, and user's intention: " +
            "In the recommendation list, the proportion of '{target_category}' items should be approximately {category_proportion}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['history', 'target_category', 'category_proportion', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{PersonalCategoryRate_task_key}",
}
TestPersonalCategoryRateEP_group[f"{PersonalCategoryRate_task_key}"] = SFTTemplate(**template)


ValSeqRec_group = {}
template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list considering user's preference from historical interactions. ",
                   "Ok, I will consider user's preference. "],
    'inst': "The historical interactions are provided as follows: {history}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['history', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{SeqRec_task_key}-{len(ValSeqRec_group)}",
}
ValSeqRec_group[f"{SeqRec_task_key}-{len(ValSeqRec_group)}"] = SFTTemplate(**template)

ValSeqRanking_group = {}
template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to select a recommendation list complying user's intention from candidate items. ",
                   "Ok, I will do it considering user's preference and candidate items. "],
    'inst': "The historical interactions are provided as follows: {history}. " +
            "The candidate items are: {candidate_titles}. " +
            "Please select a recommendation list with {item_count} different items from candidate items. ",
    'output': "{item_list}",
    'input_fields': ['history', 'item_count', 'candidate_titles'],
    'output_fields': ['item_list'],
    'template_id': f"{SeqRanking_task_key}-{len(ValSeqRanking_group)}",
}
ValSeqRanking_group[f"{SeqRanking_task_key}-{len(ValSeqRanking_group)}"] = SFTTemplate(**template)


ValControlRec_group = {}
template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list complying user's intention. ",
                   "Ok, I will do it considering user's intention. "],
    'inst': "Here is user's intention: {synthetic_intention}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['synthetic_intention', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{ControlRec_task_key}-{len(ValControlRec_group)}",
}
ValControlRec_group[f"{ControlRec_task_key}-{len(ValControlRec_group)}"] = SFTTemplate(**template)

ValControlRecRanking_group = {}
template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to select a recommendation list complying user's intention from candidate items. ",
                   "Ok, I will do it considering user's intention and candidate items. "],
    'inst': "Here is user's intention: {synthetic_intention}. " +
            "The candidate items are: {candidate_titles}. " +
            "Please select a recommendation list with {item_count} different items from candidate items. ",
    'output': "{item_list}",
    'input_fields': ['synthetic_intention', 'item_count', 'candidate_titles'],
    'output_fields': ['item_list'],
    'template_id': f"{ControlRec_task_key}-{len(ValControlRecRanking_group)}",
}
ValControlRecRanking_group[f"{ControlRec_task_key}-{len(ValControlRecRanking_group)}"] = SFTTemplate(**template)


ValPersonalControlRec_group = {}
template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': ["You need to generate a recommendation list simultaneously considering user's preference inferred from historical interactions and user's intention. " +
                   "If user's preference is conflict with his intention, you should comply with his intention. ",
                   "Ok, I will do it considering user's preference and intention. "],
    'inst': "Here is user's historical interactions: {history}, and user's intention: {synthetic_intention}. " +
            "Please generate a recommendation list with {item_count} different items. ",
    'output': "{item_list}",
    'input_fields': ['history', 'synthetic_intention', 'item_count'],
    'output_fields': ['item_list'],
    'template_id': f"{PersonalControlRec_task_key}-{len(ValPersonalControlRec_group)}",
}
ValPersonalControlRec_group[f"{PersonalControlRec_task_key}-{len(ValPersonalControlRec_group)}"] = SFTTemplate(**template)

ValPersonalControlRecRanking_group = {}
template = {
    'sys': "You are an expert recommender engine. ",
    'first_turn': [
        "You need to generate a recommendation list from candidate items simultaneously considering user's preference inferred from historical interactions and user's intention. " +
        "If user's preference is conflict with his intention, you should comply with his intention. ",
        "Ok, I will do it considering user's preference, intention and candidate items. "],
    'inst': "Here is user's historical interactions: {history}, and user's intention: {synthetic_intention}. " +
            "The candidate items are: {candidate_titles}. " +
            "Please select a recommendation list with {item_count} different items from candidate items. ",
    'output': "{item_list}",
    'input_fields': ['history', 'synthetic_intention', 'item_count', 'candidate_titles'],
    'output_fields': ['item_list'],
    'template_id': f"{PersonalControlRec_task_key}-{len(ValPersonalControlRecRanking_group)}",
}
ValPersonalControlRecRanking_group[f"{PersonalControlRec_task_key}-{len(ValPersonalControlRecRanking_group)}"] = SFTTemplate(**template)


if __name__ == '__main__':
    pass
