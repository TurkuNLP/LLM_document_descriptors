def initial_prompt(document, vocab):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

##Instruction:

You will be given a document. Your task is to create a comprehensive list of descriptors — words or phrases that distill the meaning, tone, style, genre, topics, and other characteristics of the document. Do not focus solely on the topics; also include phrases that describe the tone and style. Order the descriptors as they appear in the document.

##Requirements:

1. Generate Two Types of Descriptors:
    - General Descriptors: Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, length, language, quality, etc. They should be general enough so they could likely be applied to other, hypothetical documents.
    - Specific Descriptors: Describe minute details specific to this document such as individual words and phrases, emphasis, structure, etc.

2. Descriptor Details:
    - Descriptors can be single words or multi-word phrases. They should typically not be longer than 3 words.
    - Each descriptor should describe only one aspect of the document. Multi-aspect descriptors should be split into separate descriptors.
    - The list should be so extensive that, given only the list, a competent person should be able to rewrite the original document.
    - Do not repeat descriptors.

3. Order:
    - Sequence: Order the descriptors in the sequence they appear in the document.
    
4. Descriptor Explanations:
    - For each descriptor, explain its meaning and relevance to the document in in 2 to 3 sentences.

5. Format:
    - The output must be formatted as a JSON object. Follow this formatting exactly.
    - Never add any preamble or anything else outside the JSON object.
    - Separate the descriptors and their explanations with a colon ':'.
    - Example: {{"general": [
                            "<descriptor>: <explanation>",
                            "<descriptor>: <explanation>",
                            "<descriptor>: <explanation>",
                            ...],
                 "specific": [
                             "<descriptor>: <explanation>",
                             "<descriptor>: <explanation>",
                             "<descriptor>: <explanation>",
                             ...]
                }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{document}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def initial_prompt_one_descriptor_type(document, vocab):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

##Instruction:

You will be given a document. Your task is to create a comprehensive list of descriptors — words or phrases that distill the meaning, tone, style, genre, topics, and other characteristics of the document. Do not focus solely on the topics; also include phrases that describe the tone and style. Order the descriptors as they appear in the document.

##Requirements:

1. Generate Descriptors:
    - Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, length, language, quality, etc. Describe anything that you believe is essential for understanding, summarizing, or rewriting the document.
    - If necessary, you can add minute details into the "specifics" list. This list can contain individual words, phrases and other details that do not necessarily describe the document but are important for rewriting it. 

2. Descriptor Details:
    - Descriptors can be single words or multi-word phrases. They should typically not be longer than 3 words.
    - Each descriptor should describe only one aspect of the document. Multi-aspect descriptors should be split into separate descriptors.
    - The list should be so extensive that, given only the list, a competent person should be able to rewrite the original document.
    - Do not repeat descriptors and avoid near-duplicates.

3. Order:
    - Order the descriptors in order of importance: the most important descriptors should be listed first.
    - Importance is determined by how essential the descriptor is for reconstructing, understanding, or summarizing the document.
    
4. Descriptor Explanations:
    - For each descriptor, explain its meaning and relevance to the document in in 2 to 3 sentences.
    - The items in the specifics list do not need to be explained.
    - The descriptors themselves should be clear and descriptive enough to be understood even without the explanations. The explanations are there only for additional context.

5. Format:
    - The output must be formatted as a JSON object. Follow this formatting exactly.
    - Never add any preamble or anything else outside the JSON object.
    - Separate the descriptors and their explanations with a colon ':'.
    - Example: {{"descriptors": [
                                "<descriptor>: <explanation>",
                                "<descriptor>: <explanation>",
                                "<descriptor>: <explanation>",
                                ...],
                 "specifics": [
                              "<specifics>",
                              "<specifics>",
                              "<specifics>",
                              ...]
                }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{document}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def rewrite_prompt(general, specific):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

##Instruction:

You will be given two lists of descriptors:
    - General Descriptors: Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, length, language, quality, etc.
    - Specific Descriptors: Describe minute details specific to a document such as individual words and phrases, emphasis, structure, etc.
    - Each descriptor is followed by an explanation of its meaning and relevance to the document.

##Your Task:

Reconstruct the original document as accurately as possible using both the general and specific descriptors provided. Do not describe the document; instead, produce the document itself.

##Requirements:

1. Utilize All Descriptors:
    - Incorporate both general and specific descriptors into your writing.
    - Ensure that the content, tone, style, and structure align with the descriptors.

2. Original Content:
    - The rewritten document should be original and not copied from any existing source.
    - Maintain coherence and cohesiveness in the document.

3. Format:
    - The output must be formatted as a JSON object. Follow this formatting exactly.
    - Never add any preamble or anything else outside the JSON object.
    - Example: {{"document": "<rewritten version of the document>"}} <|eot_id|><|start_header_id|>user<|end_header_id|>

*General Descriptors*:
<start>
{general}
<end>

*Specific Descriptors*:
<start>
{specific}
<end><|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def rewrite_prompt_one_descriptor_type(descriptors, specifics):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

##Instruction:

You will be given a list of descriptors and specifics:
    - Descriptors: Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, length, language, quality, etc.
        - Each descriptor is followed by an explanation of its meaning and relevance to the document.
        - The descriptors are ordered in order of importance: the most important descriptors are listed first.
    - Specifics: Minute details such as individual words and phrases, emphasis, structure, etc. to help in rewriting the document.

  
##Your Task:

Reconstruct the original document as accurately as possible using both the descriptors and specifics provided. Do not describe the document; instead, produce the document itself.

##Requirements:

1. Utilize All Descriptors:
    - Incorporate all descriptors and specifics into your writing.
    - Ensure that the content, tone, style, and structure align with the descriptors.

2. Format:
    - The output must be formatted as a JSON object. Follow this formatting exactly.
    - Never add any preamble or anything else outside the JSON object.
    - Example: {{"document": "<rewritten version of the document>"}} <|eot_id|><|start_header_id|>user<|end_header_id|>

*Descriptors*:
<start>
{descriptors}
<end>

*Specifics*:
<start>
{specifics}
<end><|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def revise_keyphrases_prompt(original_document, rewritten_document, general, specific, vocab):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

##Instruction:

You will be given:

1. Two Documents:
    - Original Document: The initial text.
    - Rewritten Document: An attempt to recreate the original document based solely on provided descriptors.

2. Two Lists of Descriptors (describing the original document):
    - General Descriptors: Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, length, language, quality, etc.
    - Specific Descriptors: Describe minute details specific to this document such as individual words and phrases, emphasis, structure, etc.
    - Each descriptor is followed by an explanation of its meaning and relevance to the document.

##Your Task:

1. Critique the Rewritten Document:
    - Analyze the rewritten document.
    - Point out specific ways in which it differs from the original document.
    - Focus on discrepancies in content, tone, style, structure, and any missing or misrepresented elements.

2. Revise the Descriptors:
    - Create new, revised versions of the general and specific descriptors.
    - General descriptors should be general enough so they could likely be applied to other, hypothetical documents.
    - Each descriptor should describe only one aspect of the document. Multi-aspect descriptors should be split into separate descriptors.
    - Adjust the descriptors to help in creating a more faithful rewrite of the original document.
    - The revised lists should not contain any direct or indirect references or comparisons to the original lists. They should stand on their own.
    - For each descriptor, explain its meaning and relevance to the document in 2 to 3 sentences.

##Requirements:

1. Specificity: Provide detailed observations in your critique.
2. Clarity: Ensure the revised descriptors are clear and precise. Make sure to differentiate between general and specific descriptors.
3. Completeness: Cover all aspects where the rewritten document deviates from the original.
4. Conciseness: Do not make the descriptors overly long. They should typically
not be longer than 3 to 5 words. Do not repeat descriptors.
5. Format:
    - The output must be formatted as a JSON object. Follow this formatting exactly.
    - Never add any preamble or anything else outside the JSON object.
    - Separete the descriptors and their explanations with a colon ':'.
    - Example: {{"differences": "<differences between original and rewrite>",
                 "general": [
                            "<descriptor>: <explanation>",
                            "<descriptor>: <explanation>",
                            "<descriptor>: <explanation>",
                            ...],
                 "specific": [
                             "<descriptor>: <explanation>",
                             "<descriptor>: <explanation>",
                             "<descriptor>: <explanation>",
                             ...]
                }} <|eot_id|><|start_header_id|>user<|end_header_id|>
                
*Original Document*:
<start>
{original_document}
<end>

*Rewritten Document*:
<start>
{rewritten_document}
<end>

*General Descriptors*:
<start>
{general}
<end>

*Specific Descriptors*:
<start>
{specific}
<end><|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def revise_keyphrases_prompt_one_descriptor_type(original_document, rewritten_document, descriptors, specifics, vocab):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

##Instruction:

You will be given:

1. Two Documents:
    - Original Document: The initial text.
    - Rewritten Document: An attempt to recreate the original document based solely on provided descriptors and specifics.

2. A List of Descriptors and Specifics (describing the original document):
    - Descriptors: Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, length, language, quality, etc.
        - Each descriptor is followed by an explanation of its meaning and relevance to the document.
        - The descriptors are ordered in order of importance: the most important descriptors are listed first.
    - Specifics: Minute details such as individual words and phrases, emphasis, structure, etc. to help in rewriting the document.

##Your Task:

1. Critique the Rewritten Document:
    - Analyze the rewritten document.
    - Point out ways in which the rewrite differs from the original. Point out specific sentences or phrases that diverge meaningfully from the original, if applicable.
    - Focus on discrepancies in content, tone, style, structure, and any missing or misrepresented elements.

2. Revise the Descriptors and Specifics:
    - Adjust and revise the descriptors and specifics to guide a better, more faithful reconstruction of the original document.
    - Each descriptor should describe only one aspect of the document. Multi-aspect descriptors should be split into separate descriptors.
    - The specifics list can contain individual words, phrases and other details that do not necessarily describe the document but are important for rewriting it.
    - The revised lists should not contain any direct or indirect references or comparisons to the original lists. They should stand on their own.
    - For each descriptor, explain its meaning and relevance to the document in 2 to 3 sentences. The specifics do not need to be explained.
    - The descriptors themselves should be clear and descriptive enough to be understood even without the explanations. The explanations are there only for additional context.

##Requirements:

1. Specificity: Provide detailed observations in your critique.
2. Clarity: Ensure the revised descriptors are clear and precise.
3. Completeness: Cover all aspects where the rewritten document deviates from the original.
4. Conciseness: Do not make the descriptors overly long. They should typically not be longer than 3 words. Do not repeat descriptors and avoid near-duplicates.
5. Order: Order the descriptors in order of importance: the most important descriptors should be listed first.
6. Format:
    - The output must be formatted as a JSON object. Follow this formatting exactly.
    - Never add any preamble or anything else outside the JSON object.
    - Separete the descriptors and their explanations with a colon ':'.
    - Example: {{"differences": "<differences between original and rewrite>",
                 "descriptors": [
                                "<descriptor>: <explanation>",
                                "<descriptor>: <explanation>",
                                "<descriptor>: <explanation>",
                                ...],
                 "specifics": [
                              "<specifics>",
                              "<specifics>",
                              "<specifics>",
                              ...]
                }}<|eot_id|><|start_header_id|>user<|end_header_id|>
                
*Original Document*:
<start>
{original_document}
<end>

*Rewritten Document*:
<start>
{rewritten_document}
<end>

*Descriptors*:
<start>
{descriptors}
<end>

*Specifics*:
<start>
{specifics}
<end><|eot_id|><|start_header_id|>assistant<|end_header_id|>"""



def review_synonyms(group_name, synonyms):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You will be given a list of candidate words that might be synonyms. The words describe the contents of text documents.
Your task is to analyze the list and decide whether all words are sufficiently similar in meaning to form one coherent group and match the name of the group or if they should be divided into multiple groups. Please follow these instructions:

1. **Definition of Synonymy and Coherence:**
   - Words do not have to be perfectly synonymous, but they should share a significant overlap in meaning, usage, or connotation to be considered part of the same group.
   - A coherent group is one in which every word is contextually or semantically similar, even if subtle differences exist.

2. **Handling Ambiguity:**
   - Some words have multiple meanings. Use the most common sense or the sense that best fits with the other words in the list.
   - If a word has multiple plausible senses, choose the one that aligns with a potential group or consider it an outlier if no clear fit exists.

3. **Grouping Requirements:**
   - Group words such that each group is as large as possible without including words that do not fit together.
   - Avoid creating more groups than necessary. Only create a new group if one or more words clearly do not belong in an otherwise coherent group.
   - Outlier words should form their own individual groups if they do not match any of the other words.
   
4. **Naming the Groups:**
   - The name of the group can be one of the words in the group or it can be a separate term.
   - If the given name for the group does not represent the contents of the group, you should change it.
   - The name of the group should be central to the meaning of the group, i.e., all words in the group should be synonymous with the name. 

4. **Output Format:**
   - Present your results as a list of groups. Each group should be named.
   - Do not modify the words in the groups any way. Each word must appear in exactly one group.
   - Your output should be in JSON format. Do not add any text or explanation outside the JSON object.
    - Example: {{"<group_name>": [<"word">,
                                 <"word">,
                                 <"word">,
                                 ...],
                 "<group_name>": [<"word">,
                                 <"word">,
                                 <"word">,
                                 ...]
                }}<|eot_id|><|start_header_id|>user<|end_header_id|>
                
Now, here are the possible synonyms:
Group name: {group_name}
Words in group:{synonyms}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""



def reformat_output_prompt(invalid_json):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

###Instructions

You will be provided with a JSON object that contains formatting errors, making it invalid. Your task is to correct the JSON string so that it can be successfully parsed in Python using json.loads().

- Do not alter the content or structure of the data within the JSON, except as necessary to fix formatting issues (e.g., missing commas, unmatched brackets, incorrect quotes).
- If there are more than 50 items in the JSON object, you can truncate it.
- If there is any extraneous text or preamble outside the JSON object, remove it entirely.
- Ensure the corrected output adheres strictly to JSON standards.
- Output only the corrected JSON string—do not include any explanations, commentary, or additional text.<|eot_id|><|start_header_id|>user<|end_header_id|>
    
{invalid_json}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
