def initial_prompt_one_descriptor_type(document):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

##Instruction:

You will be given a document. Your task is to create a comprehensive list of descriptors — words or phrases that distill the meaning, tone, style, genre, topics, and other characteristics of the document. Do not focus solely on the topics; also include phrases that describe the tone and style.
The descriptors you generate should be useful for two purposes:
    1. Rewriting the document in a way that is faithful to the original.
    2. Classifying the document in a large variety of NLP classification tasks.

##Requirements:

1. Generate Descriptors:
    - Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, language, quality, sentiment etc. Describe anything that you believe is essential for understanding, summarizing, or rewriting the document.
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
    - For each descriptor, explain its meaning and relevance to the document in 2 to 3 sentences.
    - The items in the specifics list do not need to be explained.


5. Format:
    - The output must be formatted as a JSON object. Follow this formatting exactly.
    - Never add any preamble or anything else outside the JSON object.
    - Separate the descriptors and their explanations with a semicolon ';'. Do NOT include a semicolon anywhere else in the descriptor.
    - Example: {{"descriptors": [
                                "<descriptor>; <explanation>",
                                "<descriptor>; <explanation>",
                                "<descriptor>; <explanation>",
                                ...],
                 "specifics": [
                              "<specifics>",
                              "<specifics>",
                              "<specifics>",
                              ...]
                }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{document}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def rewrite_prompt_one_descriptor_type(descriptors, specifics):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

##Instruction:

You will be given a list of descriptors and specifics:
    - Descriptors: Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, language, quality, sentiment, etc.
        - Each descriptor is followed by an explanation of its meaning and relevance to the document.
        - The descriptors are ordered in order of importance: the most important descriptors are listed first.
    - Specifics: Minute details such as individual words and phrases, emphasis, structure, etc. to help in rewriting the document.

  
##Your Task:

Reconstruct the original document as accurately as possible using both the descriptors and specifics provided. Do not describe the document; instead, produce the document itself.

##Requirements:

1. Utilize All Descriptors:
    - Incorporate all descriptors and specifics into your writing.
    - Ensure that the content, tone, style, and structure align with the descriptors.
    - The descriptors are ordered in order of importance: the most important descriptors are listed first.

2. Format:
    - The output must be formatted as a JSON object. Follow this formatting exactly.
    - Never add any preamble or anything else outside the JSON object.
    - Example: {{"text": "<rewritten version of the document>"}} <|eot_id|><|start_header_id|>user<|end_header_id|>

*Descriptors*:
<start>
{descriptors}
<end>

*Specifics*:
<start>
{specifics}
<end><|eot_id|><|start_header_id|>assistant<|end_header_id|>"""



def revise_keyphrases_prompt_one_descriptor_type(original_document, rewritten_document, descriptors, specifics):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

##Instruction:

You will be given:

1. Two Documents:
    - Original Document: The initial text.
    - Rewritten Document: An attempt to recreate the original document based solely on provided descriptors and specifics.

2. A List of Descriptors and Specifics (describing the original document):
    - Descriptors: Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, language, quality, sentiment, etc.
        - Each descriptor is followed by an explanation of its meaning and relevance to the document.
        - The descriptors are ordered in order of importance: the most important descriptors are listed first.
    - Specifics: Minute details such as individual words and phrases, emphasis, structure, etc. to help in rewriting the document.

##Your Task:

1. Critique the Rewritten Document:
    - Analyze the rewritten document.
    - Point out ways in which the rewrite differs from the original. Point out specific sentences or phrases that diverge meaningfully from the original, if applicable.
    - Focus on discrepancies in content, tone, style, structure, and any missing or misrepresented elements.

2. Revise the Descriptors and Specifics:
    - Adjust and revise the descriptors and specifics to help in creating a more faithful rewrite of the original document and for classifying it in a large variety of NLP classification tasks.
    - Each descriptor should describe only one aspect of the document. Multi-aspect descriptors should be split into separate descriptors.
    - The specifics list can contain individual words, phrases and other details that do not necessarily describe the document but are important for rewriting it.
    - The revised lists should not contain any direct or indirect references or comparisons to the original lists. They should stand on their own.
    - For each descriptor, explain its meaning and relevance to the document in 2 to 3 sentences. The specifics do not need to be explained.

##Requirements:

1. Specificity: Provide detailed observations in your critique.
2. Clarity: Ensure the revised descriptors are clear and precise.
3. Completeness: Cover all aspects where the rewritten document deviates from the original.
4. Conciseness: Do not make the descriptors overly long. They should typically not be longer than 3 words. Do not repeat descriptors and avoid near-duplicates.
5. Order: Order the descriptors in order of importance: the most important descriptors should be listed first.
6. Format:
    - The output must be formatted as a JSON object. Follow this formatting exactly.
    - Never add any preamble or anything else outside the JSON object.
    - Separate the descriptors and their explanations with a semicolon ';'. Do NOT include a semicolon anywhere else in the descriptor.
    - Example: {{"differences": "<differences between original and rewrite>",
                 "descriptors": [
                                "<descriptor>; <explanation>",
                                "<descriptor>; <explanation>",
                                "<descriptor>; <explanation>",
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
