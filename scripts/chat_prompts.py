def initial_prompt(document, vocab):
    return [{
              "role": "system",
              "content": f"""
##Instruction:

You will be given a document. Your task is to create a comprehensive list of descriptors—words or phrases that distill the meaning, tone, style, genre, topics, and other characteristics of the document. Do not focus solely on the topics; also include phrases that describe the tone and style. Order the descriptors as they appear in the document.

##Requirements:

1. Generate Two Types of Descriptors:
    - General Descriptors: General Descriptors: Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, length, language, quality, etc. They should be general enough so they could likely be applied to other, hypothetical documents.
    - Specific Descriptors: Describe minute details specific to this document such as individual words and phrases, emphasis, structure, etc.

2. Descriptor Details:
    - Descriptors can be single words or multi-word phrases. They should typically not be longer than 3 to 5 words.
    - Each descriptor should describe only one aspect of the document. Multi-aspect descriptors should be split into separate descriptors.
    - The list should be so extensive that, given only the list, a competent person should be able to rewrite the original document.

3. Order:
    - Sequence: Order the descriptors in the sequence they appear in the document.

4. Format:
    - The output must be formatted as a JSON object. Follow this formatting exactly. Never add any preamble.
    - Example: {{"general": [<"desciptor">, <"descriptor">, <"descriptor">, ...], "specific": [<"desciptor">, <"descriptor">, <"descriptor">, ...]}}

##Possible general descriptors
Here is a list of general descriptors that you can choose from. If the descriptors do not apply to the current document, you can also invent new ones.
<start>
{vocab}
<end>
"""  
            },
            {
              "role": "user",
              "content": f"{document}"
            }]
            

def rewrite_prompt(general, specific):
    return [{
              "role": "system",
              "content": """
##Instruction:

You will be given two lists of descriptors:
- General Descriptors: General Descriptors: Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, length, language, quality, etc.
- Specific Descriptors: Describe minute details specific to a document such as individual words and phrases, emphasis, structure, etc.

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
    - The output must be formatted as a JSON object. Follow this formatting exactly. Never add any preamble. Place double quotes around the rewritten document.
    - Example: {{"document": <"rewritten version of the document">}}
    """
            },
            {
              "role": "user",
              "content": f"""
*General Descriptors*:
<start>
{general}
<end>

*Specific Descriptors*:
<start>
{specific}
<end>
"""
            }]
    


def revise_keyphrases_prompt(original_document, rewritten_document, general, specific, vocab):
    return [{
              "role": "system",
              "content": """
##Instruction:

You will be given:

1. Two Documents:
    - Original Document: The initial text.
    - Rewritten Document: An attempt to recreate the original document based solely on provided descriptors.

2. Two Lists of Descriptors (describing the original document):
    - General Descriptors: Describe the document in aspects including, but not limited to, style, tone, genre, topic, domain, length, language, quality, etc.
    - Specific Descriptors: Describe minute details specific to this document such as individual words and phrases, emphasis, structure, etc.

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

##Requirements:

1. Specificity: Provide detailed observations in your critique.
2. Clarity: Ensure the revised descriptors are clear and precise. Make sure to differentiate between general and specific descriptors.
3. Completeness: Cover all aspects where the rewritten document deviates from the original.
4. Conciseness: Do not make the descriptors overly long. They should typically not be longer than 3 to 5 words.
5. Format:
    - The output must be formatted as a JSON object. Follow this formatting exactly. Place double quotes around the discussion about differences.
    - Example: {{"differences": <"differences between original and rewrite">, "general": [<"desciptor">, <"descriptor">, <"descriptor">, ...], "specific": [<"desciptor">, <"descriptor">, <"descriptor">, ...]}}

"""
            },
            {
              "role": "user",
              "content": f"""
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
<end>
"""
            }]


def reformat_output_prompt(original):
    return [{
              "role": "system",
              "content": """

###Instructions

You will be given a JSON string object that is not properly formatted. When given to json.loads() it results in an error. Your task is to fix the formatting of the string so that json.loads() can handle the input correctly.

Give as output only the corrected JSON string object without any preamble.
Do not change anything about the contents of the string, only fix the formatting issues.
If there is some preamble outside the JSON object, it can be deleted.
Remember to always use double quotation marks as single quotes are not valid JSON.
    """
            },
            {
              "role": "user",
              "content": f"""
{original}
"""
            }]   
