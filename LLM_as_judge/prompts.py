import json


def get_query_correspondence_prompt(document, query):
    return [
        {
            "role": "system",
            "content": f"""You are a helpful assistant.
            
            ### Task
            You are a judge. Your job is to decide whether a given document corresponds to a user query.
            
            The query does not have to be explicitly stated in the document. The goal is to determine if the query would be an accurate reflection of the document's content.
            
            ### Output options and criteria
            - Yes: The document corresponds to the query completely or to a significant extent.
            - Partial: The document corresponds to the query to some extent, but there are some gaps or mismatches. 
            - No: The document does not correspond to the query or the connection is unclear or very minor.
            
            ### Output format
            First, briefly explain your reasoning.
            Then end with exactly one of the following lines:

            ANSWER: Yes

            or
            
            ANSWER: Partial

            or
            
            ANSWER: No
            """
        },
        {
            "role": "user",
            "content": f"Query: {query}\nDocument: {document}",
        },
    ]


def get_descriptor_correspondence_prompt(query, descriptor):
    return [
        {
            "role": "system",
            "content": f"""You are a helpful assistant.

            ### Task
            You are a judge. Your job is to decide whether a document descriptor is relevant to a user query.

            The descriptor describes one aspect of a document. The goal is to use the descriptor to identify documents that are likely to be relevant to the query.

            Decide whether the descriptor provides strong evidence that the document matches the query.

            ### Relevance criteria
            Answer "Yes" only if:
            - The descriptor directly matches the query, or
            - The descriptor clearly matches a specific part of the query, or
            - The descriptor is not explicitly mentioned in the query but is a necessary or strongly implied part of what the query asks for.

            Answer "No" if:
            - The descriptor is unrelated to the query,
            - The descriptor is only generally related to the query topic,
            - The descriptor could apply to many documents that would not satisfy the query,
            - The connection is vague, indirect, or speculative,
            - The descriptor reflects a broad category but the query asks for something more specific, or
            - The descriptor matches only background context rather than the actual information need.

            Do not answer "Yes" just because the descriptor and query are in the same general domain. A partial match is enough only when it is specific and clearly relevant to the user’s query.

            When uncertain, answer "No".

            ### Output options
            - Yes: The descriptor is clearly relevant to the query.
            - No: The descriptor is not clearly relevant to the query.

            ### Output format
            First, briefly explain your reasoning.
            Then end with exactly one of the following lines:

            ANSWER: Yes

            or

            ANSWER: No
            """,
        },
        {
            "role": "user",
            "content": f"Query: {query}\nDescriptor: {descriptor}",
        },
    ]


def get_descriptor_accuracy_prompt(document, descriptor):
    return [
        {
            "role": "system",
            "content": """You are a helpful assistant.

    ###Instructions

    You are given a document and a descriptor that is supposed to describe the document.
    Your task is to evaluate the accuracy of the descriptor in describing the document.
    The descriptor is one of many and is not meant to be a comprehensive description of the whole document, but it should be accurate in what it does describe.
    For example a descriptor that states that the tone of the document is humorous can be accurate even if the document also contains some serious parts, as long as the descriptor is accurate in describing the humorous parts of the document.

    ###Output options
    - Accurate: The descriptor is fully accurate in describing the document.
    - Mostly accurate: The descriptor is mostly accurate in describing the document, but there may be some minor inaccuracies.
    - Partially accurate: The descriptor is partially accurate in describing the document, but there are significant misleading inaccuracies.
    - Mostly inaccurate: The descriptor is mostly inaccurate in describing the document, but there may be some minor accurate elements.
    - Inaccurate: The descriptor is completely inaccurate and does not describe the document at all.

    ###Output format
    Your answer should be one of the following options: Accurate, Mostly accurate, Partially accurate, Mostly inaccurate, Inaccurate.
    First, provide a short justification for your answer by explaining the reasoning behind it. Then, provide a clear and concise answer to the question.
    Your answer should end with "ANSWER: " followed by one of the options listed above.
    """,
        },
        {
            "role": "user",
            "content": f"Descriptor: {descriptor}\nDocument: {document}",
        },
    ]

def get_descriptors2label_correspondence_prompt(descriptors, label):
    return [
        {
            "role": "system",
            "content": """You are a helpful assistant.

    ###Instructions

    You are given a list of descriptors and a label. The descriptors and label are all related to the same document. The descriptors describe the document in various ways, and the label is a single word or phrase that categorizes the document.
    Your task is to evaluate whether the descriptors correspond to the label. In other words, you need to determine if the label can be inferred from the descriptors. Think of the question as "Given just the descriptors, would it be reasonable to label the document with this label?".
    Not all descriptors need to be relevant to the label, but at least one descriptor should be relevant for the label to be considered corresponding to the descriptors.


    ###Output options
    - Yes: One or more of the descriptors are relevant to the label.
    - No: None of the descriptors are clearly relevant to the label and the label cannot be inferred from them.

    ###Output format
    First, briefly explain your reasoning.
    Then end with exactly one of the following lines:

    ANSWER: Yes

    or

    ANSWER: No
    """,
        },
        {
            "role": "user",
            "content": f"Label: {label}\nDescriptors: {', '.join(descriptors)}",
        },
    ]
    
def get_label2document_correspondence_prompt(document, label, label_type = "format"):
    # label_type can be "format" or "topic".
    return [
        {
            "role": "system",
            "content": f"""You are a helpful assistant.

    ###Instructions

    You are given a document and a label. The label describes the {label_type} of the document. The label is a single word or phrase that categorizes the document.
    Your task is to evaluate whether the label corresponds to the document. In other words, you need to determine if the label accurately describes the {label_type} of the document.
    The label should be relevant to the main {label_type} of the document for it to be considered corresponding.


    ###Output options
    - Yes: The label accurately describes the {label_type} of the document.
    - No: The label does not accurately describe the {label_type} of the document.

    ###Output format
    First, briefly explain your reasoning.
    Then end with exactly one of the following lines:

    ANSWER: Yes

    or

    ANSWER: No
    """,
        },
        {
            "role": "user",
            "content": f"Label: {label}\nDocument: {document}",
        },
    ]


def get_infer_labels_from_descriptors_prompt(descriptors, labels, label_type = "format"):
    return [
        {
            "role": "system",
            "content": f"""You are a helpful assistant.

###Instructions

You are given a list of descriptors and list of candidate labels.
The descriptors describe a document in various ways, and the labels categorize the {label_type} of the document.
Your task is to determine, which of the candidate labels could reasonably be applied to the document that the descriptors describe.
Choose all labels that are directly supported or clearly implied by the descriptors.
Do not invent new labels.

###Output format
Return only a JSON array of exact label strings taken from the candidate list.
Use the same casing as the candidate labels.
First, briefly explain your reasoning.
Then end with the exact string "ANSWER: " followed by a comma-separated list of labels.
""",
        },
        {
            "role": "user",
            "content": f"Candidate labels: {json.dumps(labels, ensure_ascii=False)}\nDescriptors: {json.dumps(descriptors, ensure_ascii=False)}",
        },
    ]
    

def get_descriptor_classification_prompt(descriptor, labels):
    return [
        {
            "role": "system",
            "content": """You are a helpful assistant.
            
###Instructions

You are given a descriptor and a list of categories. The descriptor describes a document in some way.
The categories are a set of predefined labels that can be used to classify the descriptor. Your task is to determine which of the given categories best describes the descriptor.
You may select one or more categories that are relevant to the descriptor. Prefer to choose only one category if possible, but if the descriptor clearly fits multiple categories, you may select more than one.
Do not invent new categories. If none of the categories are relevant, you may select "Other" but this should be a last resort.

###Output format

First, briefly explain your reasoning.
Then end you response with the exact string "ANSWER: " followed by a category index or comma-separated list of indices.
Example output: "<reasoning> ANSWER: 1, 3"
""",
        },
        {
            "role": "user",
            "content": f"Categories: {labels}\nDescriptor: {descriptor}",
        },
    ]
    
    
def get_aspect_coverage_prompt(descriptors, aspects):
    return [
        {
            "role": "system",
            "content": """You are a helpful assistant.
            
###Instructions

You are given a list of descriptors and a list of aspects. The descriptors describe a document in various ways, and the aspects are predefined categories that can be used to classify the descriptors. Your task is to determine which of the given aspects are covered by the descriptors.
For each given descriptor, think about which aspects it relates to. Then, determine which aspects are covered by at least one descriptor. It is enough for one descriptor to match an aspect for it to be considered covered.
Choose ALL possible aspects that are directly supported or clearly implied by the descriptors. Do not invent new aspects.

###Output format
First, briefly explain your reasoning.
Then end you response with the exact string "ANSWER: " followed by a comma-separated list of aspect indices.
Example output: "<reasoning> ANSWER: 1, 3"
""",
        },
        {
            "role": "user",
            "content": f"Descriptors: {descriptors}\nAspects: {aspects}",
        },
    ]