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
