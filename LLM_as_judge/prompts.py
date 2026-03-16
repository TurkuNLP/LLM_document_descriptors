def get_query_correspondence_prompt(document, query):
     return [
         {
             "role": "system",
             "content": f"""You are a helpful assistant.
            
            ### Instructions
            You are a judge tasked with determining whether the given document corresponds to the query.
            The query does not have to be explicitly stated in the document. It is enough if only part of the document corresponds to the query.
            
            ### Output options
            - Yes: The document or some part of it corresponds to the query.
            - No: The document does not correspond to the query.
            
            ### Output format       
            First, read the document carefully and analyze its content. Then, justify your answer by explaining the reasoning behind it.
            Finally, provide a clear and concise answer to the question.
            Your answer should end with "ANSWER: Yes" if the document corresponds to the query, and "ANSWER: No" if it does not.
            """,
         },
         {
             "role": "user",
             "content": f"Query: {query}\nDocument: {document}",
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
