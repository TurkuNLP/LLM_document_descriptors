def merge_descriptors_prompt(descriptor, explainers):
    return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You will be given a descriptor and a corresponding list of explainers. The descriptor is a shared label or summary phrase that has been applied to multiple different documents. Each explainer provides additional context or interpretation for how that descriptor applies to a specific document.

Your task is to assess the semantic consistency of the explainers under the shared descriptor. In other words, determine whether the explainers are similar enough in meaning, scope, and usage to be reasonably grouped under the same descriptor. Consider both the core idea expressed in each explainer and the way it relates to the descriptor.

##Instructions:

1. Read the descriptor carefully.

2. Review each explainer and consider what it reveals about the document it describes.
    - If the list of explainers is empty, ignore it.

3. Compare the explainers to each other:

    - Do they point to the same kind of interpretation, rationale, or feature?

    - Do they reflect a consistent understanding or application of the descriptor?

4. Based on this comparison, decide whether the explainers form a coherent group under the shared descriptor, or whether there are significant differences suggesting that they should be separated or relabeled.

## Output:

- If the explainers are consistent:
    - Create a new, common explainer that captures the essence of the whole group.

- If the explainers are inconsistent:
    - Split them into as many separate groups as necessary to create consistent groups.
    - For each group, either keep the original descriptor or create a new one that better captures the essence of that group. However, each group must have a unique descriptor, so you can use the original descriptor for only one group.
    - If you invent new descriptors, remember that descriptors can be single words or multi-word phrases. They should typically not be longer than 3 words.
    - For each group, create a new group explainer that captures the essence of the whole group.
    
- Output format:
    - The output must be formatted as a JSON object.
    - Never add any preamble or anything else outside the JSON object.
    - Each dictionary within "groups" should be a consistent group of explainers that share a common descriptor and explainer. 
    - Add as many groups as necessary.
    - Each group must have a unique group_descriptor.
    - Ensure that all original explainers are included in exactly one group in the field original_explainers_in_this_group.
    - If the original list of explainers is empty, the group_explainer should be an empty string and the group_descriptor should be the same as the original descriptor.
    - The output must be a valid JSON object with the following structure:
        {"original_descriptor": "<original descriptor>",
        "groups": [{
                    "group_descriptor": "<group descriptor>",
                    "group_explainer": "<group explainer>",
                    "original_explainers_in_this_group": ["<original explainer>", "<original explainer>", ...]
                    },
                    ...]
        }<|eot_id|><|start_header_id|>user<|end_header_id|>""" + f"""

Descriptor: {descriptor}
Explainers: {explainers}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""