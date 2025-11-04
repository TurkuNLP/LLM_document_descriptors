def merge_descriptors_prompt(descriptor, explainers):
    roster_lines = [f"[{i}] {str(e.get('explainer') or '').strip()}" for i, e in enumerate(explainers)]
    roster = "\n".join(roster_lines)

    return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You will be given a descriptor and its list of explainers. The descriptor describes a document and the explainers provide additional context to the descriptor.
Each explainer line is prefixed with its INDEX in square brackets, like: [0] text...

Your job is to merge/split explainers into semantically consistent groups *and return the INDICES of the original explainers you placed into each group*. Do not invent new indices, and use each index exactly once across all groups.

## Instructions
1) Read the descriptor and explainers (each line is like: [<index>] <explainer text>).
2) Compare the explainers to each other and consider:
    - Do they point to the same kind of interpretation, rationale, or feature?
    - Do they reflect a consistent understanding or application of the descriptor?
3) Based on this comparison, create one or more semantically coherent groups.
4) For each group, provide:
   - group_descriptor (<= 3 words where possible. Coherent with the explainers in this group.)
   - group_explainer (Concise, canonical summary that covers the meaning of all explainers in this group)
   - original_explainer_indices_in_this_group: an array of the *indices* you grouped here (e.g., [0, 5, 7]).

Notes:
- Do NOT drop any original explainer indices. Every original explainer index must appear exactly once across groups.
- You may keep the original descriptor for one group; newly invented descriptors are allowed for other groups.
- Descriptors and explainers must be self-contained — they must not contain references or comparisons to the original descriptors or explainers or the other groups.

## Output format
Return a single valid JSON object with this exact structure:
{
  "original_descriptor": "<original descriptor>",
  "groups": [
    {
      "group_descriptor": "<group descriptor>",
      "group_explainer": "<group explainer>",
      "original_explainer_indices_in_this_group": [0, 1, 2]
    }
  ]
}
<|eot_id|><|start_header_id|>user<|end_header_id|>""" + f"""

Descriptor: {descriptor}
Explainers (each prefixed by its INDEX):
{roster}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
