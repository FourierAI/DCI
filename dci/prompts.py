"""Author-supplied experimental templates, synchronized on 2026-09-03.

The strings match inference_prompt_templates.py distributed with the manuscript.
Flat uses a closed-set instruction; every DCI level includes the None option.
"""

FLAT_PROMPT = (
    "Please directly identify the category name of the image from the candidate category name list. "
    "Do not output sentences or explanations, only the category name as listed (respect case and singular/plural form). "
    "For example: Q: There are 3 categories, [Leopards, wild_cat and electric_guitar]. "
    "Which category does the image belong to? A: wild_cat. "
    "Now answer the question below: Q: There are {count} categories, [{labels}]. "
    "Which category does the image belong to? A:"
)

DCI_PROMPT = (
    "Please directly identify the category name of the image from the candidate category name list. "
    "Always choose the most likely category from the list whenever possible, and only output 'None' if the image clearly does not belong to any category in the list. "
    "Do not output sentences or explanations, only the category name as listed (respect case and singular/plural form). "
    "For example: Q: There are 3 categories, [Leopards, wild_cat and electric_guitar]. "
    "Which category does the image belong to? A: wild_cat. "
    "Now answer the question below: Q: There are {count} categories, [{labels}]. "
    "Which category does the image belong to? A:"
)


def baseline_prompt(classes: list[str], cate_str: str) -> str:
    return FLAT_PROMPT.format(count=len(classes), labels=cate_str)


def dci_prompt(k: int, labels: list[str]) -> str:
    return DCI_PROMPT.format(count=k, labels=",".join(labels))


def build_prompt(labels: list[str], *, flat: bool = False) -> str:
    if flat:
        return baseline_prompt(labels, ",".join(labels))
    return dci_prompt(len(labels), labels)
