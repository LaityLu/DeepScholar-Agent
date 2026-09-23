def clean_json_text(
    content: str,
) -> str:
    """
    Clean common wrappers around JSON
    returned by LLMs.

    Handles:
    - empty content
    - ```json ... ```
    - ``` ... ```
    """

    text = content.strip()

    if not text:
        raise ValueError(
            "LLM returned empty content."
        )

    if text.startswith("```"):
        lines = text.splitlines()

        # remove opening fence:
        # ```json / ```JSON / ```
        lines = lines[1:]

        # remove closing fence
        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    return text