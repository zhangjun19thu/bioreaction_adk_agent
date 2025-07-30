# src/processing/text_parser.py
import re

def preprocess_text_for_llm(text_content: str) -> str:
    """
    Removes common trailing sections from scientific text to reduce token count for LLMs.

    Args:
        text_content: The full text content of the literature.

    Returns:
        The processed text content, with trailing sections removed.
    """
    min_cut_index = len(text_content)
    STOP_SECTION_KEYWORDS = [
        "REFERENCES",
        "REFERENCE LIST",
        "LITERATURE CITED",
        "BIBLIOGRAPHY",
        "ACKNOWLEDGEMENTS", # Common alternative spelling
        "ACKNOWLEDGMENTS",
        "ACKNOWLEDGMENT", # Singular form
        "SUPPLEMENTAL MATERIAL",
        "SUPPLEMENTARY MATERIAL",
        "SUPPORTING INFORMATION",
        "SUPPLEMENTARY DATA",
        "AUTHOR CONTRIBUTIONS", #有时会出现在文献开头
        "AUTHORSHIP CONTRIBUTIONS", #有时会出现在文献开头
        "CONTRIBUTIONS", # Can be ambiguous, but often used for author contributions
        "CORRESPONDENCE", #有时会出现在文献开头
        "COMPETING INTERESTS",
        "DECLARATION OF COMPETING INTEREST",
        "CONFLICT OF INTEREST",
        "FINANCIAL DISCLOSURES",
        "DISCLOSURE STATEMENT",
        "FUNDING",
        "FUNDING INFORMATION", #有时会出现在文献开头
        "DATA AVAILABILITY",
        "AVAILABILITY OF DATA AND MATERIALS",
        "CODE AVAILABILITY",
        "ETHICS STATEMENT",
        "ETHICAL APPROVAL",
        "ANIMAL ETHICS",
        "HUMAN ETHICS",
        "APPENDIX",
        "APPENDICES"
    ]
    # Add any other sections you want to remove here
    STOP_SECTION_PATTERNS = []
    for keyword in STOP_SECTION_KEYWORDS:
        # Regex:
        # ^                  Anchor to the start of a line (re.MULTILINE)
        # \s*                Optional leading whitespace on the line
        # (?:#+\s+)?         Optional Markdown hashes (e.g., "# ", "## ")
        # {re.escape(keyword)} The keyword itself, escaped for regex safety
        # \s*                Optional spaces after keyword
        # (?::|\b|\r?\n|$)   Followed by a colon, a word boundary, a newline, or end of string.
        #                    This makes sure we match "REFERENCES" or "REFERENCES:"
        #                    and not "REFERENCESsomeotherword".
        #                    \b ensures it's a whole word.
        #                    \r?\n catches newlines.
        #                    $ catches end of string.
        pattern = re.compile(
            rf"^\s*(?:#+\s+)?{re.escape(keyword)}\s*(?::|\b|\r?\n|$)",
            re.IGNORECASE | re.MULTILINE
        )
        STOP_SECTION_PATTERNS.append(pattern)
    
    for pattern in STOP_SECTION_PATTERNS:
        # Search for all occurrences of this pattern to find one in the latter half.
        for match in pattern.finditer(text_content):
            # To prevent cutting off the main content, only consider matches
            # that appear in the second half of the document.
            text_length_threshold = len(text_content) / 2
            if match.start() > text_length_threshold:
                min_cut_index = min(min_cut_index, match.start())
                # print(f"DEBUG: Found potential cut at index {match.start()} for pattern matching '{text_content[match.start():match.start()+30].strip()[:30]}...'")
                # Once we find the first valid match for this pattern,
                # we can break and check the next pattern.
                break

    if min_cut_index < len(text_content):
        print(f"DEBUG: Truncating content. Original length: {len(text_content)}, New length: {min_cut_index}")
        print(f"DEBUG: Content removed starts with: '{text_content[min_cut_index:min_cut_index+50].strip()[:50]}...'")
        return text_content[:min_cut_index].strip() # Remove leading/trailing whitespace from the result
    else:
        print("DEBUG: No stop sections found. Using full content.")
        return text_content.strip() # Still strip, just in case
