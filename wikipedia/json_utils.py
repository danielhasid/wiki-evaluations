import json
from ast import literal_eval
from typing import Union

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


# Note that if you try to parse a string with JsonOutputParser, and the string starts with a number, it will return int/float value.
# Example: type(JsonOutputParser().parse(text="  2. The"))


def get_parser(json_string: str) -> Union[JsonOutputParser, StrOutputParser]:
    if (("```json\n{" in json_string or "```\n{" in json_string) and "}\n```" in json_string) or (
            ("```json\n[" in json_string or "```\n[" in json_string) and "]\n```" in json_string) or (
            json_string.startswith(('{', '[')) and json_string.endswith(('}', ']'))):
        return JsonOutputParser(name='json')
    else:
        return StrOutputParser(name='str')


def parse_json_string(json_string: str, output_format: str = 'structured', indent=2) -> [str]:
    print(f"start parse_json_string: {output_format}: {json_string}")
    if not json_string:
        print(f"end parse_json_string: {json_string}")
        return json_string
    try:
        json_string = json_string.strip()
        parser = get_parser(json_string)

        parsed_string = None
        try:
            parsed_string = parser.parse(json_string)
        except Exception as err:
            print(f"{parser} failed to parse:\n{json_string}")
            if output_format == 'structured':
                parsed_string = literal_eval(json_string)

        if not isinstance(parsed_string, str) and output_format == 'structured':
            cleaned_json_string = parsed_string
        elif not isinstance(parsed_string, str) and output_format == 'str':
            cleaned_json_string = json.dumps(parsed_string, ensure_ascii=False, indent=indent)
        else:
            cleaned_json_string = json_string
        print(f"end parse_json_string: {cleaned_json_string}")
        return cleaned_json_string
    except Exception as e:
        print(f"Unable to parse the input as valid JSON: {json_string[:32]}. {e}")
        return json_string
