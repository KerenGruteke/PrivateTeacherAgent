import json
import ast


def json_parser(text):
    try:
        return ast.literal_eval(text)
    except:
        pass

    try:
        return json.loads(text)
    except:
        pass

    try:
        first_idx = text.find("{")
        last_idx = text.rfind("}") + 1
        return ast.literal_eval(text[first_idx:last_idx])
    except:
        pass

    try:
        first_idx = text.find("{")
        last_idx = text.rfind("}") + 1
        return json.loads(text[first_idx:last_idx])
    except:
        pass