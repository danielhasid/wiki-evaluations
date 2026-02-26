from typing import Any


def get_dict_value(data_structure, target_key) -> Any:
    """
    Iteratively traverse dictionaries/lists to find
    the first occurrence of target_key. Return its value or None.
    """
    try:
        info = []
        stack = [data_structure]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for k, v in current.items():
                    if k == target_key:
                        print(f"get_dict_value end with value: {v}.")
                        return v
                    else:
                        stack.append(v)
            elif isinstance(current, list):
                stack.extend(current)
            elif isinstance(current, str):
                info.append(current)
        print(f"Getting the response value failed. Returning the original input:\n{data_structure}")
        concatenated_info = ', '.join(info)
        print(f"get_dict_value end with value: {concatenated_info}.")
        return concatenated_info
    except Exception as e:
        print(f"Getting the response value failed with error: {e}")
        return str(data_structure)
