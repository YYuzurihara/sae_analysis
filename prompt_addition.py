from typing import Tuple

def get_answer(op1: int, op2: int) -> Tuple[str, str]:
    ans = op1 + op2
    return f"{op1}+{op2}={ans}", f"{ans}"