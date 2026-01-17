from typing import Callable
from functools import partial

def prompt_hanoi(n: int, func_name: str = "solve") -> str:
  return f"""Solve Tower of Hanoi problem with {n} disks.

Rules:
- There are 3 pegs: A, B, and C.
- Disks are numbered from 1 (smallest) to {n} (largest).
- Initial state: all disks on peg A in order [{n} ... 1], B and C are empty.
- The goal is to move all disks to peg C.
- Only the top disk of a peg can be moved.
- A larger disk can never be placed on top of a smaller one.

Output format (strict):
- Show every recursive call explicitly using:
  CALL {func_name}(n, from, to, aux)
  RETURN
- Inside the trace, show every actual move operation using:
  move <disk> <from> <to>
- Maintain proper indentation to reflect the recursion depth.
- Do NOT include any commentary or explanation outside of the trace.

Psudocode:
def {func_name}(n, from, to, aux):
  if n == 1:
    move 1 from to
  else:
    {func_name}(n-1, from, aux, to)
    move n from to
    {func_name}(n-1, aux, to, from)

Example for the base case (n=2):
CALL {func_name}(2, A, C, B)
  CALL {func_name}(1, A, B, C)
    move 1 A B
  RETURN
  move 2 A C
  CALL {func_name}(1, B, C, A)
    move 1 B C
  RETURN
RETURN

Your task:
Generate the full recursive trace to solve Tower of Hanoi with:
N = {n}
from = A
to = C
aux = B

Output ONLY the trace, following the exact format and indentation rules above.
```
"""

def solve(n:int, fr:str, to:str, aux:str, ans:str, total_disks:int, func_name: str = "solve") -> str:
  tabs = '  ' * (total_disks - n)
  ans += f"{tabs}CALL {func_name}({n}, {fr}, {to}, {aux})\n"
  if n == 1:
    ans += f"{tabs}  move {n} {fr} {to}\n"
  else:
    ans = solve(n-1, fr, aux, to, ans, total_disks, func_name)
    ans += f"{tabs}  move {n} {fr} {to}\n"
    ans = solve(n-1, aux, to, fr, ans, total_disks, func_name)
  ans += f"{tabs}RETURN\n"
  return ans

def get_answer(n: int, func_name: str = "solve") -> tuple[str, str]:
  """
  return: (prompt, target output)
  """
  solve_fn: Callable[[int, str, str, str], str] = partial(solve, ans="", total_disks=n, func_name=func_name)
  return prompt_hanoi(n, func_name=func_name), solve_fn(n, "A", "C", "B")

# test
if __name__ == "__main__":
  N_DISKS = 3
  prompt, target_output = get_answer(N_DISKS, func_name="solve")
  print(prompt)
  print("-"*100)
  print(target_output)