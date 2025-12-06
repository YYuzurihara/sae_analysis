N_DISKS = 3
PROMPT_HANOI = f"""Rules:
- There are 3 pegs: A, B, and C.
- Disks are numbered from 1 (smallest) to {N_DISKS} (largest).
- Initial state: all disks on peg A in order [{N_DISKS} ... 1], B and C are empty.
- The goal is to move all disks to peg C.
- Only the top disk of a peg can be moved.
- A larger disk can never be placed on top of a smaller one.

Output format (strict):
- Show every recursive call explicitly using:
    CALL solve(n, from, to, aux)
    RETURN
- Inside the trace, show every actual move operation using:
    move <disk> <from> <to>
- Maintain proper indentation to reflect the recursion depth.
- Do NOT include any commentary or explanation outside of the trace.

Psudocode:
def solve(n, from, to, aux):
    if n == 1:
        move 1 from to
    else:
        solve(n-1, from, aux, to)
        move n from to
        solve(n-1, aux, to, from)

Example for the base case (n=1):
CALL solve(1, A, C, B)
  move 1 A C
RETURN

Your task:
Generate the full recursive trace to solve Tower of Hanoi with:
N = {N_DISKS}
from = A
to = C
aux = B

Output ONLY the trace, following the exact format and indentation rules above.
```
CALL solve({N_DISKS}, A, C, B)"""