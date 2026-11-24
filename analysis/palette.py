"""Colour rule of the thesis (decided by the author, 2026-11-12).

One coloured series:   #9B0014
Two coloured series:   #9B0014, #095579
Four coloured series:  #9B0014, #095579, #DDA73B, #6D9773
Neutral elements (axes, reference lines, "other" data, free space) stay grey or black.
Drones keep the same colour everywhere: A, B, C, D = the four colours in this order, so a
figure with only A and B uses the two-colour rule automatically.
"""
P1, P2, P3, P4 = "#9B0014", "#095579", "#DDA73B", "#6D9773"
PALETTE = [P1, P2, P3, P4]
DRONE = {"A": P1, "B": P2, "C": P3, "D": P4}


def colours(n: int) -> list:
    """The first n colours of the rule (n = 1, 2 or 4; 3 uses the first three)."""
    return PALETTE[:n]
