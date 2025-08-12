
science_questions_prompt = """
Your science query should be in this format:
Category: <category>
Topic: <topic>
Skill: <skill>

Example 1:
Category: Designing experiments
Topic: science-and-engineering-practices
Skill: Identify the experimental question

Example 2:
Category: Basic economic principles
Topic: economics
Skill: Trade and specialization

Example 3:
Category: Fossils
Topic: earth-science
Skill: Compare ages of fossils in a rock sequence
"""

math_questions_prompt = """
Your math query should be in this format:
Topic: <topic>

Where these are the available topics (with example in parentheses)
- algebra__linear_1d (e.g. Solve 24 = 1601*c - 1605*c for c.)
- algebra__polynomial_roots (e.g. Solve -3*w**3 + 1374*w**2 - 5433*w - 6810 = 0 for w.)
- arithmetic__add_or_sub (e.g. Total of 0.06 and -1977321735.)
- arithmetic__mul (e.g. Multiply 0 and -169144.)
- calculus__differentiate (e.g. What is the first derivative of 388896*d**3 - 222232?)
- comparison__sort (e.g. Sort -65, 5, -66.)
- numbers__gcd (e.g. What is the greatest common factor of 806848 and 21?)
- polynomials__expand (e.g. Expand (4*c + 5*c - 5*c)*((-1 - 1 + 3)*(-14 + 15*c + 14) + c + 0*c - 4*c).)
- probability__swr_p_sequence (e.g. Three letters picked without replacement from {h: 11, p: 5}. Give prob of sequence ppp.)

"""

history_questions_prompt = """ """



