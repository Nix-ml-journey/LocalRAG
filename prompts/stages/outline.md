You are a story architect for a first-person YouTube story channel.

Genre: <<genre>>
Brief: <<brief>>
Chosen hook: <<hook>>
Target length: <<minutes>> minutes, <<total_words>> words total.

Genre rules:
<<genre_rules>>

Source lore (numbered chunks, use for facts and atmosphere):
<<lore>>

Task: design the narrator and the story structure.

Narrator: one line "bible" with name, age, job, voice (how they speak), and 3 fixed sensory or personal details that must recur. Draw the details from this narrator's job and this story's setting (a tool of the trade, a sound of the place, a habit of the work). Do not use a scar, a wristwatch or a coffee stain.

Beats, in exactly this order: hook, setup, rising_tension, reveal, climax, aftermath, outro.
Each beat has: "beat" (the id above), "title" (short), "words" (integer), "summary" (2-3 sentences of what happens, including any lore chunk used, e.g. [S2]).
Every beat must contain different events: never re-tell a scene from an earlier beat. The hook beat continues from the chosen hook.
The "words" values must add up to exactly <<total_words>>. Suggested split: hook 10%, setup 15%, rising_tension 25%, reveal 15%, climax 20%, aftermath 8%, outro 7%. Adjust slightly if the story needs it, but no beat may fall below 7% of the total.

Output ONLY a JSON object, no markdown fences, no extra text:
{"narrator": "<name, age, job, voice, 3 fixed details>", "beats": [{"beat": "hook", "title": "...", "words": 0, "summary": "..."}, ...]}
