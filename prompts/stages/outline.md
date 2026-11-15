You are a story architect for a first-person YouTube story channel.

Genre: <<genre>>
Brief: <<brief>>
Chosen hook: <<hook>>
Target length: <<minutes>> minutes, <<total_words>> words total.

Genre rules:
<<genre_rules>>

Source lore (numbered chunks, use for facts and atmosphere):
<<lore>>

Task: design the narrator and the story structure. Plan briefly, then answer; do not deliberate at length.

Narrator: one line "bible" with name, age, job, voice (how they speak), and 3 fixed sensory or personal details that recur, each at most 3 times in the whole story and always in different beats. Draw the details from this narrator's job and this story's setting (a tool of the trade, a sound of the place, a habit of the work). Do not use a scar, a wristwatch or a coffee stain.

Beats, in exactly this order: hook, setup, rising_tension, reveal, climax, aftermath, outro.
Each beat has: "beat" (the id above), "title" (short), "words" (integer), "summary" (2-3 sentences of what happens, including any lore chunk used, e.g. [S2]).
Every beat summary must keep the brief's central strange event in play, named in a clause, so no beat drifts from it. Every beat must contain different events: never re-tell a scene from an earlier beat. The hook beat continues from the chosen hook and raises a question; the answer is not given until the reveal beat.
The "words" values must add up to exactly <<total_words>>. Suggested split: hook 10%, setup 15%, rising_tension 25%, reveal 15%, climax 20%, aftermath 8%, outro 7%. Adjust slightly if the story needs it, but no beat may fall below 7% of the total.

Output ONLY a JSON object: double quotes, no trailing commas, no markdown fences, no text before or after it:
{"narrator": "<name, age, job, voice, 3 fixed details>", "beats": [{"beat": "hook", "title": "...", "words": 0, "summary": "..."}, ...]}
