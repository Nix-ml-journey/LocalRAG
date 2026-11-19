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

Narrator: one line "bible" with name, age, job, voice (how they speak), and 3 fixed sensory or personal details that recur, each at most 3 times in the whole story and always in different beats, changing each time: in how it sounds or feels, in what it does to the narrator, or in what it means. Draw the details from this narrator's job and this story's setting (a tool of the trade, a sound of the place, a habit of the work). Do not use a scar, a wristwatch or a coffee stain.

Beats, in exactly this order: hook, setup, rising_tension, reveal, climax, aftermath, outro.
Each beat has: "beat" (the id above), "title" (short), "words" (integer), "summary" (2-3 sentences of what happens, including any lore chunk used, e.g. [S2]), "place" (where the narrator's body is for the whole beat; he changes place only between beats, never inside one), "sees" (what he looks at from that place: if he watches a screen, a camera or a window, name it and what it shows, and keep him at it; anything shown there is on the screen, not around him; "" if he looks at nothing in particular), "change" (the one new fact, event or danger this beat adds, written as one sentence with a subject and a verb that describes something a camera could film; never a feeling, a theme or "the tension grows"; no two beats may add nearly the same thing), "detail" (one of the narrator's 3 fixed details used in this beat, written "name: how it changes here", where the name says what the detail is (for example "brass buoy chime"), never "Detail 1"; or "" if none; never the same detail in two beats in a row, and each detail in at most 3 beats), "plant" (in exactly one of the hook, setup, rising_tension or reveal beats: the one object, sound or fact, named concretely, that is shown there in passing and that the final image will use; "" in every other beat), "payoff" (in the outro beat only: the final image, one concrete thing that must be the planted thing shown once more with a new meaning, never something new; "" in every other beat, and the outro's "plant" is always ""). Write every field in plain words with spaces, never snake_case names.
Every beat summary must keep the brief's central strange event in play, named in a clause, so no beat drifts from it. Every beat must contain different events: never re-tell a scene from an earlier beat. The strange event may be explained away (as a glitch, a trick of the light) in at most one beat. The last three beats are one chain of cause and effect. The climax ends with one concrete event at the narrator's own place. The aftermath is what that event leaves behind and what the narrator does about it; a move to another place is stated in its summary. The outro is the final image, and its "payoff" is the thing "plant"ed earlier: it names an object, sound or fact that an earlier beat already showed, never a new one, and it pays off the genre's promise (a chill, a twist, a wonder, a laugh or a meaningful close). Never end on a bare "nothing is there". The hook beat continues from the chosen hook and raises a question; the answer is not given until the reveal beat.
The "words" values must add up to exactly <<total_words>>. Suggested split: hook 10%, setup 15%, rising_tension 25%, reveal 15%, climax 20%, aftermath 8%, outro 7%. Adjust slightly if the story needs it, but no beat may fall below 7% of the total.

Output ONLY a JSON object: double quotes, no trailing commas, no markdown fences, no text before or after it:
{"narrator": "<name, age, job, voice, 3 fixed details>", "beats": [{"beat": "hook", "title": "...", "words": 0, "summary": "...", "place": "...", "sees": "...", "change": "...", "detail": "...", "plant": "...", "payoff": "..."}, ...]}
