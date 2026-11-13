# Result details: owner resolution, verifiers and speaker identification

The headline results are in the [README](../../README.md#results). This file holds the detail behind them. Every number below is generated from `results/` by `scripts/check_numbers.py`, like the README.

## Owner resolution with a model

The model sees the item, <!--n:win_before-->5<!--/n--> sentences before and <!--n:win_after-->2<!--/n--> after, and the
closed list of names, and may answer only a name or "unknown"; a name is accepted only above a confidence of
<!--n:owner_thr-->0.8<!--/n--> and if the dialogue mentions it nearby. On the held-out set the 0.5B model resolves
<!--n:o05_resolved-->0<!--/n--> items; the 1.5B model resolves <!--n:o15_resolved-->2<!--/n--> and gets <!--n:o15_wrong-->2<!--/n--> of them wrong, so it is worse than
leaving owners unassigned. Looking at the probabilities, the 1.5B model is confident and wrong on exactly the cases that
cannot be resolved from text ("Nadia" for a sentence spoken by Sol, because Nadia answered the line before): without
speaker turns the transcript simply does not contain the answer. The deterministic cues that do exist (an addressed
request followed by an acceptance, a trailing name in a request, "Thanks, Mei." after a commitment) are in the rules.
On the demo meeting the rules on the text alone name <!--n:demo_owners_plain-->6 of 7<!--/n--> owners correctly.

## Verifier

Asking the model for a free-text JSON label did not work with the 1.5B model: it answered "decision" for almost every candidate, including plain requests, and once returned an invalid label ("discussion"). The raw output of that diagnostic run is not kept in the repository, so it has no number here. The verifier therefore uses closed-choice scoring (each allowed answer is scored by its log-probability, so the model can only answer from the list), and name detection rejects words such as "Almost". The 1.5B verifier reaches <!--n:v15_prf-->0.90 / 0.81 / 0.85<!--/n--> (precision / recall / F1) on the held-out set against the rules' <!--n:rules_prf-->0.93 / 0.84 / 0.89<!--/n-->. Its errors are action versus decision confusion: it calls commitments ("I can take that, I'll re-export it tomorrow", "Sofia will own the runbook") decisions, and calls one decision an action. The 0.5B verifier adds false positives (precision <!--n:v05_p-->0.82<!--/n-->). Neither beats the rules, so neither is the default.

## Speaker identification: method, data and tuning

**Method.** Each recognised sentence is embedded with a sherpa-onnx speaker model (<!--n:diar_embedding-->NeMo TitaNet small<!--/n-->) and the
embeddings are clustered by cosine distance (average linkage, threshold <!--n:diar_threshold-->0.35<!--/n-->; sentences under
<!--n:diar_min_seconds-->1<!--/n--> s do not shape the clusters). With a participants list the number of speakers is set to its length. On the demo meeting
this takes about <!--n:demo_diar_secs-->1<!--/n--> s on the CPU. The baseline is sherpa-onnx's own segmentation pipeline (pyannote segmentation 3.0
plus a WeSpeaker embedding), which cuts the audio into arbitrary slices and clusters those.

**Data and tuning.** I synthesised all <!--n:n_all-->17<!--/n--> labelled meetings with Kokoro-82M, one distinct voice per speaker, so every meeting has audio
with exact speaker turns (`python scripts/synth_eval_meetings.py`; the audio is synthetic speech, written to the git-ignored
`outputs/`). The embedding model (<!--n:n_embeddings-->4<!--/n--> tried), the threshold and the minimum sentence length were chosen on the <!--n:diar_dev_n-->4<!--/n--> `dev` meetings only
(`python scripts/tune_diarize.py sentences`, sweep in `results/diarize_sentence_tuning_dev.json`, baseline sweep in
`results/diarize_segmentation_tuning_dev.json`) and applied unchanged to the <!--n:n_heldout-->9<!--/n--> held-out meetings. DER is the diarisation
error rate (missed speech + false alarm + confusion over reference speech, <!--n:collar-->0.25<!--/n--> s collar).

## From speakers to owners

Clusters are bound to names only where the dialogue says so: a request addressed to X and
answered by a different speaker, "Thanks, X." said after a speaker's turn, a self-introduction. A speaker gets a name
when at least <!--n:vote_conf-->60%<!--/n--> of its evidence agrees, only sentences of at least <!--n:vote_secs-->1<!--/n--> s vote (a one-word "Sure." is often
clustered with the wrong speaker), and a name goes to one speaker only. With a participants list and as many speakers as
participants, I also try every one-to-one assignment that keeps the bound names and breaks no ruled-out pair (someone who
addresses X is not X) and bind a speaker only when all consistent assignments agree. A first-person commitment then takes
its speaker's name.
