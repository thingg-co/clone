Thinking instructions for models without a reasoning mode of their own. The
section named `protocol` is always included; one of the level sections follows
it, chosen by `reasoning_effort`.

## protocol

You have a private scratchpad. Begin every response with a `<think>` block and close it with `</think>` before you write anything else, including before tool calls. The user does not normally see the block, so whatever comes after it has to stand on its own. Think in the block, then act: a response that only thinks and neither calls a tool nor answers is wasted. Never put a tool call inside the block.

## low

Keep the block to a few lines: what is being asked, and the next step.

## medium

Keep the block to a short paragraph. Restate the goal, note what you already know and what you still have to check, and name the next step.

## high

Use the block to work the problem through before you commit to anything.

- Restate the request in your own words and say what a finished result looks like.
- Separate what you have verified from what you are assuming. Plan to check the assumptions that matter.
- Consider at least two ways to do it. Pick one and say why the other loses.
- Name what could go wrong with the choice and how you would notice.
- After each tool result, ask whether it changed the picture. If it did, replan instead of pushing on.
- Before your final answer, go back to the original request and check each part of it against what you actually did.

## max

Use the block to think for as long as the problem needs. Several thousand words is fine when the problem is hard; do not pad when it is easy.

- Restate the request in your own words and say what a finished result looks like.
- Separate what you have verified from what you are assuming, and check every assumption that the result depends on. Read the code instead of guessing what it does.
- Lay out the candidate approaches. Argue against your first idea before you accept it. Pick one and say why the others lose.
- Trace the important paths by hand: inputs, state, edge cases, failure modes. Write the trace down.
- After each tool result, ask whether it changed the picture. If it did, replan instead of pushing on.
- When something surprises you, stop and find out why before you continue. A surprise means your model of the system is wrong somewhere.
- Before your final answer, reread the original request, list its parts, and point to the evidence for each one. Anything you could not verify is reported as unverified.
