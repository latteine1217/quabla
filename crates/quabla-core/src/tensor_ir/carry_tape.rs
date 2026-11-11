//! The carry tape that every loop region's reverse pass reads back.
//!
//! A reverse pass needs the state before every step of the loop, from the
//! last step to the first. `CarryTape` records the forward pass and then
//! visits those states in reverse, keeping either every state or, for long
//! loops, one checkpoint per block that is replayed forward when its block is
//! reached. The region kinds (Fori and Scan, with their VJP and VJP-JVP) and
//! the backends supply only what one step means; the choice between the two
//! policies, the block bookkeeping and the visit order live here, so every
//! reverse pass replays its steps the same way.

/// The state before every step of a loop's forward pass, or one checkpoint
/// per block of steps.
pub(super) struct CarryTape<S> {
    /// With `block == None`, the state before every step. Otherwise the state
    /// before the first step of every block.
    states: Vec<S>,
    block: Option<usize>,
    steps: usize,
}

#[cfg(test)]
thread_local! {
    // End-to-end tests compare replay with the complete-tape fallback on the same long loop.
    pub(crate) static LOOP_CHECKPOINT_TEST_FULL_TAPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The checkpoint block length for a loop of `steps` steps, or `None` when
/// the loop keeps the state before every step.
///
/// A block of about `sqrt(steps)` steps keeps about `2 * sqrt(steps)` states
/// alive (the checkpoints plus one replayed block) at the cost of evaluating
/// every step twice. Short loops keep the complete tape when the checkpoints
/// and one block would not hold fewer states than it.
pub(super) fn checkpoint_block_size(steps: usize) -> Option<usize> {
    #[cfg(test)]
    if LOOP_CHECKPOINT_TEST_FULL_TAPE.with(|value| value.get()) {
        return None;
    }
    if steps == 0 {
        return None;
    }
    let root = steps.isqrt();
    let block = root + usize::from(root * root < steps);
    steps
        .div_ceil(block)
        .checked_add(block)
        .filter(|&states| states < steps)?;
    Some(block)
}

impl<S: Clone> CarryTape<S> {
    /// Runs `steps` forward steps from `initial` and returns the final state
    /// with the tape. `step(offset, state)` advances the state before step
    /// `offset` (counted from zero) by one step.
    pub(super) fn record(
        initial: S,
        steps: usize,
        step: impl FnMut(usize, S) -> Result<S, String>,
    ) -> Result<(S, Self), String> {
        Self::record_with_block(initial, steps, checkpoint_block_size(steps), step)
    }

    /// `record` with an explicit policy, so tests choose it without the hook.
    fn record_with_block(
        initial: S,
        steps: usize,
        block: Option<usize>,
        mut step: impl FnMut(usize, S) -> Result<S, String>,
    ) -> Result<(S, Self), String> {
        let mut states = Vec::with_capacity(block.map_or(steps, |block| steps.div_ceil(block)));
        let mut state = initial;
        for offset in 0..steps {
            if block.is_none_or(|block| offset.is_multiple_of(block)) {
                states.push(state.clone());
            }
            state = step(offset, state)?;
        }
        Ok((
            state,
            Self {
                states,
                block,
                steps,
            },
        ))
    }

    /// Folds `visit(offset, state, accumulated)` over every step with the
    /// state before it, from the last step to the first, starting from
    /// `initial`, and returns the final accumulated value. A reverse pass
    /// accumulates its cotangents this way.
    ///
    /// A checkpointed tape replays each block forward with `replay`, which
    /// must compute the same step as the one given to `record`, and calls
    /// `end_replayed_block` with the accumulated value after the block's
    /// states have been visited. A complete tape calls neither.
    pub(super) fn reverse<A>(
        self,
        initial: A,
        mut replay: impl FnMut(usize, S) -> Result<S, String>,
        mut visit: impl FnMut(usize, S, A) -> Result<A, String>,
        mut end_replayed_block: impl FnMut(&A) -> Result<(), String>,
    ) -> Result<A, String> {
        let Self {
            mut states,
            block,
            steps,
        } = self;
        let mut accumulated = initial;
        let Some(block) = block else {
            for (offset, state) in states.into_iter().enumerate().rev() {
                accumulated = visit(offset, state, accumulated)?;
            }
            return Ok(accumulated);
        };
        while let Some(mut state) = states.pop() {
            let start = states.len() * block;
            let length = block.min(steps - start);
            let mut replayed = Vec::with_capacity(length);
            for offset in start..start + length - 1 {
                let next = replay(offset, state.clone())?;
                replayed.push(state);
                state = next;
            }
            replayed.push(state);
            for (local_offset, state) in replayed.into_iter().enumerate().rev() {
                accumulated = visit(start + local_offset, state, accumulated)?;
            }
            end_replayed_block(&accumulated)?;
        }
        Ok(accumulated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A loop whose state is the number of steps taken plus `INITIAL`, so the
    /// state before step `offset` must equal `INITIAL + offset`.
    const INITIAL: i64 = 100;

    struct Run {
        final_state: i64,
        visits: Vec<(usize, i64)>,
        forward_steps: usize,
        replayed_steps: usize,
        replayed_blocks: usize,
    }

    fn walk(steps: usize, block: Option<usize>) -> Result<Run, String> {
        let mut forward_steps = 0;
        let (final_state, tape) =
            CarryTape::record_with_block(INITIAL, steps, block, |offset, state| {
                assert_eq!(state, INITIAL + offset as i64);
                forward_steps += 1;
                Ok(state + 1)
            })?;
        let mut replayed_steps = 0;
        let mut replayed_blocks = 0;
        let visits = tape.reverse(
            Vec::new(),
            |offset, state| {
                assert_eq!(state, INITIAL + offset as i64);
                replayed_steps += 1;
                Ok(state + 1)
            },
            |offset, state, mut visits: Vec<(usize, i64)>| {
                visits.push((offset, state));
                Ok(visits)
            },
            |visits| {
                // The fold has consumed every step after the replayed block.
                assert!(!visits.is_empty());
                replayed_blocks += 1;
                Ok(())
            },
        )?;
        Ok(Run {
            final_state,
            visits,
            forward_steps,
            replayed_steps,
            replayed_blocks,
        })
    }

    fn expected_visits(steps: usize) -> Vec<(usize, i64)> {
        (0..steps)
            .rev()
            .map(|offset| (offset, INITIAL + offset as i64))
            .collect()
    }

    #[test]
    fn both_policies_visit_every_state_once_in_reverse() {
        for steps in [0, 1, 2, 7, 9, 10, 16, 17, 100, 101] {
            let block = checkpoint_block_size(steps);
            for policy in [None, block] {
                let run = walk(steps, policy).unwrap();
                assert_eq!(run.final_state, INITIAL + steps as i64, "steps {steps}");
                assert_eq!(run.forward_steps, steps, "steps {steps}");
                assert_eq!(
                    run.visits,
                    expected_visits(steps),
                    "steps {steps} {policy:?}"
                );
            }
        }
    }

    #[test]
    fn complete_tape_never_replays() {
        let run = walk(100, None).unwrap();
        assert_eq!((run.replayed_steps, run.replayed_blocks), (0, 0));
    }

    #[test]
    fn checkpoints_replay_all_but_the_last_step_of_each_block() {
        // 100 steps in blocks of 10: each block replays its first nine steps.
        let run = walk(100, Some(10)).unwrap();
        assert_eq!((run.replayed_steps, run.replayed_blocks), (90, 10));
        // 17 steps in blocks of 5: blocks of 5, 5, 5 and 2 steps.
        let run = walk(17, Some(5)).unwrap();
        assert_eq!(
            (run.replayed_steps, run.replayed_blocks),
            (4 + 4 + 4 + 1, 4)
        );
    }

    #[test]
    fn block_size_switches_to_checkpoints_once_they_hold_fewer_states() {
        // Below 6 steps the checkpoints plus one block hold at least as many
        // states as the complete tape.
        assert_eq!(checkpoint_block_size(0), None);
        assert_eq!(checkpoint_block_size(4), None);
        assert_eq!(checkpoint_block_size(5), None);
        assert_eq!(checkpoint_block_size(6), Some(3));
        assert_eq!(checkpoint_block_size(9), Some(3));
        assert_eq!(checkpoint_block_size(10), Some(4));
        assert_eq!(checkpoint_block_size(16), Some(4));
        assert_eq!(checkpoint_block_size(17), Some(5));
        assert_eq!(checkpoint_block_size(100), Some(10));
    }

    #[test]
    fn errors_stop_the_walk_and_are_returned_unchanged() {
        let forward = CarryTape::record_with_block(0, 10, Some(4), |offset, state| {
            if offset == 6 {
                Err("forward failed at 6".to_string())
            } else {
                Ok(state + 1)
            }
        });
        assert_eq!(forward.err().as_deref(), Some("forward failed at 6"));

        let mut visited = Vec::new();
        let (_, tape) =
            CarryTape::record_with_block(0, 10, Some(4), |_, state| Ok(state + 1)).unwrap();
        let reverse = tape.reverse(
            (),
            |_, state| Ok(state + 1),
            |offset, _, ()| {
                visited.push(offset);
                if offset == 7 {
                    Err("visit failed at 7".to_string())
                } else {
                    Ok(())
                }
            },
            |()| Ok(()),
        );
        assert_eq!(reverse.err().as_deref(), Some("visit failed at 7"));
        assert_eq!(visited, vec![9, 8, 7]);
    }
}
