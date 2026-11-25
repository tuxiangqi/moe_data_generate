"""Repetition guard behavior."""

from data_generate.repeat_guard import RepeatGuard


def test_period_needs_two_cycles_and_min_tokens():
    guard = RepeatGuard(period_max=8, min_tokens=4, diversity_window=0, diversity_max=0)
    tokens = [7]
    assert guard.observe(tokens, 1) == 0
    for generated in range(2, 4):
        tokens.append(7)
        assert guard.observe(tokens, generated) == 0
    tokens.append(7)
    assert guard.observe(tokens, 4) == 1


def test_period_does_not_stop_a_short_answer():
    guard = RepeatGuard(period_max=8, min_tokens=96, diversity_window=0, diversity_max=0)
    tokens = [3] * 40
    assert guard.observe(tokens, 40) == 0


def test_diversity_waits_for_a_full_window():
    guard = RepeatGuard(period_max=0, min_tokens=0, diversity_window=4, diversity_max=1)
    tokens = []
    for generated in range(1, 4):
        tokens.append(9)
        assert guard.observe(tokens, generated) == 0
    tokens.append(9)
    assert guard.observe(tokens, 4) == -1


def test_prompt_im_end_is_not_a_generated_stop():
    guard = RepeatGuard(period_max=4, min_tokens=4, diversity_window=0, diversity_max=0)
    prompt = [1, 248046, 2]
    assert guard.observe(prompt, 0) == 0


if __name__ == "__main__":
    test_period_needs_two_cycles_and_min_tokens()
    test_period_does_not_stop_a_short_answer()
    test_diversity_waits_for_a_full_window()
    test_prompt_im_end_is_not_a_generated_stop()
    print("repeat guard ok")
