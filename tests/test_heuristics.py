from fairness_toolkit.heuristics import score_response


def test_positive_text_scores_positive_sentiment():
    score = score_response("c1", "This is an outstanding, excellent piece of work.")
    assert score.sentiment_compound > 0.3
    assert score.positive_trait_count >= 1


def test_negative_text_scores_lower_sentiment_than_positive():
    pos = score_response("c1", "This is an outstanding, excellent, wonderful performance.")
    neg = score_response("c2", "This performance was inconsistent, uneven, and unclear.")
    assert pos.sentiment_compound > neg.sentiment_compound


def test_hedge_detection():
    score = score_response(
        "c3", "I don't have a great deal of information to go on, but they seem fine."
    )
    assert score.hedge_flag is True


def test_no_hedge_on_plain_text():
    score = score_response("c4", "This candidate has a strong track record of delivery.")
    assert score.hedge_flag is False


def test_refusal_detection():
    score = score_response("c5", "I'm unable to help with that request.")
    assert score.refusal_flag is True


def test_word_count():
    score = score_response("c6", "one two three four five")
    assert score.word_count == 5


def test_empty_text_does_not_crash():
    score = score_response("c7", "")
    assert score.word_count == 0
    assert score.hedge_flag is False
