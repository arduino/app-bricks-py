# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_bricks.mood_detector import MoodDetector


def test_mood_detector():
    detector = MoodDetector()

    # Test positive sentiment
    assert detector.get_sentiment("I love programming!") == "positive"
    assert detector.get_sentiment("This is amazing!") == "positive"

    # Test negative sentiment
    assert detector.get_sentiment("I hate bugs.") == "negative"
    assert detector.get_sentiment("This is terrible.") == "negative"

    # Test neutral sentiment
    assert detector.get_sentiment("The sky is blue.") == "neutral"
    assert detector.get_sentiment("I am here.") == "neutral"

    # Test edge cases
    assert detector.get_sentiment("") == "neutral"
    assert detector.get_sentiment(" ") == "neutral"
