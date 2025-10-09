# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

# Pre load NLTK data for sentiment analysis
import nltk
import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
nltk.data.path.append(os.path.join(current_directory, "assets", "nltk_data"))

from nltk.sentiment import SentimentIntensityAnalyzer

from arduino.app_utils import brick


@brick
class MoodDetector:
    """A class to detect mood based on text sentiment analysis. It can classify text as **positive**, **negative**, or **neutral**.

    Notes:
    - Case-insensitive; basic punctuation does not affect results.
    - English-only. Non-English or mixed-language input may be treated as neutral.
    - Empty or whitespace-only input typically returns neutral.
    - Input must be plain text (str).

    """

    def __init__(self):
        """Initialize the MoodDetector with a sentiment analyzer."""
        self._analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(self, text: str) -> str:
        """Analyze the sentiment of the provided text and return the mood.

        Args:
            text (str): The input text to analyze.

        Returns:
            str: The mood of the text — one of `positive`, `negative`, or `neutral`.
        """
        scores = self._analyzer.polarity_scores(text)
        # Determine sentiment based on compound score
        if scores["compound"] >= 0.05:
            return "positive"
        elif scores["compound"] <= -0.05:
            return "negative"
        else:
            return "neutral"
