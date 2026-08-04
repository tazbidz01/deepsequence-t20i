import re

class CommentaryParser:
    def __init__(self):
        # 1. Delivery Line Patterns
        self.line_patterns = {
            r'\b(outside off|wide of off|wide outside off)\b': 'Outside Off',
            r'\b(on off|off stump|middle and off)\b': 'Off Stump',
            r'\b(middle|middle stump)\b': 'Middle Stump',
            r'\b(leg stump|middle and leg)\b': 'Leg Stump',
            r'\b(down the leg|down leg|leg side)\b': 'Down Leg'
        }
        
        # 2. Delivery Length Patterns
        self.length_patterns = {
            r'\b(yorker|blockhole|toe-crusher)\b': 'Yorker',
            r'\b(full|half-volley|pitched up)\b': 'Full',
            r'\b(slot|in the slot)\b': 'Slot',
            r'\b(good length|length delivery|on a length)\b': 'Good Length',
            r'\b(short|bouncer|back of a length|banged in)\b': 'Short'
        }
        
        # 3. Shot Intent Patterns (Classical Shots)
        self.shot_patterns = {
            r'\b(drive|driven|drives|cover drive|straight drive)\b': 'Drive',
            r'\b(pull|pulled|pulls)\b': 'Pull',
            r'\b(cut|cuts|cut away|square cut|late cut)\b': 'Cut',
            r'\b(sweep|swept|sweeps)\b': 'Sweep',
            r'\b(flick|flicks|flicked|glance|glances)\b': 'Flick',
            r'\b(defend|defends|defended|blocked|blocks|block)\b': 'Defend',
            r'\b(hook|hooks|hooked)\b': 'Hook',
            r'\b(leave|leaves|left alone)\b': 'Leave',
            r'\b(loft|lofted|lofts|smashes|smash|hit)\b': 'Loft/Smash'
        }

    def _extract(self, text, patterns_dict):
        """Helper to extract feature based on dict of regex patterns."""
        for pattern, label in patterns_dict.items():
            if re.search(pattern, text, re.IGNORECASE):
                return label
        return "Unknown"

    def extract_features(self, text):
        """
        Takes raw commentary text and returns extracted Line, Length, and Shot.
        """
        line = self._extract(text, self.line_patterns)
        length = self._extract(text, self.length_patterns)
        shot = self._extract(text, self.shot_patterns)
        
        return {
            'line': line,
            'length': length,
            'shot': shot
        }

if __name__ == "__main__":
    # Test block
    parser = CommentaryParser()
    test_str = "Starc bowls a full delivery outside off-stump, Kohli attempts a drive but edges it to first slip for a dismissal"
    print(parser.extract_features(test_str))
