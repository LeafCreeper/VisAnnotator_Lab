import trueskill
import random

def init_ratings(indices):
    """
    Initialize TrueSkill ratings for a list of document indices.
    """
    return {idx: trueskill.Rating() for idx in indices}

def update_comparison(ratings, item_a_idx, item_b_idx, winner=None):
    """
    Update ratings based on comparison.
    winner: 'a', 'b', or 'draw'
    """
    rating_a = ratings[item_a_idx]
    rating_b = ratings[item_b_idx]
    
    if winner == 'a':
        new_a, new_b = trueskill.rate_1vs1(rating_a, rating_b)
    elif winner == 'b':
        new_b, new_a = trueskill.rate_1vs1(rating_b, rating_a)
    else: # draw
        new_a, new_b = trueskill.rate_1vs1(rating_a, rating_b, drawn=True)
        
    ratings[item_a_idx] = new_a
    ratings[item_b_idx] = new_b
    return ratings

def generate_pairs(indices, num_pairs):
    """
    Generate random pairs for comparison.
    """
    if len(indices) < 2:
        return []
    
    pairs = []
    for _ in range(num_pairs):
        pairs.append(random.sample(indices, 2))
    return pairs

def is_trueskill_applicable(fields):
    """
    Checks if schema contains only Integer type fields.
    """
    if not fields:
        return False
    return all(f["type"] == "Integer" for f in fields)
