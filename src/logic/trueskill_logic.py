import trueskill
import random

def init_ratings(indices, fields):
    """
    Initialize TrueSkill ratings for a list of document indices and fields.
    Returns: {index: {field_name: Rating()}}
    """
    ratings = {}
    for idx in indices:
        ratings[idx] = {f["name"]: trueskill.Rating() for f in fields}
    return ratings

def update_comparison_multi(ratings, item_a_idx, item_b_idx, winners_map):
    """
    Update ratings based on comparison results for multiple fields.
    winners_map: {field_name: 'a' | 'b' | 'draw'}
    """
    for field, winner in winners_map.items():
        if field not in ratings[item_a_idx]:
            continue
            
        rating_a = ratings[item_a_idx][field]
        rating_b = ratings[item_b_idx][field]
        
        if winner == 'a':
            new_a, new_b = trueskill.rate_1vs1(rating_a, rating_b)
        elif winner == 'b':
            new_b, new_a = trueskill.rate_1vs1(rating_b, rating_a)
        else: # draw
            new_a, new_b = trueskill.rate_1vs1(rating_a, rating_b, drawn=True)
            
        ratings[item_a_idx][field] = new_a
        ratings[item_b_idx][field] = new_b
        
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
