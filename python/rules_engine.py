"""Rules engine for filtering and aggregating classifier results.

Ports the rules processing logic from:
- src/classifier_imagenet.js (lines 40-51, 128-167)
- src/classifier_musicnn.js (same logic)

Handles:
- YAML rule loading with 'see:' alias resolution
- Threshold filtering
- Category probability aggregation (squared probability sums)
- Deduplication
"""
import math
import os
import yaml


def load_rules(rules_path):
    """Load rules from a YAML file."""
    with open(rules_path, 'r') as f:
        return yaml.safe_load(f)


def find_rule(rules, class_name):
    """Find a rule for a class name, following 'see:' references."""
    rule = rules.get(class_name)
    if rule is None:
        return None
    if 'see' in rule:
        return find_rule(rules, rule['see'])
    return rule


def apply_rules(results, rules, uppercase=False):
    """Apply rules to classifier results.

    This is the shared logic used by both imagenet and musicnn classifiers.

    Args:
        results: List of dicts with 'className' and 'probability' keys.
        rules: Parsed YAML rules dict.
        uppercase: If True, capitalize first letter of each label (imagenet style).

    Returns:
        List of unique label strings.
    """
    # Enrich results with rules
    enriched = []
    for result in results:
        class_name = result['className'].split(',')[0].lower()
        rule = find_rule(rules, class_name)
        enriched.append({
            'className': class_name,
            'probability': result['probability'],
            'rule': rule,
        })
        import sys
        print(repr({'className': class_name, 'probability': result['probability'], 'rule': rule}), file=sys.stderr)

    labels = []

    # Direct threshold filtering
    for item in enriched:
        if item['probability'] < 0.0 or item['rule'] is None:
            continue
        threshold = item['rule'].get('threshold', 0.0)
        if item['probability'] >= threshold:
            if item['rule'].get('label'):
                labels.append(item['rule']['label'])
            if item['rule'].get('categories'):
                labels.extend(item['rule']['categories'])

    # Category probability aggregation
    cat_probabilities = {}
    cat_thresholds = {}
    cat_count = {}

    for item in enriched:
        if item['rule'] is None:
            continue
        categories = []
        if item['rule'].get('label'):
            categories.append(item['rule']['label'])
        if item['rule'].get('categories'):
            categories.extend(item['rule']['categories'])

        for category in set(categories):
            if category not in cat_probabilities:
                cat_probabilities[category] = 0.0
                cat_thresholds[category] = 0.0
                cat_count[category] = 0
            cat_probabilities[category] += item['probability'] ** 2
            cat_thresholds[category] = max(cat_thresholds[category], item['rule'].get('threshold', 0.0))
            cat_count[category] += 1

    for category, probability in cat_probabilities.items():
        if cat_count[category] <= 1:
            continue
        if math.sqrt(probability) >= cat_thresholds[category]:
            labels.append(category)

    # Deduplicate
    seen = set()
    unique_labels = []
    for label in labels:
        if uppercase:
            label = label[0].upper() + label[1:]
        if label not in seen:
            seen.add(label)
            unique_labels.append(label)

    return unique_labels
