import json
from collections import Counter
from pathlib import Path

def run_audit():
    with open('reports/human_review_samples.jsonl') as f:
        samples = [json.loads(l) for l in f]

    print(f'Inspecting {len(samples)} samples...')

    category_counts = Counter(s['category'] for s in samples)
    print('\nCategory breakdown in sample:')
    for cat, cnt in category_counts.items():
        print(f'  {cat:35s}: {cnt}')

    issues_found = {
        'grammatical_awkwardness': [],
        'context_disconnection': [],
        'canonical_schema_check': [],
        'safety_integrity': []
    }

    areas_list = [
        'indiranagar', 'koramangala', 'hsr layout', 'jayanagar', 'malleshwaram',
        'whitefield', 'jp nagar', 'church street', 'sadashivanagar', 'bellandur',
        'basavanagudi', 'frazer town', 'richmond town', 'electronic city', 'rajajinagar',
        'sankey tank area', 'ulsoor lake area', 'kalyan nagar', 'btm layout',
        'bandra west', 'colaba', 'lower parel', 'juhu', 'kala ghoda', 'versova',
        'dadar', 'powai', 'andheri lokhandwala', 'fort', 'matunga', 'santacruz',
        'hauz khas village', 'connaught place', 'cyberhub gurgaon', 'saket',
        'khan market', 'greater kailash', 'lodhi art district', 'vasant kunj',
        'koregaon park', 'baner', 'kalyani nagar', 'fc road', 'viman nagar', 'aundh',
        'jubilee hills', 'banjara hills', 'gachibowli', 'madhapur', 'kondapur'
    ]

    for s in samples:
        cat = s['category']
        msgs = s['messages']
        u_msgs = [m['content'] for m in msgs if m.get('role') == 'user']
        a_msgs = [m['content'] for m in msgs if m.get('role') == 'assistant']
        
        # 1. Grammar check: 'a [plural/gerund/preposition]'
        for m in msgs:
            content = m.get('content', '')
            for awkward in ['a leather craft basics', 'a vintage arcade games', 'a modern strategy board games', 'a kayaking on the lake', 'a indie card games']:
                if awkward in content.lower():
                    issues_found['grammatical_awkwardness'].append((s['id'], s['line_number'], cat, awkward, content[:120]))
        
        # 2. Context disconnection: User specified area, assistant asked what area
        for u, a in zip(u_msgs, a_msgs):
            u_lower = u.lower()
            a_lower = a.lower()
            if 'which area around' in a_lower:
                for ar in areas_list:
                    if ar in u_lower:
                        issues_found['context_disconnection'].append((s['id'], s['line_number'], cat, ar, u, a))
                        break
                    
        # 3. Canonical preference extraction check
        if cat == 'structured_preference_extraction':
            for a in a_msgs:
                if '```json' in a:
                    json_str = a.split('```json')[1].split('```')[0].strip()
                    try:
                        obj = json.loads(json_str)
                        required_keys = {'action', 'interests', 'preferred_neighborhoods', 'budget_inr_max'}
                        missing = required_keys - set(obj.keys())
                        if missing or obj.get('action') != 'save_preferences':
                            issues_found['canonical_schema_check'].append((s['id'], s['line_number'], missing, obj))
                    except Exception as exc:
                        issues_found['canonical_schema_check'].append((s['id'], s['line_number'], str(exc), json_str))

    print('\nAudit Results across 100 samples:')
    print(f'Grammatical awkwardness matches: {len(issues_found["grammatical_awkwardness"])}')
    for item in issues_found['grammatical_awkwardness'][:5]:
        print(f'  [{item[0]} | Line {item[1]}] Found "{item[3]}" in: {item[4]}...')

    print(f'\nContext disconnection matches: {len(issues_found["context_disconnection"])}')
    for item in issues_found['context_disconnection'][:5]:
        print(f'  [{item[0]} | Line {item[1]}] Stated area: {item[3]}')
        print(f'    User: {item[4]}')
        print(f'    Assistant: {item[5]}')

    print(f'\nCanonical schema issues in sample: {len(issues_found["canonical_schema_check"])}')

if __name__ == '__main__':
    run_audit()
