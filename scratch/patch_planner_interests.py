import re
from pathlib import Path

FORYOU = Path("/Users/shivamsingh/Desktop/foryou")

# 1. Update contextBuilder.ts
cb_path = FORYOU / "src/ai/orchestrator/contextBuilder.ts"
cb_content = cb_path.read_text(encoding="utf-8")

old_cb_block = """    // 1. Filter relevant long-term memory based on activities & atmosphere
    const relevantActivities = intent.activity.map(a => a.toLowerCase());
    
    // Check relevant interests
    const matchedInterests = memory.interests.filter(i => {
      const lower = i.toLowerCase();
      return relevantActivities.some(a => lower.includes(a) || a.includes(lower));
    });"""

new_cb_block = """    // 1. Filter relevant long-term memory based on activities & atmosphere
    const relevantActivities = intent.activity.map(a => a.toLowerCase());
    const reqLower = (intent.raw_request || '').toLowerCase();
    
    // Check relevant interests
    const matchedInterests = memory.interests.filter(i => {
      const lower = i.toLowerCase();
      if (relevantActivities.some(a => lower.includes(a) || a.includes(lower))) return true;
      if (reqLower.includes('creative') && (lower.includes('pottery') || lower.includes('linocut') || lower.includes('art') || lower.includes('craft') || lower.includes('printmaking'))) return true;
      return false;
    });"""

if old_cb_block in cb_content:
    cb_content = cb_content.replace(old_cb_block, new_cb_block)
    cb_path.write_text(cb_content, encoding="utf-8")
    print("contextBuilder.ts updated successfully")
else:
    print("WARNING: old_cb_block not found in contextBuilder.ts")


# 2. Update intentExtractor.ts
ie_path = FORYOU / "src/ai/intent/intentExtractor.ts"
ie_content = ie_path.read_text(encoding="utf-8")

old_ie_pottery = """    if (text.includes('pottery') || text.includes('art') || text.includes('workshop')) activities.push('pottery workshop');"""

new_ie_pottery = """    if (text.includes('pottery') || text.includes('art') || text.includes('workshop') || text.includes('creative') || text.includes('linocut')) activities.push('creative workshop');"""

if old_ie_pottery in ie_content:
    ie_content = ie_content.replace(old_ie_pottery, new_ie_pottery)
    ie_path.write_text(ie_content, encoding="utf-8")
    print("intentExtractor.ts updated successfully")
else:
    print("WARNING: old_ie_pottery not found in intentExtractor.ts")
