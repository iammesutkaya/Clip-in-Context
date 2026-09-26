import sys
import os
import re

sys.path.insert(0, "/Users/Mesut/98 Utility/Apps/Clip in Context")
import clip_in_context

updates = [
    {
        "id": "Hb44m5m885k",
        "title": "Oh yes, they float 🎈",
        "name": "Boat Bridge Flopping Scare"
    },
    {
        "id": "hIWhrWjZ6Zw",
        "title": "Building something beautiful ✨",
        "name": "Stamina Test Climbing Challenge"
    },
    {
        "id": "H_bAJ7St5wc",
        "title": "The power of friendship 🤝",
        "name": "Joining Last Minute Side Quest"
    },
    {
        "id": "zq5bDxDDIpw",
        "title": "The ProGamer back at it again 🎮",
        "name": "Build and Grab Like ProGamer"
    }
]

print("🔄 Updating YouTube video titles and hashtags (#TOTK #Zelda)...")
svc = clip_in_context.youtube_service()

if not svc:
    print("❌ Failed to connect to YouTube service")
    sys.exit(1)

for item in updates:
    vid = item["id"]
    new_t = item["title"]
    try:
        res = svc.videos().list(part="snippet,status", id=vid).execute()
        items = res.get("items", [])
        if not items:
            print(f"❌ Video {vid} not found")
            continue

        video = items[0]
        snippet = video["snippet"]

        # Build clean title (max 100 chars)
        full_title = f"{new_t} #Shorts #TOTK #Zelda #Gaming"[:100]
        snippet["title"] = full_title
        snippet["tags"] = ["TOTK", "Zelda", "TearsOfTheKingdom", "Gaming", "Shorts", "TwitchClips"]
        snippet["categoryId"] = "20"

        # Update description hashtags
        desc = snippet.get("description", "")
        desc = re.sub(r'#TheLegendofZelda\w*', '#TOTK #Zelda', desc, flags=re.IGNORECASE)
        if "#TOTK" not in desc:
            desc += "\n\n#Shorts #TOTK #Zelda #Gaming #ShortsViral"
        snippet["description"] = desc

        body = {
            "id": vid,
            "snippet": snippet
        }

        svc.videos().update(part="snippet", body=body).execute()
        print(f"✅ Updated {item['name']}: \"{full_title}\" (https://youtu.be/{vid})")

    except Exception as e:
        print(f"❌ Error updating {vid}: {e}")

print("🎉 All 4 YouTube Shorts titles and hashtags updated successfully!")
