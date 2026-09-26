import sys
import os

sys.path.insert(0, "/Users/Mesut/98 Utility/Apps/Clip in Context")
import clip_in_context

clips = [
    {
        "path": "/Users/Mesut/OBS recordings/02_vertical/02_videos/New Folder With Items/Boat Bridge Flopping Scare Fly_STORY.mp4",
        "title": "NEVER BUILD A LOG BRIDGE IN ZELDA! 😱",
        "raw": "It was flying earlier right? I'm so scared... That was so close!",
        "game": "The Legend of Zelda: Tears of the Kingdom"
    },
    {
        "path": "/Users/Mesut/OBS recordings/02_vertical/02_videos/New Folder With Items/Stamina Test Climbing Challenge_STORY.mp4",
        "title": "STAMINA TEST CLIMBING CHALLENGE! 🧗",
        "raw": "Praising the game and now I'm stuck... Nope okay I don't think we needed all those logs",
        "game": "The Legend of Zelda: Tears of the Kingdom"
    },
    {
        "path": "/Users/Mesut/OBS recordings/02_vertical/02_videos/New Folder With Items/Joining Last Minute Side Quest_STORY.mp4",
        "title": "JOINING LAST MINUTE SIDE QUEST! 🎮",
        "raw": "I guess we need the boat so should we get the boat?",
        "game": "The Legend of Zelda: Tears of the Kingdom"
    },
    {
        "path": "/Users/Mesut/OBS recordings/02_vertical/02_videos/New Folder With Items/Build and Grab Like ProGamer_STORY.mp4",
        "title": "BUILD AND GRAB LIKE A PRO GAMER! 🎮",
        "raw": "Look how good it works, it's been an hour and I feel like... It works!",
        "game": "The Legend of Zelda: Tears of the Kingdom"
    }
]

print("🚀 Starting YouTube Shorts batch upload queue...")
for item in clips:
    if os.path.exists(item["path"]):
        print(f"➕ Queued: {os.path.basename(item['path'])} → \"{item['title']}\"")
        clip_in_context.upload_youtube_async(item["path"], item["title"], item["raw"], item["game"])
    else:
        print(f"❌ File not found: {item['path']}")

print("⏳ Waiting for all 4 YouTube Shorts uploads to finish...")
clip_in_context._upload_q.join()
print("🎉 All 4 YouTube Shorts uploaded successfully!")
