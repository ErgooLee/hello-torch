from google import genai
from google.genai import types
import os

# 🔑 配置你的 API Key（建议用环境变量管理）
API_KEY = os.getenv("GEMINI_API_KEY")

print(API_KEY)
# 初始化客户端
client = genai.Client(api_key=API_KEY)

def analyze_video(video_path: str, task_prompt: str):
    """
    1) 上传视频（适合大于 ~20MB 的文件）
    2) 使用 gemini-3-pro-preview 分析视频内容
    """

    # 1) 上传视频文件到 Google GenAI 的文件 API
    print(f"Uploading video {video_path} ...")
    upload_result = client.files.upload(
        file=video_path,
        config={"mimeType": "video/mp4"}
    )
    video_uri = upload_result.uri
    print(f"Uploaded video URI: {video_uri}")

    # 2) 调用模型分析视频
    # 你可以根据需要在 prompt 中指定任务，比如：
    # "Summarize the key events in this video with timestamps."
    contents = types.Content(
        parts=[
            types.Part(
                file_data=types.FileData(file_uri=video_uri, mime_type="video/mp4")
            ),
            types.Part(
                text=task_prompt
            )
        ]
    )

    print("Calling Gemini 3 Pro Preview for video analysis ...")
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=contents
    )

    # 返回文本输出
    return response.text

if __name__ == "__main__":
    video_file = "output_480p.mp4"  # 本地视频文件

    ACTION_PROMPT = """
    你现在是一个5D电影脚本的制作员，需要分析视频的内容设计5D脚本。5D设备有下面硬件可以控制。

    suspension1：可以做动作的悬架，是5D电影的动感来源。动作包括：01=打桩 02=震动-强 03=震动-弱 04=震动 05=车轮敲击 06=左右摇晃 07=前后摇晃 08=腾空 09=落地 10=抬头 11=点头 12=左倾 13=右倾 14=弹跳。打桩的动作可以用在咆哮等突然、意外、惊悚等场景。需要结合视频情节和声音，设计悬架脚本。动感尽量强烈，持续，精准。

    输出的脚步文件为json格式，格式可参考。
    "00:00:21": {
        "base": {
            "scene": "公主第1次敲门-01打桩",
            "time": "00:00:21"
        },
        "suspension1": {
            "duration": "200",
            "mode": "01"
        }
    },
    "00:00:22": {
        "base": {
            "scene": "发生了碰撞-01打桩",
            "time": "00:00:22"
        },
        "suspension1": {
            "duration": "200",
            "mode": "01"
        }
    }
        """

    result_text = analyze_video(video_file, ACTION_PROMPT)
    print("=== Gemini Video Analysis Result ===")
    print(result_text)