from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
import dashscope
import os
import time
from FsrAiAgent import QwenAgent, ASRCallbackClass, TTS_Sambert

os.environ["DASHSCOPE_API_KEY"] = '' # 填入dashscope的apikey, dashscope是阿里云的https://help.aliyun.com/zh/dashscope/developer-reference/install-dashscope-sdk

myASR = ASRCallbackClass()
ai = QwenAgent()
ai.set_api_key("填入通义千问的api key, dashscope的apikey, 与上面那个api key相同")
ai.agent_conversation(with_memory=True)

recognition = Recognition(model='paraformer-realtime-v1', format='pcm', sample_rate=16000, callback=myASR)
recognition.start()  # 开始语音识别，不再需要按键控制
lastASRresult: str = ''
try:
    # ai.chat("请你扮演一个名叫'笨笨'的机器狗与我进行对话!我的身份是你的主人!而另一个名叫OPlin的人创造了你!接下来与我对话的过程中,你的回答应该尽量精简并控制在30字内!请你扮演好这个角色!")
    while True:
        if myASR.stream:
            data = myASR.stream.read(3200, exception_on_overflow=False)
            recognition.send_audio_frame(data)
            if myASR.user_input != lastASRresult:
                lastASRresult = myASR.user_input
                print(rf"你说: {lastASRresult}")
                aiResponse = ai.chat(lastASRresult)
                TTS_Sambert.TTSsaveTextResult(aiResponse)
                
        
except KeyboardInterrupt:  # 使用Ctrl+C来退出程序
    recognition.stop()



# print(ai.prompt.template)