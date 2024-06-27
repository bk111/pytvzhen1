
import time
start_time = time.time()
import os
import shutil
# if not os.path.exists('/kaggle/working/pytvzhen_kaggle'):
#     shutil.copytree('/kaggle/input/pytvzhen-kaggle/pytvzhen_kaggle', '/kaggle/working/pytvzhen_kaggle', symlinks=False, ignore=None, ignore_dangling_symlinks=False)
#     time.sleep(20)
if not os.path.exists('test'):
    os.system('pip install -r requirements.txt')
    os.system('python3 -m pip install -U "yt-dlp[default]"')
    shutil.copy('simhei.ttf', '/home/studio-lab-user/.conda/envs/studiolab/fonts')
    os.system('chmod 777 /home/studio-lab-user/.conda/envs/studiolab/fonts/simhei.ttf')
    os.system('fc-cache -f -v')

end_time = time.time()  # 记录结束时间
time_copySRC_setupENV = end_time - start_time  # 计算运行时长

from openai import OpenAI
client = OpenAI(
    base_url = 'https://api.chatanywhere.tech/v1',
    api_key = "sk-loHrzSP2KEPOWUVFE5KeARwjTpSbSzKgud3vyMQYjIxFZ2xv"      # 设置你的 API 密钥
)
def srtFileGPTtran(questionfile, srtZhFileNameAndPath):
    questionfile = "translate to Chinese:\n" + open(questionfile, "r", encoding="utf-8").read()
    print(questionfile)
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. please omitting any additional text or instructions."},
        {"role": "user", "content": questionfile}
    ]
    )
    reply = response.choices[0].message.content
    with open(srtZhFileNameAndPath, "w", encoding="utf-8") as file:
        file.write(reply)

from tools.audio_remove import audio_remove
from tools.warning_file import WarningFile

# 检查命令是否存在
def check_command_exists(command):
    return shutil.which(command) is not None

import torch
if torch.cuda.is_available():
    device1 = "cuda"  # 如果存在GPU则选择使用GPU
else:
    device1 = "cpu"   # 否则选择使用CPU
print("当前设备:", device1)



import copy
import json
from pytube import YouTube
from pytube.cli import on_progress
from faster_whisper import WhisperModel
import srt
import re
from pygtrans import Translate
import requests
from tqdm import tqdm
from pydub import AudioSegment
import asyncio  
import edge_tts
import datetime
#from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, CompositeVideoClip, TextClip
from moviepy.video.tools.subtitles import SubtitlesClip
import sys
import traceback
import deepl
import wave
import math
import struct
# import tkinter as tk
# from tkinter import messagebox
# from tkinter import filedialog
################################################################  
#from rookiepy import load
# domain1_main = '.youtube.com'
def get_cookie(domain1_main):
# def get_cookie(url, domain1_main):
    # domain1_main = input('please input a domain like "bilibili.com":')
    cookies = load()
    cookie_simple = ""
    cookie_netscape_format = '# Netscape HTTP Cookie File\n\n'                   # Netscape 格式的cookies能被很多包括yt-dlp来使用。  第一行为 # Netscape HTTP Cookie File\n ， 后续每行以 .域名 开始

    for cookie in cookies:
        if domain1_main in cookie["domain"]:
            # print(cookie)
            cookie_simple = cookie_simple + f'''{cookie["name"]}={cookie["value"]}''' + """; """
            if re.search(r'(^.)', f'''{cookie["domain"]}''').group() == '.':   # netscape_format 的cookie第一行必须是 # Netscape HTTP Cookie File，正式行必须以 .域名  开头
                cookie_netscape_format = cookie_netscape_format + f'''{cookie["domain"]}\tTRUE\t/\tFALSE\t{cookie["expires"]}\t{cookie["name"]}\t{cookie["value"]}\n'''.replace('None', '0')   # www开头的行是 FALSE FALSE

    # print(cookie_simple)
    domain2_main = domain1_main.replace(r'.', '_')
    with open(f'{domain2_main}_cookie_netscape_format.txt', 'w', encoding='utf-8') as f:               #可以给yt-dlp用，yt-dlp --cookies cookie.txt 
        print(cookie_netscape_format, file=f)
    # return cookie_simple, domain2_main
    return cookie_simple, domain2_main, f'{domain2_main}_cookie_netscape_format.txt'
#cookie_filename = get_cookie('.youtube.com')[2]
#print(cookie_filename)
#####################################################


PROXY = "127.0.0.1:7890"
proxies = None
TTS_MAX_TRY_TIMES = 16

paramDictTemplate = {
    "proxy": "127.0.0.1:7890", # 代理地址，留空则不使用代理
    "video Id": "VIDEO_ID", # 油管视频ID
    "work path": "WORK_PATH", # 工作目录  unix下的路径名格式  "work path": "/root/pytvzhen-master/test",   "audio remove model path": "/root/pytvzhen-master/models/baseline.pth", ; windows下的路径名格式 "work path": "H:\\test\\pytvzhen\\pytvzhen-v1.2.1\\test", "audio remove model path": "H:\\test\\pytvzhen\\pytvzhen-v1.2.1\\models\\baseline.pth",
    "download video": True, # [工作流程开关]下载视频
    "download fhd video": True, # [工作流程开关]下载1080p视频
    "extract audio": True, # [工作流程开关]提取音频
    "audio remove": True, # [工作流程开关]去除音乐
    "audio remove model path": "/path/to/your/audio_remove_model", # 去音乐模型路径  unix下的路径名格式  "work path": "/root/pytvzhen-master/test",   "audio remove model path": "/root/pytvzhen-master/models/baseline.pth", ; windows下的路径名格式 "work path": "H:\\test\\pytvzhen\\pytvzhen-v1.2.1\\test", "audio remove model path": "H:\\test\\pytvzhen\\pytvzhen-v1.2.1\\models\\baseline.pth",
    "audio transcribe": True, # [工作流程开关]语音转文字
    "audio transcribe model": "base.en", # [工作流程开关]英文语音转文字模型名称
    "srt merge": True, # [工作流程开关]字幕合并
    "srt merge en to text": True, # [工作流程开关]英文字幕转文字
    "srt merge translate": True, # [工作流程开关]字幕翻译
    "srt merge translate tool": "google", # 翻译工具，目前支持google和deepl
    "srt merge translate key": "", # 翻译工具的key
    "srt merge zh to text": True, # [工作流程开关]中文字幕转文字
    "srt to voice srouce": True, # [工作流程开关]字幕转语音
    "TTS": "edge", # [工作流程开关]合成语音，目前支持edge和GPT-SoVITS
    "TTS param": "", # TTS参数，GPT-SoVITS为地址，edge为角色。edge模式下可以不填，建议不要用GPT-SoVITS。
    "voice connect": True, # [工作流程开关]语音合并
    "audio zh transcribe": True, # [工作流程开关]合成后的语音转文字
    "audio zh transcribe model": "medium", # 中文语音转文字模型名称
    "video zh preview": True # [工作流程开关]视频预览
}
diagnosisLog = None
executeLog = None

def create_param_template(path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(paramDictTemplate, file, indent=4)

def load_param(path):
    with open(path, "r", encoding="utf-8") as file:
        paramDict = json.load(file)
    return paramDict

def download_youtube_video(video_id, fileNameAndPath):
    from pytube import YouTube
    YouTube(f'https://youtu.be/{video_id}', proxies=proxies).streams.first().download(filename=fileNameAndPath)

def transcribeAudioEn(path, modelName="base.en", languate="en",srtFilePathAndName="VIDEO_FILENAME.srt"):

    # 非静音检测阈值，单位为分贝，越小越严格
    NOT_SILENCE_THRESHOLD_DB = -30

    END_INTERPUNCTION = ["…", ".", "!", "?", ";"]
    NUMBER_CHARACTERS = "0123456789"
     # 确保简体中文 
    initial_prompt=None
    if languate=="zh":
        initial_prompt="简体"

    # model = WhisperModel(modelName, device="cuda", compute_type="float16", download_root="faster-whisper_models", local_files_only=False)
    model = WhisperModel(modelName, device1, compute_type="auto", download_root="faster-whisper_models", local_files_only=False)
    print("Whisper model loaded.")

    # faster-whisper
    segments, _ = model.transcribe(audio=path,  language=languate, word_timestamps=True, initial_prompt=initial_prompt)

    # 转换为srt的Subtitle对象
    index = 1
    subs = []
    subtitle = None
    for segment in segments:
        for word in segment.words:
            if subtitle is None:
                subtitle = srt.Subtitle(index, datetime.timedelta(seconds=word.start), datetime.timedelta(seconds=word.end), "")
            finalWord = word.word.strip()
            subtitle.end = datetime.timedelta(seconds=word.end)

            # 一句结束。但是要特别排除小数点被误认为是一句结尾的情况。
            if (finalWord[-1] in END_INTERPUNCTION) and not (len(finalWord)>1 and finalWord[-2] in NUMBER_CHARACTERS):
                pushWord = " " +finalWord
                subtitle.content += pushWord
                subs.append(subtitle)
                index += 1
                subtitle = None
            else:
                if subtitle.content == "":
                    subtitle.content = finalWord
                else:
                    subtitle.content = subtitle.content + " " + finalWord
    # 补充最后一个字幕 
    if subtitle is not None:
        subs.append(subtitle)
        index += 1


    print("Transcription complete.")

    # 重新校准字幕开头，以字幕开始时间后声音大于阈值的第一帧为准
    audio = wave.open(path, 'rb')
    frameRate = audio.getframerate()
    notSilenceThreshold = math.pow(10, NOT_SILENCE_THRESHOLD_DB / 20)
    for sub in subs:
        startTime = sub.start.total_seconds()
        startFrame = int(startTime * frameRate)
        endTime = sub.end.total_seconds()
        endFrame = int(endTime * frameRate)

        newStartTime = startTime
        audio.setpos(startFrame)
        readFrames = endFrame - startFrame
        for i in range(readFrames):
            frame  = audio.readframes(1)
            if not frame :
                break
            samples = struct.iter_unpack("<h", frame) 
            sampleVolumes = []  # 用于存储每个样本的音量值
            for sample_tuple  in samples:
                # sample是一个样本值
                # 调用calculate_volume函数计算样本的音量值，并将结果添加到sampleVolumes列表中
                sample = sample_tuple[0]
                sample_volume = abs(sample) / 32768
                sampleVolumes.append(sample_volume)  # 将音量值添加到列表中
            # 找出所有样本的音量值中的最大值
            maxVolume = max(sampleVolumes)

            if maxVolume > notSilenceThreshold:
                newStartTime = startTime + i / frameRate
                break
    
        sub.start = datetime.timedelta(seconds=newStartTime)
    
    content = srt.compose(subs)
    with open(srtFilePathAndName, "w", encoding="utf-8") as file:
        file.write(content)

    print("SRT file created.")
    print("Output file: " + srtFilePathAndName)
    return True

def transcribeAudioZh(path, modelName="base.en", languate="en",srtFilePathAndName="VIDEO_FILENAME.srt"):
    END_INTERPUNCTION = ["。", "！", "？", "…", "；", "，", "、", ",", ".", "!", "?", ";"]
    ENGLISH_AND_NUMBER_CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # model = WhisperModel(modelName, device="cuda", compute_type="float16", download_root="faster-whisper_models", local_files_only=False)
    model = WhisperModel(modelName, device1, compute_type="auto", download_root="faster-whisper_models", local_files_only=False)
    segments, _ = model.transcribe(audio=path,  language="zh", word_timestamps=True, initial_prompt="简体")
    index = 1
    subs = []
    for segment in segments:
        subtitle = None
        for word in segment.words:
            if subtitle is None:
                subtitle = srt.Subtitle(index, datetime.timedelta(seconds=word.start), datetime.timedelta(seconds=word.end), "")
            finalWord = word.word.strip()
            subtitle.end = datetime.timedelta(seconds=word.end)

            # 排除英文字母+. 情况
            if (finalWord[-1] in END_INTERPUNCTION and not(finalWord[-1] == "." and len(finalWord)>1 and finalWord[-2] in ENGLISH_AND_NUMBER_CHARACTERS)) \
                or (subtitle is not None and len(subtitle.content) > 20) :
                if not ((finalWord[-1] == "." and len(finalWord)>1 and finalWord[-2] in ENGLISH_AND_NUMBER_CHARACTERS) or (subtitle is not None and len(subtitle.content) > 20) ):
                    pushWord = finalWord[:-1]
                else:
                    pushWord = finalWord
                subtitle.content += pushWord
                subs.append(subtitle)
                index += 1
                subtitle = None
            else:
                subtitle.content += finalWord

        if subtitle is not None:
            subs.append(subtitle)
            index += 1

    content = srt.compose(subs)
    with open(srtFilePathAndName, "w", encoding="utf-8") as file:
        file.write(content)

def srtSentanceMerge(sourceSrtFilePathAndName, OutputSrtFilePathAndName):
    srtContent = open(sourceSrtFilePathAndName, "r", encoding="utf-8").read()
    subGenerator = srt.parse(srtContent)
    subList = list(subGenerator)
    if len(subList) == 0:
        print("No subtitle found.")
        return False
    
    diagnosisLog.write("\n<Sentence Merge Section>", False)

    subPorcessingIndex = 1
    subItemList = []
    subItemProcessing = None
    for subItem in subList:
        dotIndex = subItem.content.rfind('.')
        exclamationIndex = subItem.content.rfind('!')
        questionIndex = subItem.content.rfind('?')
        endSentenceIndex = max(dotIndex, exclamationIndex, questionIndex)

        # 异常情况，句号居然在中间
        if endSentenceIndex != -1 and endSentenceIndex != len(subItem.content) - 1:
            logString = f"Warning: Sentence (index:{endSentenceIndex}) not end at the end of the subtitle.\n"
            logString += f"Content: {subItem.content}"
            diagnosisLog.write(logString)
    
        # 以后一个字幕，直接拼接送入就可以了
        if subItem == subList[-1]:
            if subItemProcessing is None:
                subItemProcessing = copy.copy(subItem)
                subItemList.append(subItemProcessing)
                break
            else:
                subItemProcessing.end = subItem.end
                subItemProcessing.content += subItem.content
                subItemList.append(subItemProcessing)
                break

        # 新处理一串字符，则拷贝
        if subItemProcessing is None:
            subItemProcessing = copy.copy(subItem)
            subItemProcessing.content = '' # 清空内容是为了延续后面拼接的逻辑
        
        subItemProcessing.index = subPorcessingIndex
        subItemProcessing.end = subItem.end
        subItemProcessing.content += subItem.content
        # 如果一句话结束了，就把这一句话送入处理
        if endSentenceIndex == len(subItem.content) - 1:
            subItemList.append(subItemProcessing)
            subItemProcessing = None
            subPorcessingIndex += 1

    srtContent = srt.compose(subItemList)
    # 如果打开错误则返回false
    with open(OutputSrtFilePathAndName, "w") as file:
        file.write(srtContent)

def srt_to_text(srt_path):
    with open(srt_path, "r", encoding="utf-8") as file:
        lines = [line.rstrip() for line in file.readlines()]
    text = ""
    for line in lines:
        line = line.replace('\r', '')
        if line.isdigit():
            continue
        if line == "\n":
            continue
        if line == "":
            continue
        if re.search('\\d{2}:\\d{2}:\\d{2},\\d{3} --> \\d{2}:\\d{2}:\\d{2},\\d{3}', line):
            continue
        text += line + '\n'
    return text

def googleTrans(texts):
    if PROXY == "":
        client = Translate()
    else:
        client = Translate(proxies={'https': proxies['https']})
    textsResponse = client.translate(texts, target='zh')
    textsTranslated = []
    for txtResponse in textsResponse:
        textsTranslated.append(txtResponse.translatedText)
    return textsTranslated

def deeplTranslate(texts, key):
    translator = deepl.Translator(key)
    # list to string
    textEn = ""
    for oneLine in texts:
        textEn += oneLine + "\n"

    textZh = translator.translate_text(textEn, target_lang="zh")
    textZh = str(textZh)
    textsZh = textZh.split("\n")
    return textsZh

def srtFileGoogleTran(sourceFileNameAndPath, outputFileNameAndPath):
    srtContent = open(sourceFileNameAndPath, "r", encoding="utf-8").read()
    subGenerator = srt.parse(srtContent)
    subTitleList = list(subGenerator)
    contentList = []
    for subTitle in subTitleList:
        contentList.append(subTitle.content)
    
    contentList = googleTrans(contentList)

    for i in range(len(subTitleList)):
        subTitleList[i].content = contentList[i]
    
    srtContent = srt.compose(subTitleList)
    with open(outputFileNameAndPath, "w", encoding="utf-8") as file:
        file.write(srtContent)

def srtFileDeeplTran(sourceFileNameAndPath, outputFileNameAndPath, key):
    srtContent = open(sourceFileNameAndPath, "r", encoding="utf-8").read()
    subGenerator = srt.parse(srtContent)
    subTitleList = list(subGenerator)
    contentList = []
    for subTitle in subTitleList:
        contentList.append(subTitle.content)
    
    contentList = deeplTranslate(contentList, key)

    for i in range(len(subTitleList)):
        subTitleList[i].content = contentList[i]
    
    srtContent = srt.compose(subTitleList)
    with open(outputFileNameAndPath, "w", encoding="utf-8") as file:
        file.write(srtContent)

def stringToVoice(url, string, outputFile):
    data = {
        "text": string,
        "text_language": "zh"
    }
    response = requests.post(url, json=data)
    if response.status_code != 200:
        return False
    
    with open(outputFile, "wb") as f:
        f.write(response.content)

    return True

def srtToVoice(url, srtFileNameAndPath, outputDir):
    # create output directory if not exists
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    srtContent = open(srtFileNameAndPath, "r", encoding="utf-8").read()
    subGenerator = srt.parse(srtContent)
    subTitleList = list(subGenerator)
    index = 1
    fileNames = []
    print("Start to convert srt to voice")
    with tqdm(total=len(subTitleList)) as pbar:
        for subTitle in subTitleList:
            string = subTitle.content
            fileName = str(index) + ".wav"
            outputNameAndPath = os.path.join(outputDir, fileName)
            fileNames.append(fileName)
            tryTimes = 0

            while tryTimes < TTS_MAX_TRY_TIMES:
                if not stringToVoice(url, string, outputNameAndPath):
                    return False
                
                # 获取outputNameAndPath的时间长度
                audio = AudioSegment.from_wav(outputNameAndPath)
                duration = len(audio)
                # 获取最大音量
                maxVolume = audio.max_dBFS

                # 如果音频长度小于500ms，则重试，应该是数据有问题了
                if duration > 600 and maxVolume > -15:
                    break

                tryTimes += 1
            
            if tryTimes >= TTS_MAX_TRY_TIMES:
                print(f"Warning Failed to convert {fileName} to voice.")
                print(f"Convert {fileName} duration: {duration}ms, max volume: {maxVolume}dB")

            index += 1
            pbar.update(1) # update progress bar

    voiceMapSrt = copy.deepcopy(subTitleList)
    for i in range(len(voiceMapSrt)):
        voiceMapSrt[i].content = fileNames[i]
    voiceMapSrtContent = srt.compose(voiceMapSrt)
    voiceMapSrtFileAndPath = os.path.join(outputDir, "voiceMap.srt")
    with open(voiceMapSrtFileAndPath, "w", encoding="utf-8") as f:
        f.write(voiceMapSrtContent)
    
    srtAtitionalFile = os.path.join(outputDir, "zh.srt")
    with open(srtAtitionalFile, "w", encoding="utf-8") as f:
        f.write(srtContent)
    
    print("Convert srt to voice successfully")
    return True

def srtToVoiceEdge(srtFileNameAndPath, outputDir, charactor = "zh-CN-XiaoyiNeural"):
    # create output directory if not exists
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    srtContent = open(srtFileNameAndPath, "r", encoding="utf-8").read()
    subGenerator = srt.parse(srtContent)
    subTitleList = list(subGenerator)
    index = 1
    fileNames = []
    fileMp3Names = []
    
    async def convertSrtToVoiceEdge(text, path):
        print(f"Start to convert srt to voice into {path}, text: {text}")
        communicate = edge_tts.Communicate(text, charactor)
        await communicate.save(path)

    coroutines  = []
    for subTitle in subTitleList:
        fileMp3Name = str(index) + ".mp3"
        fileName = str(index) + ".wav"
        outputMp3NameAndPath = os.path.join(outputDir, fileMp3Name)
        fileMp3Names.append(fileMp3Name)
        fileNames.append(fileName)
        coroutines.append(convertSrtToVoiceEdge(subTitle.content, outputMp3NameAndPath))
        index += 1

    # wait for all coroutines to finish
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*coroutines))
    
    print("\nConvert srt to mp3 voice successfully")

    # convert mp3 to wav
    for i in range(len(fileMp3Names)):
        mp3FileName = fileMp3Names[i]
        wavFileName = fileNames[i]
        mp3FileAndPath = os.path.join(outputDir, mp3FileName)
        wavFileAndPath = os.path.join(outputDir, wavFileName)
        sound = AudioSegment.from_mp3(mp3FileAndPath)
        sound.export(wavFileAndPath, format="wav")
        os.remove(mp3FileAndPath)

    voiceMapSrt = copy.deepcopy(subTitleList)
    for i in range(len(voiceMapSrt)):
        voiceMapSrt[i].content = fileNames[i]
    voiceMapSrtContent = srt.compose(voiceMapSrt)
    voiceMapSrtFileAndPath = os.path.join(outputDir, "voiceMap.srt")
    with open(voiceMapSrtFileAndPath, "w", encoding="utf-8") as f:
        f.write(voiceMapSrtContent)
    
    srtAtitionalFile = os.path.join(outputDir, "sub.srt")
    with open(srtAtitionalFile, "w", encoding="utf-8") as f:
        f.write(srtContent)
    
    print("Convert srt to wav voice successfully")
    return True

def zhVideoPreview(videoFileNameAndPath, voiceFileNameAndPath, insturmentFileNameAndPath, srtFileNameAndPath, outputFileNameAndPath):
    
    video_clip = VideoFileClip(videoFileNameAndPath)

    # 加载音频
    voice_clip = None
    if (voiceFileNameAndPath is not None) and os.path.exists(voiceFileNameAndPath):
        voice_clip = AudioFileClip(voiceFileNameAndPath)
    insturment_clip = None
    if (insturmentFileNameAndPath is not None) and os.path.exists(insturmentFileNameAndPath):
        insturment_clip = AudioFileClip(insturmentFileNameAndPath)
    
    # 组合音频剪辑
    final_audio = None
    if voiceFileNameAndPath is not None and insturmentFileNameAndPath is not None:
        final_audio = CompositeAudioClip([voice_clip, insturment_clip])
    elif voiceFileNameAndPath is not None:
        final_audio = voice_clip
    elif insturmentFileNameAndPath is not None:
        final_audio = insturment_clip
    
    video_clip = video_clip.set_audio(final_audio)  
    video_clip.write_videofile(outputFileNameAndPath, codec='libx264', audio_codec='aac', remove_temp=True)
    
    return True

def zhVideoPreview1(videoFileNameAndPath, voiceFileNameAndPath, insturmentFileNameAndPath, srtFileNameAndPath, outputFileNameAndPath, srtEnFileNameMergeAndPath, srtZhFileNameAndPath):
    
    video_clip = VideoFileClip(videoFileNameAndPath)

    original_width, original_height = video_clip.size
    print(original_width, original_height)
    new_height1 = 450
    new_height2 = 490
    print(new_height1, new_height2, type(new_height1))
    # generator_ori = lambda txt : TextClip(txt, font='Arial', fontsize=20, color='white', stroke_color='black', stroke_width=0.1, method='caption', size=(original_width - 10, 20))
    # generator_trans = lambda txt : TextClip(txt, font='/usr/share/fonts/windows/simsun.ttc', fontsize=20, color='red', stroke_color='black', stroke_width=0.1, method='caption', size=(original_width - 10, 20))    # 用 print (TextClip.list("font") ) 来看支持的字体，找出中文的
    generator_ori = lambda txt : TextClip(txt, font='Times-Italic', fontsize=45, color='white', method='caption', size=(original_width*0.85, 0))    # Arial
    generator_trans = lambda txt : TextClip(txt, font='黑体', fontsize=45, color='red', method='caption', size=(original_width*0.85, 0))    # 用 print (TextClip.list("font") ) 来看支持的字体，找出中文的


    subtitles_ori = SubtitlesClip(srtEnFileNameMergeAndPath, generator_ori)
    subtitles_trans = SubtitlesClip(srtZhFileNameAndPath, generator_trans)

    # 加载音频
    voice_clip = None
    if (voiceFileNameAndPath is not None) and os.path.exists(voiceFileNameAndPath):
        voice_clip = AudioFileClip(voiceFileNameAndPath)
    insturment_clip = None
    if (insturmentFileNameAndPath is not None) and os.path.exists(insturmentFileNameAndPath):
        insturment_clip = AudioFileClip(insturmentFileNameAndPath)
    
    # 组合音频剪辑
    final_audio = None
    if voiceFileNameAndPath is not None and insturmentFileNameAndPath is not None:
        final_audio = CompositeAudioClip([voice_clip, insturment_clip])
    elif voiceFileNameAndPath is not None:
        final_audio = voice_clip
    elif insturmentFileNameAndPath is not None:
        final_audio = insturment_clip
    
    video_clip = video_clip.set_audio(final_audio)  
    #video_clip.write_videofile(outputFileNameAndPath, codec='libx264', audio_codec='aac', remove_temp=True)
    video_clip = CompositeVideoClip([video_clip, subtitles_ori.set_position((50, original_height - 200)).set_duration(video_clip.duration), subtitles_trans.set_position((50, original_height - 280)).set_duration(video_clip.duration)])    # 500好像在中间，被减值越大越靠上
    video_clip.write_videofile(outputFileNameAndPath, codec="libx264", audio_codec='aac', remove_temp=True)     #  threads = 8, fps=24,
    
    return True

def zhVideoPreview_ffmpeg(s_video, viedoFileNameAndPath, voiceFileNameAndPath, insturmentFileNameAndPath, outputFileNameAndPath, srtEnFileNameMergeAndPath, srtZhFileNameAndPath):
    
    video_clip = VideoFileClip(s_video)

    original_width, original_height = video_clip.size
    print(original_width, original_height)
    new_height1 = 450
    new_height2 = 490
    print(new_height1, new_height2, type(new_height1))
    # generator_ori = lambda txt : TextClip(txt, font='Arial', fontsize=20, color='white', stroke_color='black', stroke_width=0.1, method='caption', size=(original_width - 10, 20))
    # generator_trans = lambda txt : TextClip(txt, font='/usr/share/fonts/windows/simsun.ttc', fontsize=20, color='red', stroke_color='black', stroke_width=0.1, method='caption', size=(original_width - 10, 20))    # 用 print (TextClip.list("font") ) 来看支持的字体，找出中文的
    
    # generator_ori = lambda txt : TextClip(txt, font='Times-Italic', fontsize=45, color='white', method='caption', size=(original_width*0.85, 0))    # Arial
    # generator_trans = lambda txt : TextClip(txt, font='黑体', fontsize=45, color='red', method='caption', size=(original_width*0.85, 0))    # 用 print (TextClip.list("font") ) 来看支持的字体，找出中文的


    # subtitles_ori = SubtitlesClip(srtEnFileNameMergeAndPath, generator_ori)
    # subtitles_trans = SubtitlesClip(srtZhFileNameAndPath, generator_trans)

    # 加载音频
    voice_clip = None
    if (voiceFileNameAndPath is not None) and os.path.exists(voiceFileNameAndPath):
        voice_clip = AudioFileClip(voiceFileNameAndPath)
    insturment_clip = None
    if (insturmentFileNameAndPath is not None) and os.path.exists(insturmentFileNameAndPath):
        insturment_clip = AudioFileClip(insturmentFileNameAndPath)
    
    # 组合音频剪辑
    final_audio = None
    if voiceFileNameAndPath is not None and insturmentFileNameAndPath is not None:
        final_audio = CompositeAudioClip([voice_clip, insturment_clip])
    elif voiceFileNameAndPath is not None:
        final_audio = voice_clip
    elif insturmentFileNameAndPath is not None:
        final_audio = insturment_clip
    
    # video_clip = video_clip.set_audio(final_audio)  
    #video_clip.write_videofile(outputFileNameAndPath, codec='libx264', audio_codec='aac', remove_temp=True)
    # video_clip = CompositeVideoClip([video_clip, subtitles_ori.set_position((50, original_height - 200)).set_duration(video_clip.duration), subtitles_trans.set_position((50, original_height - 280)).set_duration(video_clip.duration)])    # 500好像在中间，被减值越大越靠上
    # video_clip.write_videofile(outputFileNameAndPath, codec="libx264", audio_codec='aac', remove_temp=True)     #  threads = 8, fps=24,
    #os.system(f"""ffmpeg -i {videoFileNameAndPath} -i {voiceFileNameAndPath} -i {insturmentFileNameAndPath} -i {srtZhFileNameAndPath} -i {srtEnFileNameMergeAndPath} -map 0:v -map 1:a -map 2:a -scodec mov_text -map 3 -metadata:s:s:0 language=eng -metadata:s:s:0 title="eng1" -scodec mov_text -map 4 -metadata:s:s:1 language=sme -metadata:s:s:1 title="background" -c:v libx264 -c:a aac -b:v 3000k -b:a 128k -vf subtitles={srtZhFileNameAndPath}:force_style='Fontsize=9\,Fontname=黑体\,WrapStyle=2\,MarginV=30\,BorderStyle=1\,Outline=1\,Shadow=0\,PrimaryColour=&HFFFFFF&\,OutlineColour=&H853F1B&\,Spacing=0'\,subtitles={srtEnFileNameMergeAndPath}:force_style='Fontsize=9\,Fontname=黑体\,MarginV=02\,BorderStyle=1\,Outline=1\,Shadow=0\,PrimaryColour=&HFFFFFF&\,OutlineColour=&H5A6A83&\,Spacing=0' {outputFileNameAndPath} """)
    import subprocess
    def judge_video_resolution(video_path):
        # 执行ffprobe命令获取视频信息
        command = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'json', video_path]
        result = subprocess.run(command, capture_output=True, text=True)

        # 解析ffprobe输出结果
        output = json.loads(result.stdout)
        streams = output.get('streams')
        # print(streams, type(streams))

        if streams:                                         # 有的视频streams的值类似[{'width': 1920, 'height': 1080}]，有的类似 [{}, {'width': 1920, 'height': 1080}, {}, {}]  所以要遍历这个list。找到长度大于0的字典才行
            for index, x1 in enumerate(streams):
                if len(streams[index]) > 0:
                    width = streams[index].get('width')
                    height = streams[index].get('height')
                    # 计算宽高比
                    aspect_ratio = width / height
                    # 判断宽高比
                    if aspect_ratio >= 1.7:  # 宽屏通常16:9或者更宽
                        # print("for_pc")
                        return 'for_pc'
                    elif aspect_ratio <= 0.65:  # 竖屏通常为9:16
                        # print("for_phone")
                        return 'for_phone'
                    else:
                        # print("这是一个未知类型视频")
                        return 'unknown'
                else:
                    pass
        else:
            print("无法解析视频信息")

    video_type = judge_video_resolution(s_video)
    print(video_type, '-----------------------------------------------------1')
    
    import pysrt
    import textwrap
    # 加载字幕文件
    def textwrap1(srt_file):
        subtitle = pysrt.open(srt_file, encoding='utf-8')
        # 修改第一条字幕的内容
        if video_type == 'for_phone':
            width_line = 14
        else:
            width_line = 35
        print(width_line, '-----------------------------------------------------2')
        for sub in subtitle:
            len1 = len(sub.text)
            len_en = len(re.sub(r'[^a-zA-Z]', '', sub.text))
            print(len_en, len1, '-----------------------------------------------------3')
            if len_en/len1 > 1/2:
                sub.text = "\n".join(textwrap.wrap(sub.text, width_line*2))
            else:
                sub.text = "\n".join(textwrap.wrap(sub.text, width_line))
        
        # 保存修改后的字幕
        subtitle.save(srt_file + '_break.srt')
        return srt_file + '_break.srt'
    srt_file = f"""{srtZhFileNameAndPath}"""
    tmp_srtZhFileNameAndPath = textwrap1(srt_file)
    
    print(s_video, '************************************************************************ 1')
    print(viedoFileNameAndPath, '************************************************************************ 1.5')
    print(voiceFileNameAndPath, '*************************************************************** 2')
    print(insturmentFileNameAndPath, '***************************************************** 3')
    
    
    
    # burn_en_subtitle = paramDict["ffmpeg_srtEnFileNameMergeAndPath"]:
    
    if change_origin_speech == 'y' and burn_en_subtitle == 'n':   #  变语音 + 烧中字 + 不烧英字
        print('burn En subtitale?-------------------', burn_en_subtitle)
        # srtEnFileNameMergeAndPath = os.path.join(workPath, "srtEnFileNameMerge_no.srt")

        command = f'ffmpeg -y -i {s_video} -i {insturmentFileNameAndPath} -i {voiceFileNameAndPath} -i {tmp_srtZhFileNameAndPath} '  #注意，insturment 在 voiceFile 之前。跟下面命令的位置是对应的。#不压英文字幕
        command += '-filter_complex "[1:a:0]channelsplit=channel_layout=mono[a1];[2:a]channelsplit=channel_layout=stereo[a2][a3];[a1][a2][a3]amerge=inputs=3,pan=stereo|c0<c0+c2|c1<c1+c2[out]" '
        command += '-map 0:v ' 
        command += '-map "[out]" -map 2:a '
        # command += '-scodec mov_text -map 3 -metadata:s:s:0 language=zh -metadata:s:s:0 title="zh" '
        # command += '-c:v libx264 -c:a aac -b:v 3000k -b:a 128k ' 
        command += '-c:v libx264 -preset slow -crf 26 -c:a aac -strict experimental -ab 128000 -ac 2 -ar 48000 '
        command += f'-vf "subtitles={tmp_srtZhFileNameAndPath}:fontsdir=/home/studio-lab-user/.conda/envs/studiolab/fonts/simhei.ttf:force_style=\'Fontsize=9,Fontname=黑体,MarginV=35,BorderStyle=1,Outline=1,Shadow=0,PrimaryColour=&HFFFFFF&,OutlineColour=&H853F1B&,Spacing=0\'" '
        command += f'{outputFileNameAndPath}'
        os.system(f"""{command}""")

    elif change_origin_speech == 'y' and burn_en_subtitle == 'y':   #  变语音 + 烧中字 + 烧英字:
        print("s_video:", s_video)
        print("voiceFileNameAndPath:", voiceFileNameAndPath)
        command = f'ffmpeg -y -i {s_video} -i {insturmentFileNameAndPath} -i {voiceFileNameAndPath} -i {tmp_srtZhFileNameAndPath} -i {srtEnFileNameMergeAndPath} '  #压英文字幕
        command += '-filter_complex "[1:a:0]channelsplit=channel_layout=mono[a1];[2:a]channelsplit=channel_layout=stereo[a2][a3];[a1][a2][a3]amerge=inputs=3,pan=stereo|c0<c0+c2|c1<c1+c2[out]" '
        # 视频流
        command += '-map 0:v ' 
        # 音频流
        command += '-map "[out]" -map 2:a '
        # 混音滤镜
        #command += '-filter_complex "[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3" ' 
        command += '-c:v libx264 -preset slow -crf 26 -c:a aac -strict experimental -ab 128000 -ac 2 -ar 48000 '
        # 字幕流
        # command += '-scodec mov_text -map 3 -metadata:s:s:0 language=eng -metadata:s:s:0 title="eng1" '
        # command += '-scodec mov_text -map 4 -metadata:s:s:1 language=sme -metadata:s:s:1 title="background" '
        # 过滤器选项
        command += f'-vf "subtitles={tmp_srtZhFileNameAndPath}:fontsdir=/home/studio-lab-user/.conda/envs/studiolab/fonts/simhei.ttf:force_style=\'Fontsize=9,Fontname=黑体,MarginV=35,BorderStyle=1,Outline=1,Shadow=0,PrimaryColour=&HFFFFFF&,OutlineColour=&H853F1B&,Spacing=0\', '
        # if paramDict["ffmpeg_srtEnFileNameMergeAndPath"] == 'no':
        #     command += ''
        # else:
        command += f'subtitles={srtEnFileNameMergeAndPath}:fontsdir=/home/studio-lab-user/.conda/envs/studiolab/fonts/simhei.ttf:force_style=\'Fontsize=9,Fontname=黑体,MarginV=08,BorderStyle=1,Outline=1,Shadow=0,PrimaryColour=&HFFFFFF&,OutlineColour=&H5A6A83&,Spacing=0\'" '
        # 输出文件
        command += f'{outputFileNameAndPath}'
        #print(f'{command}')
        os.system(f"""{command}""")

    elif change_origin_speech == 'n' and burn_en_subtitle == 'n':   #  不变语音 + 烧中字 + 不烧英字
        command = f'ffmpeg -y -i {viedoFileNameAndPath} -i {tmp_srtZhFileNameAndPath} '   #不压英文字幕
        command += '-map 0:v ' 
        # 音频流
        command += '-map 0:a '
        command += '-c:v libx264 -c:a aac -b:v 3000k -b:a 128k '
        command += f'-vf "subtitles={tmp_srtZhFileNameAndPath}:fontsdir=/home/studio-lab-user/.conda/envs/studiolab/fonts/simhei.ttf:force_style=\'Fontsize=9,Fontname=黑体,MarginV=35,BorderStyle=1,Outline=1,Shadow=0,PrimaryColour=&HFFFFFF&,OutlineColour=&H853F1B&,Spacing=0\'" '
        command += f'{outputFileNameAndPath}'
        os.system(f"""{command}""")

    else: #  不变语音 + 烧中字 + 烧英字
        command = f'ffmpeg -y -i {viedoFileNameAndPath} -i {tmp_srtZhFileNameAndPath} -i {srtEnFileNameMergeAndPath} '  #压英文字幕
        command += '-map 0:v ' 
        # 音频流
        command += '-map 0:a '
        command += '-c:v libx264 -c:a aac -b:v 3000k -b:a 128k '
        command += f'-vf "subtitles={tmp_srtZhFileNameAndPath}:fontsdir=/home/studio-lab-user/.conda/envs/studiolab/fonts/simhei.ttf:force_style=\'Fontsize=9,Fontname=黑体,MarginV=35,BorderStyle=1,Outline=1,Shadow=0,PrimaryColour=&HFFFFFF&,OutlineColour=&H853F1B&,Spacing=0\', '
        command += f'subtitles={srtEnFileNameMergeAndPath}:fontsdir=/usr/share/fonts/truetype:force_style=\'Fontsize=9,Fontname=黑体,MarginV=08,BorderStyle=1,Outline=1,Shadow=0,PrimaryColour=&HFFFFFF&,OutlineColour=&H5A6A83&,Spacing=0\'" '
        command += f'{outputFileNameAndPath}'
        os.system(f"""{command}""")


    
    return True



def voiceConnect(sourceDir, outputAndPath):
    MAX_SPEED_UP = 1.2  # 最大音频加速
    MIN_SPEED_UP = 1.05  # 最小音频加速
    MIN_GAP_DURATION = 0.1  # 最小间隔时间，单位秒。低于这个间隔时间就认为音频重叠了

    if not os.path.exists(sourceDir):
        return False
    
    srtMapFileName = "voiceMap.srt"
    srtMapFileAndPath = os.path.join(sourceDir, srtMapFileName)
    if not os.path.exists(srtMapFileAndPath):
        return False
    
    voiceMapSrtContent = ""
    with open(srtMapFileAndPath, "r", encoding="utf-8") as f:
        voiceMapSrtContent = f.read()

    # 确定音频长度
    voiceMapSrt = list(srt.parse(voiceMapSrtContent))
    duration = voiceMapSrt[-1].end.total_seconds() * 1000
    finalAudioFileAndPath = os.path.join(sourceDir, voiceMapSrt[-1].content)
    finalAudioEnd = voiceMapSrt[-1].start.total_seconds() * 1000
    finalAudioEnd += AudioSegment.from_wav(finalAudioFileAndPath).duration_seconds * 1000
    duration = max(duration, finalAudioEnd)

    diagnosisLog.write("\n<Voice connect section>", False)

    # 初始化一个空的音频段
    combined = AudioSegment.silent(duration=duration)
    for i in range(len(voiceMapSrt)):
        audioFileAndPath = os.path.join(sourceDir, voiceMapSrt[i].content)
        audio = AudioSegment.from_wav(audioFileAndPath)
        audio = audio.strip_silence(silence_thresh=-40, silence_len=100) # 去除头尾的静音
        audioPosition = voiceMapSrt[i].start.total_seconds() * 1000

        if i != len(voiceMapSrt) - 1:
            # 检查上这一句的结尾到下一句的开头之间是否有静音，如果没有则需要缩小音频
            audioEndPosition = audioPosition + audio.duration_seconds * 1000 + MIN_GAP_DURATION *1000
            audioNextPosition = voiceMapSrt[i+1].start.total_seconds() * 1000
            if audioNextPosition < audioEndPosition:
                speedUp = (audio.duration_seconds * 1000 + MIN_GAP_DURATION *1000) / (audioNextPosition - audioPosition)
                seconds = audioPosition / 1000.0
                timeStr = str(datetime.timedelta(seconds=seconds))
                if speedUp > MAX_SPEED_UP:
                    # 转换为 HH:MM:SS 格式
                    logStr = f"Warning: The audio {i+1} , at {timeStr} , is too short, speed up is {speedUp}."
                    diagnosisLog.write(logStr)
                
                # 音频如果提速一个略大于1，则speedup函数可能会出现一个错误的音频，所以这里确定最小的speedup为1.01
                if speedUp < MIN_SPEED_UP:
                    logStr = f"Warning: The audio {i+1} , at {timeStr} , speed up {speedUp} is too near to 1.0. Set to {MIN_SPEED_UP} forcibly."
                    diagnosisLog.write(logStr)
                    speedUp = MIN_SPEED_UP
                audio = audio.speedup(playback_speed=speedUp)

        combined = combined.overlay(audio, position=audioPosition)
    
    combined.export(outputAndPath, format="wav")
    return True

def envCheck():
    # 检查环境变量中是否包含 ffmpeg
    # ffmpeg_path = os.environ.get('PATH', '').split(os.pathsep)
    # ffmpeg_found = any('ffmpeg' in path.lower() for path in ffmpeg_path)
    waringMessage = ""

    # print(ffmpeg_found)
    # if not ffmpeg_found:
    #     waringMessage += "未安装ffmpeg，请安装ffmpeg并将其所在目录添加到环境变量PATH中。\n"

    command = "ffmpeg"  # 你可以替换为任何你想检查的命令
    if check_command_exists(command):
        print(f"命令 {command} 存在。")
    else:
        print(f"命令 {command} 不存在。")
        waringMessage += "未安装ffmpeg，请安装ffmpeg并将其所在目录添加到环境变量PATH中。\n"
    
    if waringMessage:
        #root = tk.Tk()
        #root.deiconify()  # 隐藏主窗口
        #messagebox.showwarning("环境依赖警告", waringMessage)
        print("环境依赖警告", waringMessage)
        #root.destroy()  # 销毁主窗口
        return False
    return True


if __name__ == "__main__":

    # if not envCheck():
    #     exit(-1)

    #paramDirPathAndName = input("Please input the path and name of the parameter file (json format): ")
    paramDirPathAndName = "./example/paramDict.json"
    
    #root = tk.Tk()
    #root.deiconify()  # 打开主窗口
    #paramDirPathAndName = filedialog.askopenfilename()  # 打开文件选择对话框
    #root.destroy()  # 关闭主窗口

    # 检查paramDirPathAndName是否存在，是否为json文件
    if not os.path.exists(paramDirPathAndName) or not os.path.isfile(paramDirPathAndName) or not paramDirPathAndName.endswith(".json"):
        print("Please select a valid parameter file.")
        exit(-1)

    # paramDirPathAndName = input("Please input the path and name of the parameter file: ")
    if not os.path.exists(paramDirPathAndName):
        create_param_template(paramDirPathAndName)
        print(f"Parameter file created at {paramDirPathAndName}.")
        print("Please edit the file and run the script again.")
        exit(0)
    
    paramDict = load_param(paramDirPathAndName)
    workPath = paramDict["work path"]
    # videoId = paramDict["video Id"]
    #videoId = input('input youtube vid:')
    videoId = sys.argv[1]
    PROXY = paramDict["proxy"]
    audioRemoveModelNameAndPath = paramDict["audio remove model path"]

    #change_origin_speech = input('输入  y 为 变为中音,   n----> 保持原英音:')
    #burn_en_subtitle = input('输入  y 为 压英文字幕,   n----> 不压英文字幕:')
    change_origin_speech = sys.argv[2]
    burn_en_subtitle = sys.argv[3]

    proxies = None if not PROXY else {
        'http': f"{PROXY}",
        'https': f"{PROXY}",
        'socks5': f"{PROXY}"
    }

    # create the working directory if it does not exist
    if not os.path.exists(workPath):
        os.makedirs(workPath)
        print(f"Directory {workPath} created.")
    
    # 日志
    logFileName = "diagnosis.log"
    diagnosisLog = WarningFile(os.path.join(workPath, logFileName))
    # 执行日志文件的格式为excute_yyyyMMdd_HHmmss.log
    logFileName = "execute_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    executeLog = WarningFile(os.path.join(workPath, logFileName))

    nowString = str(datetime.datetime.now())
    executeLog.write(f"Start at: {nowString}")
    executeLog.write("Params\n" + json.dumps(paramDict, indent=4) + "\n")

    # 下载视频
    #print(paramDict["ffmpeg_srtEnFileNameMergeAndPath"],  '*************************************  4')
    voiceFileName = f"{videoId}.mp4"
    viedoFileNameAndPath = os.path.join(workPath, voiceFileName)
    
    if paramDict["download video"]:
        print(f"Downloading video {videoId} to {viedoFileNameAndPath}")
        try:
            # yt = YouTube(f'https://www.youtube.com/watch?v={videoId}', proxies=proxies, on_progress_callback=on_progress)
            # video  = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').asc().first()
            # video.download(output_path=workPath, filename=voiceFileName)
            # ---------------------------------------------------------------------------------------------------------------------------pytube 的下载经常会遇到aged 等问题下不了，所以改用 yt-dlp 带cookie 下载
            os.system(f"yt-dlp -S res:720,vcodec:h264,fps,res,acodec:m4a -o {viedoFileNameAndPath} https://www.youtube.com/watch?v={videoId}")   #------------------------------------------------------
            # go back to the script directory
            # executeLog.write(f"[WORK o] Download video {videoId} to {viedoFileNameAndPath} whith {video.resolution}.")
            executeLog.write(f"[WORK o] Download video {videoId} to {viedoFileNameAndPath} whith yt-dlp.")
        except Exception as e:
            logStr = f"[WORK x] Error: Program blocked while downloading video {videoId} to {viedoFileNameAndPath}."
            executeLog.write(logStr)
            error_str = traceback.format_exception_only(type(e), e)[-1].strip()
            executeLog.write(error_str)
            sys.exit(-1)
    else:
        logStr = "[WORK -] Skip downloading video."
        executeLog.write(logStr)

    end_time = time.time()  # 记录结束时间
    time_downloaded_video = end_time - start_time  # 计算运行时长
    # try download more high-definition video
    # 需要单独下载最高分辨率视频，因为pytube下载的1080p视频没音频
    voiceFhdFileName = f"{videoId}_fhd.mp4"
    voiceFhdFileNameAndPath = os.path.join(workPath, voiceFhdFileName)
    if paramDict["download fhd video"]:
        try:
            print(f"Try to downloading more high-definition video {videoId} to {voiceFhdFileNameAndPath}")
            # yt = YouTube(f'https://www.youtube.com/watch?v={videoId}', proxies=proxies, on_progress_callback=on_progress)
            # video  = yt.streams.filter(progressive=False, file_extension='mp4').order_by('resolution').desc().first()
            # video.download(output_path=workPath, filename=voiceFhdFileName)
            # executeLog.write(f"[WORK o] Download 1080p high-definition {videoId} to {voiceFhdFileNameAndPath} whith {video.resolution}.")
            os.system(f"yt-dlp -S res:720,vcodec:h264,fps,res,acodec:m4a -o  {voiceFhdFileNameAndPath} https://www.youtube.com/watch?v={videoId}")   #---------------------------------------------------
            executeLog.write(f"[WORK o] Download 1080p high-definition {videoId} to {voiceFhdFileNameAndPath} whith yt-dlp.")
        except:
            # logStr = f"[WORK x] Error: Program blocked while downloading high-definition video {videoId} to {voiceFhdFileNameAndPath} whith {video.resolution}."
            logStr = f"[WORK x] Error: Program blocked while downloading high-definition video {videoId} to {voiceFhdFileNameAndPath} whith yt-dlp."
            executeLog.write(logStr)
            logStr = f"Program will not exit for that the error is not critical."
            executeLog.write(logStr)
    else:
        logStr = "[WORK -] Skip downloading high-definition video."
        executeLog.write(logStr)

    # 打印当前系统时间
    print("Now at: " + str(datetime.datetime.now()))
    end_time = time.time()  # 记录结束时间
    time_downloaded_video_fhd = end_time - start_time  # 计算运行时长
    # 视频转声音提取
    audioFileName = f"{videoId}.wav"
    audioFileNameAndPath = os.path.join(workPath, audioFileName)
    if paramDict["extract audio"]:
        # remove the audio file if it exists
        print(f"Extracting audio from {viedoFileNameAndPath} to {audioFileNameAndPath}")
        try:
            video = VideoFileClip(viedoFileNameAndPath)
            audio = video.audio
            audio.write_audiofile(audioFileNameAndPath)
            executeLog.write(f"[WORK o] Extract audio from {viedoFileNameAndPath} to {audioFileNameAndPath} successfully.")
        except Exception as e:
            logStr = f"[WORK x] Error: Program blocked while extracting audio from {viedoFileNameAndPath} to {audioFileNameAndPath}."
            executeLog.write(logStr)
            error_str = traceback.format_exception_only(type(e), e)[-1].strip()
            executeLog.write(error_str)
            sys.exit(-1)
    else:
        logStr = "[WORK -] Skip extracting audio."
        executeLog.write(logStr)
    end_time = time.time()  # 记录结束时间
    time_video_get_wav = end_time - start_time  # 计算运行时长
    # 去除音频中的音乐
    voiceName = videoId + "_voice.wav"
    voiceNameAndPath = os.path.join(workPath, voiceName)
    insturmentName = videoId + "_insturment.wav"
    insturmentNameAndPath = os.path.join(workPath, insturmentName)
    if paramDict["audio remove"]:
        print(f"Removing music from {audioFileNameAndPath} to {voiceNameAndPath} and {insturmentNameAndPath}")
        try:
            audio_remove(audioFileNameAndPath, voiceNameAndPath, insturmentNameAndPath, audioRemoveModelNameAndPath)
            executeLog.write(f"[WORK o] Remove music from {audioFileNameAndPath} to {voiceNameAndPath} and {insturmentNameAndPath} successfully.")
        except Exception as e:
            logStr = f"[WORK x] Error: Program blocked while removing music from {audioFileNameAndPath} to {voiceNameAndPath} and {insturmentNameAndPath}."
            executeLog.write(logStr)
            error_str = traceback.format_exception_only(type(e), e)[-1].strip()
            executeLog.write(error_str)
            sys.exit(-1)
    else:
        logStr = "[WORK -] Skip removing music."
        executeLog.write(logStr)
    end_time = time.time()  # 记录结束时间
    time_video_remove_music = end_time - start_time  # 计算运行时长      
    # 语音转文字
    srtEnFileName = videoId + "_en.srt"
    srtEnFileNameAndPath = os.path.join(workPath, srtEnFileName)
    if paramDict["audio transcribe"]:
        try:
            print(f"Transcribing audio from {voiceNameAndPath} to {srtEnFileNameAndPath}")
            transcribeAudioEn(voiceNameAndPath, paramDict["audio transcribe model"], "en", srtEnFileNameAndPath)
            executeLog.write(f"[WORK o] Transcribe audio from {voiceNameAndPath} to {srtEnFileNameAndPath} successfully.")
        except Exception as e:
            logStr = f"[WORK x] Error: Program blocked while transcribing audio from {voiceNameAndPath} to {srtEnFileNameAndPath}."
            executeLog.write(logStr)
            error_str = traceback.format_exception_only(type(e), e)[-1].strip()
            executeLog.write(error_str)
            sys.exit(-1)
    else:
        logStr = "[WORK -] Skip transcription."
        executeLog.write(logStr)
    end_time = time.time()  # 记录结束时间
    time_wav2srt = end_time - start_time  # 计算运行时长
    # 字幕语句合并
    srtEnFileNameMerge = videoId + "_en_merge.srt"
    srtEnFileNameMergeAndPath = os.path.join(workPath, srtEnFileNameMerge)
    if paramDict["srt merge"]:
        try:
            print(f"Merging sentences in {srtEnFileNameAndPath} to {srtEnFileNameMergeAndPath}")
            srtSentanceMerge(srtEnFileNameAndPath, srtEnFileNameMergeAndPath)
            executeLog.write(f"[WORK o] Merge sentences in {srtEnFileNameAndPath} to {srtEnFileNameMergeAndPath} successfully.")
        except Exception as e:
            logStr = f"[WORK x] Error: Program blocked while merging sentences in {srtEnFileNameAndPath} to {srtEnFileNameMergeAndPath}."
            executeLog.write(logStr)
            error_str = traceback.format_exception_only(type(e), e)[-1].strip()
            executeLog.write(error_str)
            sys.exit(-1)
    else:
        logStr = "[WORK -] Skip sentence merge."
        executeLog.write(logStr)
    end_time = time.time()  # 记录结束时间
    time_wav2srt_merge = end_time - start_time  # 计算运行时长
    # 英文字幕转文字
    tetEnFileName = videoId + "_en_merge.txt"
    tetEnFileNameAndPath = os.path.join(workPath, tetEnFileName)
    if paramDict["srt merge en to text"]:
        try:
            enText = srt_to_text(srtEnFileNameMergeAndPath)
            print(f"Writing EN text to {tetEnFileNameAndPath}")
            with open(tetEnFileNameAndPath, "w") as file:
                file.write(enText)
            executeLog.write(f"[WORK o] Write EN text to {tetEnFileNameAndPath} successfully.")
        except Exception as e:
            logStr = f"[WORK x] Error: Writing EN text to {tetEnFileNameAndPath} failed."
            executeLog.write(logStr)
            error_str = traceback.format_exception_only(type(e), e)[-1].strip()
            executeLog.write(error_str)
            # 这不是关键步骤，所以不退出程序
            logStr = f"Program will not exit for that the error is not critical."
            executeLog.write(logStr)
    else:
        logStr = "[WORK -] Skip writing EN text."
        executeLog.write(logStr)

    # 字幕翻译
    srtZhFileName = videoId + "_zh_merge.srt"
    srtZhFileNameAndPath = os.path.join(workPath, srtZhFileName)
    if paramDict["srt merge translate"]:
        try:
            print(f"Translating subtitle from {srtEnFileNameMergeAndPath} to {srtZhFileNameAndPath}")
            if paramDict["srt merge translate tool"] == "deepl":
                if paramDict["srt merge translate key"] == "":
                    logStr = "[WORK x] Error: DeepL API key is not provided. Please provide it in the parameter file."
                    executeLog.write(logStr)
                    sys.exit(-1)
                srtFileDeeplTran(srtEnFileNameMergeAndPath, srtZhFileNameAndPath, paramDict["srt merge translate key"])
            else:
                # srtFileGoogleTran(srtEnFileNameMergeAndPath, srtZhFileNameAndPath)
                srtFileGPTtran(srtEnFileNameMergeAndPath, srtZhFileNameAndPath)
                executeLog.write(f"[WORK o] Translate subtitle from {srtEnFileNameMergeAndPath} to {srtZhFileNameAndPath} successfully.")
        except Exception as e:
            logStr = f"[WORK x] Error: Program blocked while translating subtitle from {srtEnFileNameMergeAndPath} to {srtZhFileNameAndPath}."
            executeLog.write(logStr)
            error_str = traceback.format_exception_only(type(e), e)[-1].strip()
            executeLog.write(error_str)
            sys.exit(-1)
    else:
        logStr = "[WORK -] Skip subtitle translation."
        executeLog.write(logStr)

    # 中文字幕转文字
    textZhFileName = videoId + "_zh_merge.txt"
    textZhFileNameAndPath = os.path.join(workPath, textZhFileName)
    if paramDict["srt merge zh to text"]:
        try:
            zhText = srt_to_text(srtZhFileNameAndPath)
            print(f"Writing ZH text to {textZhFileNameAndPath}")
            with open(textZhFileNameAndPath, "w", encoding="utf-8") as file:
                file.write(zhText)
            executeLog.write(f"[WORK o] Write ZH text to {textZhFileNameAndPath} successfully.")
        except Exception as e:
            logStr = f"[WORK x] Error: Writing ZH text to {textZhFileNameAndPath} failed."
            executeLog.write(logStr)
            error_str = traceback.format_exception_only(type(e), e)[-1].strip()
            executeLog.write(error_str)
            # 这不是关键步骤，所以不退出程序
            logStr = f"Program will not exit for that the error is not critical."
            executeLog.write(logStr)
    else:
        logStr = "[WORK -] Skip writing ZH text."
        executeLog.write(logStr)

    # 字幕转语音
    ttsSelect = paramDict["TTS"]
    voiceDir = os.path.join(workPath, videoId + "_zh_source")
    voiceSrcSrtName = "zh.srt"
    voiceSrcSrtNameAndPath = os.path.join(voiceDir, voiceSrcSrtName)
    voiceSrcMapName = "voiceMap.srt"
    voiceSrcMapNameAndPath = os.path.join(voiceDir, voiceSrcMapName)
    if paramDict["srt to voice srouce"]:
        try:
            if ttsSelect == "GPT-SoVITS":
                print(f"Converting subtitle to voice by GPT-SoVITS  in {srtZhFileNameAndPath} to {voiceDir}")
                voiceUrl = paramDict["TTS param"]
                srtToVoice(voiceUrl, srtZhFileNameAndPath, voiceDir)
            else:
                charator = paramDict["TTS param"]
                if charator == "":
                    srtToVoiceEdge(srtZhFileNameAndPath, voiceDir)
                else:
                    srtToVoiceEdge(srtZhFileNameAndPath, voiceDir, charator)
                print(f"Converting subtitle to voice by EdgeTTS in {srtZhFileNameAndPath} to {voiceDir}")
            executeLog.write(f"[WORK o] Convert subtitle to voice in {srtZhFileNameAndPath} to {voiceDir} successfully.")
        except Exception as e:
            logStr = f"[WORK x] Error: Program blocked while converting subtitle to voice in {srtZhFileNameAndPath} to {voiceDir}."
            executeLog.write(logStr)
            error_str = traceback.format_exception_only(type(e), e)[-1].strip()
            executeLog.write(error_str)
            sys.exit(-1)
    else:
        logStr = "[WORK -] Skip voice conversion."
        executeLog.write(logStr)
    
    # 语音合并
    voiceConnectedName = videoId + "_zh.wav"
    voiceConnectedNameAndPath = os.path.join(workPath, voiceConnectedName)
    if paramDict["voice connect"]:
        try:
            print(f"Connecting voice in {voiceDir} to {voiceConnectedNameAndPath}")
            ret = voiceConnect(voiceDir, voiceConnectedNameAndPath)
            if ret == True:
                executeLog.write(f"[WORK o] Connect voice in {voiceDir} to {voiceConnectedNameAndPath} successfully.")
            else:
                executeLog.write(f"[WORK x] Connect voice in {voiceDir} to {voiceConnectedNameAndPath} failed.")
                sys.exit(-1)
        except Exception as e:
            logStr = f"[WORK x] Error: Program blocked while connecting voice in {voiceDir} to {voiceConnectedNameAndPath}."
            executeLog.write(logStr)
            error_str = traceback.format_exception_only(type(e), e)[-1].strip()
            executeLog.write(error_str)
            sys.exit(-1)
    else:
        logStr = "[WORK -] Skip voice connection."
        executeLog.write(logStr)
    
    # 合成后的语音转文字
    srtVoiceFileName = videoId + "_zh.srt"
    srtVoiceFileNameAndPath = os.path.join(workPath, srtVoiceFileName)
    if paramDict["audio zh transcribe"]:
        try:
            print(f"Transcribing audio from {voiceConnectedNameAndPath} to {srtVoiceFileNameAndPath}")
            transcribeAudioZh(voiceConnectedNameAndPath, paramDict["audio zh transcribe model"] ,"zh", srtVoiceFileNameAndPath)
            executeLog.write(f"[WORK o] Transcribe audio from {voiceConnectedNameAndPath} to {srtVoiceFileNameAndPath} successfully.")
        except Exception as e:
            logStr = f"[WORK x] Error: Program blocked while transcribing audio from {voiceConnectedNameAndPath} to {srtVoiceFileNameAndPath}."
            executeLog.write(logStr)
            error_str = traceback.format_exception_only(type(e), e)[-1].strip()
            executeLog.write(error_str)
            sys.exit(-1)
    else:
        logStr = "[WORK -] Skip transcription."
        executeLog.write(logStr)

    # 合成预览视频
    previewVideoName = videoId + "_preview.mp4"
    previewVideoNameAndPath = os.path.join(workPath, previewVideoName)
    if paramDict["video zh preview"]:
        try:
            sourceVideoNameAndPath = ""
            if os.path.exists(voiceFhdFileNameAndPath):
                sourceVideoNameAndPath = voiceFhdFileNameAndPath
            elif os.path.exists(viedoFileNameAndPath):
                print(f"Cannot find high-definition video, use low-definition video {viedoFileNameAndPath} for preview video {previewVideoNameAndPath}")
                sourceVideoNameAndPath = viedoFileNameAndPath
            else:
                logStr = f"[WORK x] Error: Cannot find source video for preview video {previewVideoNameAndPath}."
                executeLog.write(logStr)
                sys.exit(-1)

            print(f"Generating zh preview video in {previewVideoNameAndPath}")
            #zhVideoPreview(sourceVideoNameAndPath, voiceConnectedNameAndPath, insturmentNameAndPath, srtVoiceFileNameAndPath, previewVideoNameAndPath)
            # zhVideoPreview1(sourceVideoNameAndPath, voiceConnectedNameAndPath, insturmentNameAndPath, srtVoiceFileNameAndPath, previewVideoNameAndPath, srtEnFileNameMergeAndPath, srtZhFileNameAndPath)

            # zhVideoPreview_ffmpeg(sourceVideoNameAndPath, voiceConnectedNameAndPath, insturmentNameAndPath, srtVoiceFileNameAndPath, previewVideoNameAndPath, srtEnFileNameMergeAndPath, srtZhFileNameAndPath)
            
            # zhVideoPreview_ffmpeg(sourceVideoNameAndPath, voiceConnectedNameAndPath, insturmentNameAndPath, previewVideoNameAndPath, srtVoiceFileNameAndPath, srtZhFileNameAndPath)  
            zhVideoPreview_ffmpeg(sourceVideoNameAndPath, viedoFileNameAndPath, voiceConnectedNameAndPath, insturmentNameAndPath, previewVideoNameAndPath, srtEnFileNameMergeAndPath, srtZhFileNameAndPath)

            executeLog.write(f"[WORK o] Generate zh preview video in {previewVideoNameAndPath} successfully.")
        except Exception as e:
            logStr = f"[WORK x] Error: Program blocked while generating zh preview video in {previewVideoNameAndPath}."
            executeLog.write(logStr)
            error_str = traceback.format_exception_only(type(e), e)[-1].strip()
            executeLog.write(error_str)
            sys.exit(-1)
    else:
        logStr = "[WORK -] Skip zh preview video."
        executeLog.write(logStr)

    executeLog.write("All done!!")
    print("dir: " + workPath)

    # push any key to exit
    #input("Press any key to exit...")
end_time = time.time()  # 记录结束时间
time_burned_sub = end_time - start_time  # 计算运行时长

print(f"1------------------------复制目录，安装环境：{time_copySRC_setupENV}秒")
print(f"2------------------------下完video：{time_downloaded_video}秒")
print(f"3------------------------下载完fhd：{time_downloaded_video_fhd}秒")
print(f"4------------------------人声分离完：{time_video_get_wav}秒")
print(f"5------------------------背景音乐移除完：{time_video_remove_music}秒")
print(f"6------------------------英语字幕识别完：{time_wav2srt}秒")
print(f"7------------------------英语字幕合并完：{time_wav2srt_merge}秒")
print(f"total------------------------总时长：{time_burned_sub}秒")
